from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import read_config  # noqa: E402
from src.problem1_dataset import (  # noqa: E402
    covarep_frame_times,
    load_covarep_mat,
    load_mfa_reference,
    pool_points_by_word_windows,
)
from src.utils import safe_sample_name  # noqa: E402
from src.vision_35 import AU_INTENSITY, AU_PRESENCE, source_csv_for_sample  # noqa: E402


RTOL = 1.0e-5
ATOL = 1.0e-6


def _normalize_cli(argv: list[str]) -> list[str]:
    """Allow the documented --sample-id "-..." syntax with argparse."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--sample-id" and index + 1 < len(argv):
            result.append(f"--sample-id={argv[index + 1]}")
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export media and independently verify word-level multimodal alignment."
    )
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--pkl", default="outputs/problem1_aligned_features.pkl")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-id")
    selection.add_argument("--sample-index", type=int)
    selection.add_argument("--auto", action="store_true")
    parser.add_argument("--indices", help="Comma-separated real word indices")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--words-per-sample", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Audit output directory. By default, uses "
            "outputs/alignment_audit_mfa or outputs/alignment_audit_whisperx."
        ),
    )
    raw = sys.argv[1:] if argv is None else argv
    return parser.parse_args(_normalize_cli(list(raw)))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def locate_ffmpeg(config: dict[str, Any]) -> str:
    configured = str(config["paths"].get("ffmpeg", "ffmpeg"))
    candidate = Path(configured)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(configured)
    if found:
        return found
    beside_python = Path(sys.executable).resolve().parent / "ffmpeg"
    if beside_python.is_file():
        return str(beside_python)
    raise FileNotFoundError(f"ffmpeg was not found: configured={configured}")


def run_ffmpeg(ffmpeg: str, arguments: list[str]) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), list):
        raise ValueError(f"Unexpected PKL schema: {path}")
    return dataset


def require_file(label: str, path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def alignment_provider(dataset: dict[str, Any]) -> str:
    """Return the timestamp/text branch encoded by the selected PKL."""
    meta = dataset.get("meta", {})
    provider = str(meta.get("alignment_provider", "")).strip().lower()
    if provider in {"mfa", "whisperx"}:
        return provider
    reference = str(meta.get("alignment_reference", "")).lower()
    if "whisperx" in reference:
        return "whisperx"
    if "montreal" in reference or "mfa" in reference:
        return "mfa"
    raise ValueError(
        "Cannot determine alignment provider from PKL metadata. "
        f"alignment_provider={provider!r}, alignment_reference={reference!r}"
    )


def apply_dataset_work_dir(config: dict[str, Any], dataset: dict[str, Any]) -> Path:
    """Select the isolated A/B cache tree recorded by the chosen PKL.

    Remote PKLs record an absolute Linux path.  When a PKL and its caches have
    been copied to another machine, fall back to ``work/<dataset_variant>`` so
    the same inspector remains portable.
    """
    meta = dataset.get("meta", {})
    recorded = str(meta.get("isolated_work_dir", "")).strip()
    if recorded:
        recorded_path = Path(recorded).expanduser()
        if recorded_path.is_dir():
            config["paths"]["work_dir"] = str(recorded_path.resolve())
            return recorded_path.resolve()
    variant = str(meta.get("dataset_variant", "")).strip()
    if variant:
        local_candidate = (PROJECT_ROOT / "work" / variant).resolve()
        if local_candidate.is_dir():
            config["paths"]["work_dir"] = str(local_candidate)
            return local_candidate
    configured = Path(config["paths"]["work_dir"]).expanduser()
    if not configured.is_absolute():
        configured = (PROJECT_ROOT / configured).resolve()
        config["paths"]["work_dir"] = str(configured)
    return configured


def load_whisperx_reference(path: Path, sample_id: str) -> tuple[list[str], np.ndarray]:
    payload = json.loads(require_file("WhisperX JSON", path).read_text(encoding="utf-8"))
    if str(payload.get("sample_id", "")) != sample_id:
        raise ValueError(
            f"WhisperX sample_id mismatch: expected={sample_id!r}, "
            f"actual={payload.get('sample_id')!r}, path={path}"
        )
    words: list[str] = []
    intervals: list[list[float]] = []
    for index, item in enumerate(payload.get("words", [])):
        word = str(item.get("word", "")).strip()
        start = float(item["start"])
        end = float(item["end"])
        if not word or not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(
                f"Invalid WhisperX word at {sample_id}[{index}]: "
                f"word={word!r}, start={start}, end={end}"
            )
        words.append(word)
        intervals.append([start, end])
    times = np.asarray(intervals, dtype=np.float64)
    if times.size == 0:
        times = np.zeros((0, 2), dtype=np.float64)
    return words, times


def load_alignment_reference(
    config: dict[str, Any], dataset: dict[str, Any], sample_id: str
) -> tuple[str, Path, list[str], np.ndarray]:
    provider = alignment_provider(dataset)
    stem = safe_sample_name(sample_id)
    work = Path(config["paths"]["work_dir"])
    if provider == "whisperx":
        path = work / "asr" / "whisperx" / f"{stem}.json"
        words, times = load_whisperx_reference(path, sample_id)
    else:
        path = work / "mfa" / "word_timestamps" / f"{stem}.json"
        words, times = load_mfa_reference(config, sample_id)
    return provider, path, words, times


def locate_inputs(
    config: dict[str, Any], dataset: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Path | None]:
    sample_id = str(sample["sample_id"])
    stem = safe_sample_name(sample_id)
    work = Path(config["paths"]["work_dir"])
    provider = alignment_provider(dataset)
    video = Path(config["paths"]["dataset_root"]) / str(sample["video_id"]) / f"{sample['clip_id']}.mp4"
    if provider == "whisperx":
        timestamp_json = work / "asr" / "whisperx" / f"{stem}.json"
        text_feature_npz = work / "features" / "text_whisperx" / f"{stem}.npz"
    else:
        timestamp_json = work / "mfa" / "word_timestamps" / f"{stem}.json"
        text_feature_npz = work / "features" / "text" / f"{stem}.npz"
    inputs: dict[str, Path | None] = {
        "video": require_file("original MP4", video),
        "wav": require_file("extracted WAV", work / "audio" / f"{stem}.wav"),
        "timestamp_json": require_file(f"{provider} timestamp JSON", timestamp_json),
        "text_feature_npz": require_file(f"{provider} BERT NPZ", text_feature_npz),
        "covarep_mat": work / "audio" / f"{stem}.mat",
        "openface_npz": require_file(
            "OpenFace NPZ", work / "features" / "vision" / f"{stem}.npz"
        ),
        "openface_csv": require_file(
            "OpenFace CSV", source_csv_for_sample(config, sample_id)
        ),
    }
    if not inputs["covarep_mat"].is_file():
        inputs["covarep_mat"] = None
    return inputs


def parse_indices(text: str | None, valid_length: int, limit: int) -> list[int]:
    if text:
        try:
            values = [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid --indices value: {text}") from exc
        if not values:
            raise ValueError("--indices did not contain any indices")
        duplicates = [value for value in values if values.count(value) > 1]
        if duplicates:
            raise ValueError(f"Duplicate word indices: {sorted(set(duplicates))}")
    else:
        values = list(range(min(valid_length, limit)))
    invalid = [value for value in values if value < 0 or value >= valid_length]
    if invalid:
        raise IndexError(f"Indices outside real word range [0,{valid_length - 1}]: {invalid}")
    return values


def representative_indices(sample: dict[str, Any], limit: int, category: str) -> list[int]:
    length = int(sample["valid_length"])
    real = list(range(length))
    if category == "normal":
        preferred = [
            index
            for index in real
            if int(sample["audio_mask"][index]) == 1 and int(sample["vision_mask"][index]) == 1
        ]
    elif category == "vision_partial":
        missing = [index for index in real if int(sample["vision_mask"][index]) == 0]
        present = [index for index in real if int(sample["vision_mask"][index]) == 1]
        preferred = []
        while (missing or present) and len(preferred) < limit:
            if missing:
                preferred.append(missing.pop(0))
            if present and len(preferred) < limit:
                preferred.append(present.pop(0))
    elif category == "truncated" and length:
        preferred = [0, 1, 2, max(0, length - 2), length - 1]
        preferred = list(dict.fromkeys(value for value in preferred if value < length))
    else:
        preferred = real
    return preferred[:limit]


def auto_select(samples: list[dict[str, Any]], number: int) -> list[tuple[str, dict[str, Any]]]:
    if number < 1:
        raise ValueError("--num-samples must be positive")

    def real_mask(sample: dict[str, Any], name: str) -> np.ndarray:
        return np.asarray(sample[name])[: int(sample["valid_length"])]

    predicates = [
        (
            "normal",
            lambda s: not bool(s["truncated"])
            and bool(s.get("mfa_bert_match", True))
            and len(real_mask(s, "audio_mask")) > 0
            and bool(np.all(real_mask(s, "audio_mask") == 1))
            and bool(np.all(real_mask(s, "vision_mask") == 1)),
        ),
        (
            "vision_partial",
            lambda s: int(s["valid_length"]) > 0 and s["vision_status"] == "partial",
        ),
        (
            "vision_missing",
            lambda s: int(s["valid_length"]) > 0 and s["vision_status"] == "missing",
        ),
        (
            "silent",
            lambda s: int(s["valid_length"]) > 0 and s["audio_status"] == "silent",
        ),
        (
            "truncated",
            lambda s: bool(s["truncated"]),
        ),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for category, predicate in predicates:
        match = next(
            (sample for sample in samples if sample["sample_id"] not in used and predicate(sample)),
            None,
        )
        if match is not None:
            selected.append((category, match))
            used.add(str(match["sample_id"]))
        if len(selected) >= number:
            return selected
    for sample in samples:
        if sample["sample_id"] not in used and bool(sample.get("mfa_bert_match", True)):
            selected.append(("additional", sample))
            used.add(str(sample["sample_id"]))
            if len(selected) >= number:
                break
    return selected


def compare_vectors(recomputed: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    recomputed = np.asarray(recomputed, dtype=np.float32)
    stored = np.asarray(stored, dtype=np.float32)
    if recomputed.shape != stored.shape:
        return {
            "match": False,
            "max_abs_error": math.inf,
            "mean_abs_error": math.inf,
            "shape_recomputed": list(recomputed.shape),
            "shape_stored": list(stored.shape),
        }
    difference = np.abs(recomputed.astype(np.float64) - stored.astype(np.float64))
    return {
        "match": bool(np.allclose(recomputed, stored, rtol=RTOL, atol=ATOL)),
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "shape_recomputed": list(recomputed.shape),
        "shape_stored": list(stored.shape),
    }


def extract_audio_segment(wav_path: Path, destination: Path, start: float, end: float) -> dict[str, Any]:
    info = sf.info(wav_path)
    start_frame = max(0, int(round(start * info.samplerate)))
    end_frame = min(info.frames, int(round(end * info.samplerate)))
    if end_frame <= start_frame:
        raise ValueError(f"Empty audio interval after sample conversion: [{start},{end}]")
    signal, sample_rate = sf.read(
        wav_path,
        start=start_frame,
        stop=end_frame,
        always_2d=True,
        dtype="float32",
    )
    sf.write(destination, signal, sample_rate, subtype="PCM_16")
    mono = signal.mean(axis=1, dtype=np.float64)
    return {
        "audio_rms": float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0,
        "audio_max_abs": float(np.max(np.abs(mono))) if len(mono) else 0.0,
        "audio_segment_duration": float(len(signal) / sample_rate),
        "audio_sample_rate": int(sample_rate),
        "audio_start_sample": start_frame,
        "audio_end_sample_exclusive": end_frame,
    }


def export_video_materials(
    ffmpeg: str,
    video_path: Path,
    destination: Path,
    start: float,
    end: float,
    fps: float,
) -> dict[str, float]:
    frame_step = 1.0 / fps if fps > 0 else 1.0 / 25.0
    moments = {
        "frame_start.jpg": start,
        "frame_middle.jpg": (start + end) / 2.0,
        "frame_end.jpg": max(start, end - frame_step),
    }
    for filename, timestamp in moments.items():
        run_ffmpeg(
            ffmpeg,
            ["-ss", f"{timestamp:.9f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(destination / filename)],
        )
    duration = end - start
    run_ffmpeg(
        ffmpeg,
        [
            "-i",
            str(video_path),
            "-ss",
            f"{start:.9f}",
            "-t",
            f"{duration:.9f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(destination / "clip.mp4"),
        ],
    )
    return {
        "frame_start_time": moments["frame_start.jpg"],
        "frame_middle_time": moments["frame_middle.jpg"],
        "frame_end_time": moments["frame_end.jpg"],
        "video_clip_requested_duration": duration,
    }


def load_visual_csv(path: Path, expected_names: list[str]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(value).strip() for value in frame.columns]
    intensity = [value for value in frame.columns if AU_INTENSITY.fullmatch(value)]
    presence = [value for value in frame.columns if AU_PRESENCE.fullmatch(value)]
    names = intensity + presence
    if len(intensity) != 17 or len(presence) != 18:
        raise ValueError(f"OpenFace CSV is not 17+18: AU_r={intensity}, AU_c={presence}")
    if names != expected_names:
        raise ValueError(f"OpenFace feature order differs from PKL metadata: {names}")
    features = frame[names].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"OpenFace CSV contains NaN/Inf in selected features: {path}")
    confidence = frame["confidence"].to_numpy(dtype=np.float32)
    success = frame["success"].to_numpy(dtype=np.float32)
    valid = ((success > 0) & (confidence >= 0.8)).astype(np.uint8)
    return {
        # Match the extraction pipeline: CSV timestamps are cached as float32.
        "times": frame["timestamp"].to_numpy(dtype=np.float32),
        "features": features,
        "valid": valid,
        "feature_names": np.asarray(names, dtype=np.str_),
    }


def create_overview(sample_dir: Path, rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(rows), 4, figsize=(15, max(3.0, 3.0 * len(rows))), squeeze=False, constrained_layout=True
    )
    for row_index, record in enumerate(rows):
        word_dir = sample_dir / f"word_{int(record['word_index']):03d}"
        for column, filename in enumerate(("frame_start.jpg", "frame_middle.jpg", "frame_end.jpg")):
            axes[row_index, column].imshow(mpimg.imread(word_dir / filename))
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(filename.replace("frame_", "").replace(".jpg", ""))
        info = (
            f"Word {record['word_index']}: {record['word']}\n"
            f"[{record['start']:.3f}, {record['end']:.3f}) s\n"
            f"Audio RMS: {record['audio_rms']:.6g}\n"
            f"Masks A/V: {record['audio_mask']}/{record['vision_mask']}\n"
            f"Frames A/V: {record['audio_frame_counts']}/{record['vision_frame_counts']}\n"
            f"Text: {'PASS' if record['text_feature_match'] else 'FAIL'}\n"
            f"Audio: {'PASS' if record['audio_feature_match'] else 'FAIL'}\n"
            f"Vision: {'PASS' if record['vision_feature_match'] else 'FAIL'}"
        )
        axes[row_index, 3].axis("off")
        axes[row_index, 3].text(0.02, 0.95, info, ha="left", va="top", fontsize=11, parse_math=False)
    figure.suptitle(f"Word-level alignment audit: {rows[0]['sample_id']}", fontsize=15, parse_math=False)
    output = sample_dir / "overview.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def create_html(sample_dir: Path, rows: list[dict[str, Any]], overall: bool) -> Path:
    cards: list[str] = []
    for record in rows:
        index = int(record["word_index"])
        relative = f"word_{index:03d}"
        cards.append(
            f"""
<section class="card">
  <h2>Word {index}: {html.escape(str(record['word']))}</h2>
  <p>Time: {record['start']:.6f} - {record['end']:.6f} s</p>
  <div class="frames">
    <figure><img src="{relative}/frame_start.jpg"><figcaption>start</figcaption></figure>
    <figure><img src="{relative}/frame_middle.jpg"><figcaption>middle</figcaption></figure>
    <figure><img src="{relative}/frame_end.jpg"><figcaption>end−1/fps</figcaption></figure>
  </div>
  <div class="players">
    <audio controls src="{relative}/audio.wav"></audio>
    <video controls preload="metadata" src="{relative}/clip.mp4"></video>
  </div>
  <table>
    <tr><th>Text match</th><td>{'PASS' if record['text_feature_match'] else 'FAIL'}</td></tr>
    <tr><th>Audio match</th><td>{'PASS' if record['audio_feature_match'] else 'FAIL'}</td></tr>
    <tr><th>Vision match</th><td>{'PASS' if record['vision_feature_match'] else 'FAIL'}</td></tr>
    <tr><th>Audio frames</th><td>{record['audio_frame_counts']}</td></tr>
    <tr><th>Vision frames</th><td>{record['vision_frame_counts']}</td></tr>
    <tr><th>Audio RMS</th><td>{record['audio_rms']:.9g}</td></tr>
  </table>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alignment audit — {html.escape(str(rows[0]['sample_id']))}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1280px; margin: auto; padding: 20px; color: #202530; }}
.status {{ padding: 12px; background: {'#dff3e4' if overall else '#f8d7da'}; font-weight: bold; }}
.card {{ border: 1px solid #ccd2d8; border-radius: 8px; padding: 16px; margin: 20px 0; }}
.frames {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
figure {{ margin: 0; }} img {{ width: 100%; height: auto; }} figcaption {{ text-align: center; }}
.players {{ display: grid; grid-template-columns: 1fr 2fr; gap: 16px; align-items: center; margin-top: 12px; }}
audio, video {{ width: 100%; max-height: 360px; }} table {{ border-collapse: collapse; margin-top: 12px; }}
th, td {{ border: 1px solid #ccd2d8; padding: 6px 10px; text-align: left; }}
</style></head><body>
<h1>Word-level alignment audit</h1>
<p>Sample: {html.escape(str(rows[0]['sample_id']))}</p>
<p>Alignment provider: {html.escape(str(rows[0]['alignment_provider']))}</p>
<p class="status">Overall: {'PASS' if overall else 'FAIL'}</p>
<p><a href="overview.png">Open overview.png</a> · <a href="report.json">report.json</a> · <a href="report.csv">report.csv</a></p>
{''.join(cards)}
</body></html>"""
    output = sample_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def inspect_sample(
    config: dict[str, Any],
    dataset: dict[str, Any],
    sample: dict[str, Any],
    indices: list[int],
    output_root: Path,
    ffmpeg: str,
    category: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    inputs = locate_inputs(config, dataset, sample)
    provider, reference_path, reference_words, reference_times = load_alignment_reference(
        config, dataset, sample_id
    )
    timestamp_tolerance = 1.0e-5 if provider == "whisperx" else float(
        config.get("problem1", {}).get("timestamp_tolerance", 1.0e-6)
    )
    with np.load(inputs["text_feature_npz"], allow_pickle=False) as bert_file:
        bert = {key: bert_file[key] for key in bert_file.files}

    expected_vision_names = [str(value) for value in dataset["meta"]["vision_feature_names"]]
    vision_frames = load_visual_csv(inputs["openface_csv"], expected_vision_names)

    audio_frames: np.ndarray | None = None
    audio_times: np.ndarray | None = None
    if inputs["covarep_mat"] is not None:
        audio_frames, audio_names, _ = load_covarep_mat(
            inputs["covarep_mat"],
            str(config["problem1"]["covarep_feature_key"]),
            str(config["problem1"]["covarep_names_key"]),
            int(config["problem1"]["covarep_expected_dim"]),
        )
        if audio_names != list(dataset["meta"]["audio_feature_names"]):
            raise ValueError(f"COVAREP feature order mismatch for {sample_id}")
        audio_times = covarep_frame_times(
            inputs["wav"],
            len(audio_frames),
            float(config["problem1"]["covarep_hop_seconds"]),
            float(config["problem1"].get("covarep_first_sample_fraction", 0.5)),
        )

    manifest = pd.read_csv(
        Path(config["paths"]["work_dir"]) / config["output"]["manifest_file"],
        dtype={"sample_id": str},
        encoding="utf-8-sig",
    )
    manifest_row = manifest.loc[manifest["sample_id"] == sample_id]
    if len(manifest_row) != 1:
        raise ValueError(f"Manifest lookup for {sample_id} returned {len(manifest_row)} rows")
    fps = float(manifest_row.iloc[0]["fps"])

    rows: list[dict[str, Any]] = []
    for index in indices:
        if int(sample["sequence_mask"][index]) != 1:
            raise ValueError(f"Requested position is padding: {sample_id} index={index}")
        word = str(sample["words"][index])
        start, end = (float(value) for value in sample["timestamps"][index])
        if index >= len(reference_words):
            raise IndexError(
                f"{provider} reference lacks requested word index {index} for {sample_id}"
            )
        timestamp_valid = bool(
            start >= 0
            and end > start
            and end <= float(sample["duration"]) + 0.25
            and np.allclose(
                [start, end], reference_times[index], rtol=0, atol=timestamp_tolerance
            )
        )

        word_dir = sample_dir / f"word_{index:03d}"
        word_dir.mkdir(parents=True, exist_ok=True)
        audio_stats = extract_audio_segment(inputs["wav"], word_dir / "audio.wav", start, end)
        video_stats = export_video_materials(
            ffmpeg, inputs["video"], word_dir, start, end, fps
        )

        bert_words = [str(value) for value in bert["words"].tolist()]
        word_match = bool(
            index < len(bert_words)
            and reference_words[index] == bert_words[index] == word
        )
        text_comparison = compare_vectors(bert["features"][index], sample["text"][index])
        text_match = word_match and bool(text_comparison["match"])

        stored_audio_count = int(sample["audio_frame_counts"][index])
        if audio_frames is not None and audio_times is not None:
            audio_pooled, audio_mask, audio_counts = pool_points_by_word_windows(
                audio_times,
                audio_frames,
                np.asarray([reference_times[index]], dtype=np.float64),
            )
            audio_recomputed = audio_pooled[0]
            selected_audio_count = int(audio_counts[0])
            recomputed_audio_mask = int(audio_mask[0])
            audio_check_mode = "covarep_repool"
        else:
            if int(sample["audio_mask"][index]) != 0:
                raise FileNotFoundError(
                    f"COVAREP MAT missing but audio_mask=1: {sample_id} index={index}"
                )
            audio_recomputed = np.zeros_like(sample["audio"][index], dtype=np.float32)
            selected_audio_count = 0
            recomputed_audio_mask = 0
            audio_check_mode = "missing_modality_zero_check"
        audio_comparison = compare_vectors(audio_recomputed, sample["audio"][index])
        audio_count_match = selected_audio_count == stored_audio_count
        audio_mask_match = recomputed_audio_mask == int(sample["audio_mask"][index])
        audio_match = bool(audio_comparison["match"] and audio_count_match and audio_mask_match)

        vision_pooled, vision_mask, vision_counts = pool_points_by_word_windows(
            vision_frames["times"],
            vision_frames["features"],
            np.asarray([reference_times[index]], dtype=np.float64),
            vision_frames["valid"],
        )
        vision_recomputed = vision_pooled[0]
        selected_vision_count = int(vision_counts[0])
        recomputed_vision_mask = int(vision_mask[0])
        stored_vision_count = int(sample["vision_frame_counts"][index])
        vision_comparison = compare_vectors(vision_recomputed, sample["vision"][index])
        vision_count_match = selected_vision_count == stored_vision_count
        vision_mask_match = recomputed_vision_mask == int(sample["vision_mask"][index])
        vision_match = bool(vision_comparison["match"] and vision_count_match and vision_mask_match)

        frame_count_consistent = bool(
            audio_count_match and audio_mask_match and vision_count_match and vision_mask_match
        )
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "category": category,
            "alignment_provider": provider,
            "alignment_reference_file": str(reference_path),
            "word_index": index,
            "word": word,
            "reference_word": reference_words[index],
            "feature_word": bert_words[index] if index < len(bert_words) else None,
            "start": start,
            "end": end,
            "duration": end - start,
            "sequence_mask": int(sample["sequence_mask"][index]),
            "text_mask": int(sample["text_mask"][index]),
            "audio_mask": int(sample["audio_mask"][index]),
            "vision_mask": int(sample["vision_mask"][index]),
            "audio_frame_counts": stored_audio_count,
            "vision_frame_counts": stored_vision_count,
            "audio_recomputed_frame_count": selected_audio_count,
            "vision_recomputed_frame_count": selected_vision_count,
            "timestamp_valid": timestamp_valid,
            "word_sequence_match": word_match,
            "text_feature_match": text_match,
            "text_max_abs_error": text_comparison["max_abs_error"],
            "text_mean_abs_error": text_comparison["mean_abs_error"],
            "audio_feature_match": audio_match,
            "audio_max_abs_error": audio_comparison["max_abs_error"],
            "audio_mean_abs_error": audio_comparison["mean_abs_error"],
            "audio_count_match": audio_count_match,
            "audio_mask_match": audio_mask_match,
            "audio_check_mode": audio_check_mode,
            "vision_feature_match": vision_match,
            "vision_max_abs_error": vision_comparison["max_abs_error"],
            "vision_mean_abs_error": vision_comparison["mean_abs_error"],
            "vision_count_match": vision_count_match,
            "vision_mask_match": vision_mask_match,
            "frame_count_consistent": frame_count_consistent,
            "overall_pass": bool(
                timestamp_valid
                and int(sample["sequence_mask"][index]) == 1
                and text_match
                and audio_match
                and vision_match
                and frame_count_consistent
            ),
            **audio_stats,
            **video_stats,
            "source_files": {key: str(value) if value is not None else None for key, value in inputs.items()},
            "text_stored": np.asarray(sample["text"][index]),
            "text_reloaded": np.asarray(bert["features"][index]),
            "audio_stored": np.asarray(sample["audio"][index]),
            "audio_recomputed": audio_recomputed,
            "vision_stored": np.asarray(sample["vision"][index]),
            "vision_recomputed": vision_recomputed,
            "vision_feature_names": expected_vision_names,
        }

        text_lines = [
            f"sample_id: {sample_id}",
            f"alignment_provider: {provider}",
            f"word_index: {index}",
            f"word: {word}",
            f"start: {start:.9f}",
            f"end: {end:.9f}",
            f"duration: {end - start:.9f}",
            f"sequence_mask: {record['sequence_mask']}",
            f"text_mask: {record['text_mask']}",
            f"audio_mask: {record['audio_mask']}",
            f"vision_mask: {record['vision_mask']}",
            f"audio_frame_counts: {record['audio_frame_counts']}",
            f"vision_frame_counts: {record['vision_frame_counts']}",
        ]
        (word_dir / "text.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        (word_dir / "metadata.json").write_text(
            json.dumps(json_ready(record), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        rows.append(record)

    summary_rows = [
        {
            key: record[key]
            for key in (
                "word_index",
                "word",
                "start",
                "end",
                "sequence_mask",
                "audio_mask",
                "vision_mask",
                "audio_frame_counts",
                "vision_frame_counts",
                "audio_recomputed_frame_count",
                "vision_recomputed_frame_count",
                "text_feature_match",
                "audio_feature_match",
                "vision_feature_match",
                "text_max_abs_error",
                "audio_max_abs_error",
                "vision_max_abs_error",
                "audio_rms",
                "timestamp_valid",
                "frame_count_consistent",
                "overall_pass",
            )
        }
        for record in rows
    ]
    overall = bool(all(record["overall_pass"] for record in rows))
    report = {
        "sample_id": sample_id,
        "category": category,
        "alignment_provider": provider,
        "alignment_reference": str(dataset.get("meta", {}).get("alignment_reference", "")),
        "word_positions_checked": indices,
        "text_numeric_match": f"{sum(bool(r['text_feature_match']) for r in rows)}/{len(rows)}",
        "audio_numeric_match": f"{sum(bool(r['audio_feature_match']) for r in rows)}/{len(rows)}",
        "vision_numeric_match": f"{sum(bool(r['vision_feature_match']) for r in rows)}/{len(rows)}",
        "timestamp_validity": bool(all(bool(r["timestamp_valid"]) for r in rows)),
        "frame_count_consistency": bool(all(bool(r["frame_count_consistent"]) for r in rows)),
        "overall_pass": overall,
        "max_errors": {
            "text": max(float(r["text_max_abs_error"]) for r in rows),
            "audio": max(float(r["audio_max_abs_error"]) for r in rows),
            "vision": max(float(r["vision_max_abs_error"]) for r in rows),
        },
        "source_files": {key: str(value) if value is not None else None for key, value in inputs.items()},
        "rows": summary_rows,
    }
    (sample_dir / "report.json").write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (sample_dir / "report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    create_overview(sample_dir, rows)
    create_html(sample_dir, rows, overall)
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"Sample: {report['sample_id']}")
    print(f"Alignment provider: {report['alignment_provider']}")
    print(f"Word positions checked: {report['word_positions_checked']}")
    print(f"Text numeric match: {report['text_numeric_match']}")
    print(f"Audio numeric match: {report['audio_numeric_match']}")
    print(f"Vision numeric match: {report['vision_numeric_match']}")
    print(f"Timestamp validity: {'PASS' if report['timestamp_validity'] else 'FAIL'}")
    print(f"Frame-count consistency: {'PASS' if report['frame_count_consistency'] else 'FAIL'}")
    print(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print()


def main() -> None:
    args = parse_args()
    config = read_config(resolve_project_path(args.config))
    dataset = load_dataset(resolve_project_path(args.pkl))
    apply_dataset_work_dir(config, dataset)
    samples = dataset["samples"]
    provider = alignment_provider(dataset)
    default_output = f"outputs/alignment_audit_{provider}"
    output_root = resolve_project_path(args.output_dir or default_output)
    output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = locate_ffmpeg(config)

    selected: list[tuple[str, dict[str, Any]]]
    if args.sample_id is not None:
        matches = [sample for sample in samples if sample["sample_id"] == args.sample_id]
        if len(matches) != 1:
            raise KeyError(f"sample_id lookup returned {len(matches)} matches: {args.sample_id}")
        selected = [("specified", matches[0])]
    elif args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise IndexError(f"sample-index outside [0,{len(samples) - 1}]: {args.sample_index}")
        selected = [("specified", samples[args.sample_index])]
    else:
        selected = auto_select(samples, args.num_samples)

    reports: list[dict[str, Any]] = []
    for category, sample in selected:
        if args.indices:
            indices = parse_indices(args.indices, int(sample["valid_length"]), args.words_per_sample)
        else:
            indices = representative_indices(sample, args.words_per_sample, category)
        if not indices:
            raise ValueError(
                f"Sample {sample['sample_id']} has no real word intervals to inspect "
                f"in the {alignment_provider(dataset)} alignment branch."
            )
        report = inspect_sample(
            config, dataset, sample, indices, output_root, ffmpeg, category
        )
        print_report(report)
        reports.append(report)

    aggregate = {
        "samples_checked": len(reports),
        "sample_ids": [report["sample_id"] for report in reports],
        "overall_pass": bool(all(report["overall_pass"] for report in reports)),
        "reports": [str(output_root / report["sample_id"] / "report.json") for report in reports],
    }
    (output_root / "latest_run.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not aggregate["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
