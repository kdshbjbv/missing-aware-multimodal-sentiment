"""Build an independent WhisperX-word-aligned multimodal Problem-1 dataset.

This entry point deliberately does not call or overwrite the official-text/MFA
pipeline.  It uses WhisperX ASR words and timestamps, creates a separate BERT
cache for those words, and pools the existing COVAREP/OpenFace frame features
over the WhisperX word intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import read_config  # noqa: E402
from src.extract_text_bert import BertWordExtractor  # noqa: E402
from src.prepare_manifest import load_manifest  # noqa: E402
from src.problem1_dataset import (  # noqa: E402
    covarep_frame_times,
    load_covarep_mat,
    pool_points_by_word_windows,
    wav_statistics,
)
from src.utils import check_finite, package_version, safe_sample_name  # noqa: E402
from src.vision_35 import audit_openface_au_fields, extract_visual35_from_csv  # noqa: E402


DATASET_FILENAME = "whisperx_aligned.pkl"
SUMMARY_FILENAME = "whisperx_feature_summary.csv"
VALIDATION_FILENAME = "problem1_whisperx_validation_report.json"
TEXT_CACHE_SUBDIR = Path("features") / "text_whisperx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--whisperx-dir", default="work/asr/whisperx")
    parser.add_argument("--text-cache-dir", default="work/features/text_whisperx")
    parser.add_argument("--output", default=f"outputs/{DATASET_FILENAME}")
    parser.add_argument("--summary", default=f"outputs/{SUMMARY_FILENAME}")
    parser.add_argument("--validation-report", default=f"outputs/{VALIDATION_FILENAME}")
    parser.add_argument("--sample-id", action="append", help="Build selected IDs only")
    parser.add_argument("--force-text", action="store_true", help="Recompute WhisperX BERT cache")
    parser.add_argument("--check-only", action="store_true", help="Build and validate without writing outputs")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _pad_matrix(values: np.ndarray, max_len: int, width: int) -> np.ndarray:
    result = np.zeros((max_len, width), dtype=np.float32)
    length = min(len(values), max_len)
    if length:
        result[:length] = np.asarray(values[:length], dtype=np.float32)
    return result


def _pad_vector(values: np.ndarray, max_len: int, dtype: Any) -> np.ndarray:
    result = np.zeros(max_len, dtype=dtype)
    length = min(len(values), max_len)
    if length:
        result[:length] = np.asarray(values[:length], dtype=dtype)
    return result


def _scalar_text(value: np.ndarray | str) -> str:
    return str(np.asarray(value).item())


def load_whisperx_reference(path: Path, expected_sample_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_id = str(payload.get("sample_id", ""))
    if sample_id != expected_sample_id:
        raise ValueError(
            f"WhisperX sample_id mismatch: file={path}, expected={expected_sample_id!r}, actual={sample_id!r}"
        )
    status = str(payload.get("status", "unknown"))
    if status == "error":
        raise RuntimeError(f"WhisperX extraction failed for {sample_id}: {payload.get('error', 'unknown error')}")

    raw_items = payload.get("words", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"WhisperX words is not a list: {path}")
    words: list[str] = []
    intervals: list[list[float]] = []
    scores: list[float] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"WhisperX word item {index} is not an object: {path}")
        word = str(item.get("word", "")).strip()
        start = float(item["start"])
        end = float(item["end"])
        if not word or not np.isfinite(start) or not np.isfinite(end) or end <= start or start < 0:
            raise ValueError(
                f"Invalid WhisperX word at {sample_id}[{index}]: word={word!r}, start={start}, end={end}"
            )
        words.append(word)
        intervals.append([start, end])
        score = item.get("score")
        scores.append(float(score) if score is not None and np.isfinite(float(score)) else np.nan)

    timestamps = np.asarray(intervals, dtype=np.float64)
    if timestamps.size == 0:
        timestamps = np.zeros((0, 2), dtype=np.float64)
    if len(timestamps) > 1 and np.any(timestamps[1:, 0] < timestamps[:-1, 1] - 1.0e-6):
        raise ValueError(f"Overlapping/out-of-order WhisperX word intervals: {sample_id}")
    return {
        "payload": payload,
        "status": status,
        "words": words,
        "timestamps": timestamps,
        "scores": np.asarray(scores, dtype=np.float32),
    }


def _text_cache_matches(
    cached: dict[str, np.ndarray], words: list[str], timestamps: np.ndarray, model_name: str
) -> bool:
    cached_words = [str(value) for value in cached.get("words", np.asarray([])).tolist()]
    cached_times = np.asarray(cached.get("timestamps", np.zeros((0, 2))), dtype=np.float64)
    cached_model = _scalar_text(cached.get("extractor", np.asarray("")))
    return (
        cached_words == words
        and cached_times.shape == timestamps.shape
        and np.allclose(cached_times, timestamps, atol=1.0e-7, rtol=0)
        and cached_model == model_name
        and np.asarray(cached.get("features", np.zeros((0, 0)))).shape == (len(words), 768)
    )


def prepare_whisperx_text_feature(
    destination: Path,
    words: list[str],
    timestamps: np.ndarray,
    extractor: BertWordExtractor,
    force: bool,
) -> dict[str, np.ndarray]:
    if destination.is_file() and not force:
        cached = _load_npz(destination)
        if _text_cache_matches(cached, words, timestamps, extractor.model_name):
            return cached

    features = extractor.encode_words(words)
    if features.shape != (len(words), extractor.hidden_size):
        raise ValueError(f"Unexpected WhisperX BERT shape {features.shape} for {destination.stem}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        words=np.asarray(words, dtype=np.str_),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        features=np.asarray(features, dtype=np.float32),
        feature_names=np.asarray([f"bert_{index:03d}" for index in range(extractor.hidden_size)]),
        extractor=np.asarray(extractor.model_name, dtype=np.str_),
        timestamp_provider=np.asarray("WhisperX", dtype=np.str_),
    )
    return _load_npz(destination)


def _select_feature_names(variants: Iterable[list[str]], label: str) -> list[str]:
    values = [list(value) for value in variants if value]
    if not values:
        return []
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Inconsistent {label} feature-name order")
    return values[0]


def build_one_sample(
    config: dict[str, Any],
    row: pd.Series,
    whisperx_dir: Path,
    text_cache_dir: Path,
    extractor: BertWordExtractor,
    vision_audit: dict[str, Any],
    force_text: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    sample_id = str(row["sample_id"])
    stem = safe_sample_name(sample_id)
    max_len = int(config["project"]["max_len"])
    settings = config["problem1"]
    text_dim = int(settings["text_expected_dim"])
    audio_dim = int(settings["covarep_expected_dim"])
    vision_dim = int(settings["vision_expected_dim"])
    vision_legacy_dim = int(settings["vision_legacy_dim"])
    work = Path(config["paths"]["work_dir"])

    reference = load_whisperx_reference(whisperx_dir / f"{stem}.json", sample_id)
    all_words = reference["words"]
    all_timestamps = reference["timestamps"]
    original_word_count = len(all_words)
    valid_length = min(original_word_count, max_len)
    words = all_words[:valid_length]
    timestamps = all_timestamps[:valid_length]
    truncated = original_word_count > max_len

    text_data = prepare_whisperx_text_feature(
        text_cache_dir / f"{stem}.npz", words, timestamps, extractor, force_text
    )
    text_raw = np.asarray(text_data["features"], dtype=np.float32)
    if text_raw.shape != (valid_length, text_dim):
        raise ValueError(f"{sample_id}: WhisperX BERT shape {text_raw.shape} != {(valid_length, text_dim)}")

    text = _pad_matrix(text_raw, max_len, text_dim)
    sequence_mask = np.zeros(max_len, dtype=np.uint8)
    sequence_mask[:valid_length] = 1
    text_mask = sequence_mask.copy()
    padded_words = words + [""] * (max_len - valid_length)
    padded_timestamps = np.zeros((max_len, 2), dtype=np.float32)
    if valid_length:
        padded_timestamps[:valid_length] = timestamps.astype(np.float32)

    wav_path = work / "audio" / f"{stem}.wav"
    mat_path = work / "audio" / f"{stem}.mat"
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    audio_names: list[str] = []
    covarep_keys: list[str] = []
    if mat_path.is_file():
        audio_frames, audio_names, covarep_keys = load_covarep_mat(
            mat_path,
            str(settings["covarep_feature_key"]),
            str(settings["covarep_names_key"]),
            audio_dim,
        )
        audio_times = covarep_frame_times(
            wav_path,
            len(audio_frames),
            float(settings["covarep_hop_seconds"]),
            float(settings.get("covarep_first_sample_fraction", 0.5)),
        )
        audio_raw, audio_mask_raw, audio_counts_raw = pool_points_by_word_windows(
            audio_times, audio_frames, timestamps
        )
        audio = _pad_matrix(audio_raw, max_len, audio_dim)
        audio_mask = _pad_vector(audio_mask_raw, max_len, np.uint8)
        audio_counts = _pad_vector(audio_counts_raw, max_len, np.int32)
        audio_status = "covarep" if valid_length else "no_word_windows"
        audio_available = True
        silence_stats: dict[str, float | int] | None = None
    else:
        silence_stats = wav_statistics(wav_path)
        threshold = float(settings.get("silent_max_abs_threshold", 1.0e-12))
        is_silent = float(silence_stats["max_abs"]) <= threshold
        audio = np.zeros((max_len, audio_dim), dtype=np.float32)
        audio_mask = np.zeros(max_len, dtype=np.uint8)
        audio_counts = np.zeros(max_len, dtype=np.int32)
        audio_status = "silent" if is_silent else "covarep_missing"
        audio_available = False

    legacy_frames = _load_npz(work / "features" / "vision" / f"{stem}.npz")
    vision22_raw, _, _ = pool_points_by_word_windows(
        legacy_frames["times"], legacy_frames["features"], timestamps, legacy_frames.get("valid")
    )
    if vision22_raw.shape != (valid_length, vision_legacy_dim):
        raise ValueError(
            f"{sample_id}: legacy OpenFace shape {vision22_raw.shape} != {(valid_length, vision_legacy_dim)}"
        )
    vision_22 = _pad_matrix(vision22_raw, max_len, vision_legacy_dim)
    legacy_names = [str(value) for value in legacy_frames["feature_names"].tolist()]

    vision_frames = extract_visual35_from_csv(config, sample_id, vision_audit)
    vision_raw, vision_mask_raw, vision_counts_raw = pool_points_by_word_windows(
        vision_frames["times"], vision_frames["features"], timestamps, vision_frames.get("valid")
    )
    vision = _pad_matrix(vision_raw, max_len, vision_dim)
    vision_mask = _pad_vector(vision_mask_raw, max_len, np.uint8)
    vision_counts = _pad_vector(vision_counts_raw, max_len, np.int32)
    if valid_length == 0 or int(vision_mask[:valid_length].sum()) == 0:
        vision_status = "missing"
    elif int(vision_mask[:valid_length].sum()) < valid_length:
        vision_status = "partial"
    else:
        vision_status = "available"

    for name, values in (("text", text), ("audio", audio), ("vision", vision), ("vision_22", vision_22)):
        check_finite(f"{sample_id} {name}", values)

    payload = reference["payload"]
    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "video_id": str(row["video_id"]),
        "clip_id": str(row["clip_id"]),
        "raw_text": str(row["raw_text"]),
        "asr_text": str(payload.get("text", "")),
        "words": padded_words,
        "timestamps": padded_timestamps,
        "text": text,
        "audio": audio,
        "vision": vision,
        "vision_22": vision_22,
        "sequence_mask": sequence_mask,
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "vision_mask": vision_mask,
        "audio_frame_counts": audio_counts,
        "vision_frame_counts": vision_counts,
        "valid_length": valid_length,
        "original_word_count": original_word_count,
        "duration": float(row["duration_s"]),
        "truncated": truncated,
        "audio_status": audio_status,
        "audio_available": audio_available,
        "vision_status": vision_status,
        "alignment_status": "success" if valid_length else str(reference["status"]),
        "alignment_provider": "whisperx",
        "whisperx_status": str(reference["status"]),
        "whisperx_model": payload.get("model"),
        "whisperx_alignment_model": payload.get("alignment_model"),
        "regression_label": float(row["regression_label"]),
        "classification_label": int(row["classification_label"]),
        "annotation": str(row["annotation"]),
    }
    if silence_stats is not None:
        sample["silence_check"] = silence_stats

    summary = {
        "sample_id": sample_id,
        "video_id": sample["video_id"],
        "clip_id": sample["clip_id"],
        "duration": sample["duration"],
        "whisperx_status": sample["whisperx_status"],
        "original_word_count": original_word_count,
        "valid_length": valid_length,
        "truncated": truncated,
        "text_shape": str(tuple(text.shape)),
        "audio_shape": str(tuple(audio.shape)),
        "vision_shape": str(tuple(vision.shape)),
        "vision_22_shape": str(tuple(vision_22.shape)),
        "text_valid_windows": int(text_mask[:valid_length].sum()),
        "audio_valid_windows": int(audio_mask[:valid_length].sum()),
        "vision_valid_windows": int(vision_mask[:valid_length].sum()),
        "audio_status": audio_status,
        "vision_status": vision_status,
        "alignment_status": sample["alignment_status"],
        "regression_label": sample["regression_label"],
        "classification_label": sample["classification_label"],
        "annotation": sample["annotation"],
    }
    return sample, summary, audio_names, legacy_names


def build_dataset(
    config: dict[str, Any],
    whisperx_dir: Path,
    text_cache_dir: Path,
    sample_ids: list[str] | None,
    force_text: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = load_manifest(config)
    if sample_ids:
        requested = list(dict.fromkeys(sample_ids))
        missing = sorted(set(requested) - set(manifest["sample_id"].astype(str)))
        if missing:
            raise KeyError(f"Unknown sample IDs: {missing}")
        order = {sample_id: index for index, sample_id in enumerate(requested)}
        manifest = manifest[manifest["sample_id"].isin(requested)].copy()
        manifest["_order"] = manifest["sample_id"].map(order)
        manifest = manifest.sort_values("_order")
    else:
        manifest = manifest.sort_values("feature_index")

    ids = manifest["sample_id"].astype(str).tolist()
    vision_audit = audit_openface_au_fields(config, ids)
    extractor = BertWordExtractor(config)
    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    audio_name_variants: list[list[str]] = []
    legacy_name_variants: list[list[str]] = []
    covarep_keys: set[str] = set()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        sample, summary, audio_names, legacy_names = build_one_sample(
            config,
            row,
            whisperx_dir,
            text_cache_dir,
            extractor,
            vision_audit,
            force_text,
        )
        samples.append(sample)
        summaries.append(summary)
        if audio_names:
            audio_name_variants.append(audio_names)
        legacy_name_variants.append(legacy_names)
        print(
            f"[{index:03d}/{len(manifest):03d}] {sample['sample_id']} "
            f"status={sample['whisperx_status']} words={sample['valid_length']} "
            f"A={int(sample['audio_mask'].sum())} V={int(sample['vision_mask'].sum())}",
            flush=True,
        )

    # MAT public keys are stable in the official COVAREP exports.  Recover
    # them from one readable file without retaining scipy objects in samples.
    for sample in samples:
        mat_path = Path(config["paths"]["work_dir"]) / "audio" / f"{safe_sample_name(sample['sample_id'])}.mat"
        if mat_path.is_file():
            _, _, keys = load_covarep_mat(
                mat_path,
                str(config["problem1"]["covarep_feature_key"]),
                str(config["problem1"]["covarep_names_key"]),
                int(config["problem1"]["covarep_expected_dim"]),
            )
            covarep_keys.update(keys)

    model_names = sorted({str(sample["whisperx_model"]) for sample in samples if sample["whisperx_model"]})
    align_models = sorted(
        {str(sample["whisperx_alignment_model"]) for sample in samples if sample["whisperx_alignment_model"]}
    )
    mapping = {str(key): int(value) for key, value in config["data"]["classification_mapping"].items()}
    dataset = {
        "samples": samples,
        "meta": {
            "schema_version": "whisperx-word-aligned-1.0",
            "dataset_variant": "whisperx_asr",
            "num_samples": len(samples),
            "max_len": int(config["project"]["max_len"]),
            "alignment_unit": "word",
            "alignment_provider": "whisperx",
            "alignment_reference": "WhisperX ASR word timestamps",
            "whisperx_models": model_names,
            "whisperx_alignment_models": align_models,
            "whisperx_status_counts": dict(Counter(str(s["whisperx_status"]) for s in samples)),
            "official_text_role": "reference only; not used for ASR words, timestamps, or BERT features",
            "label_source": "official Excel labels via manifest; unchanged",
            "text_extractor": extractor.model_name,
            "text_input": "WhisperX ASR words",
            "text_dim": int(config["problem1"]["text_expected_dim"]),
            "text_cache_dir": str(text_cache_dir),
            "audio_extractor": "COVAREP",
            "audio_dim": int(config["problem1"]["covarep_expected_dim"]),
            "audio_feature_names": _select_feature_names(audio_name_variants, "COVAREP"),
            "covarep_mat_keys": sorted(covarep_keys),
            "covarep_feature_key": str(config["problem1"]["covarep_feature_key"]),
            "covarep_names_key": str(config["problem1"]["covarep_names_key"]),
            "covarep_hop_seconds": float(config["problem1"]["covarep_hop_seconds"]),
            "covarep_timestamp_rule": "(round(0.005*fs)-1)/fs + k*round(0.01*fs)/fs; k=0..T-1",
            "vision_extractor": "OpenFace 2.2",
            "vision_dim": int(config["problem1"]["vision_expected_dim"]),
            "vision_definition": "17 AU intensity + 18 AU presence",
            "vision_feature_names": list(vision_audit["feature_names"]),
            "vision_is_facet": False,
            "vision_22_dim": int(config["problem1"]["vision_legacy_dim"]),
            "vision_22_definition": "17 AU intensity + 3 head pose + 2 gaze angle",
            "vision_22_feature_names": _select_feature_names(legacy_name_variants, "legacy OpenFace"),
            "openface_csv_schema_consistent": bool(vision_audit["schema_consistent"]),
            "openface_csv_count": int(vision_audit["num_csv"]),
            "padding_strategy": "zero padding; empty word; timestamp [0,0]",
            "truncation_strategy": "first 50 WhisperX words",
            "classification_mapping": mapping,
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "torch", "transformers", "scipy", "soundfile")
            },
            "notes": "Independent WhisperX-ASR word-level representation; does not replace the official-text/MFA dataset.",
        },
    }
    return dataset, pd.DataFrame.from_records(summaries)


def validate_dataset(dataset: dict[str, Any], expected_ids: list[str]) -> dict[str, Any]:
    meta = dataset["meta"]
    samples = dataset["samples"]
    max_len = int(meta["max_len"])
    errors: list[str] = []
    sample_results: list[dict[str, Any]] = []
    if [str(sample["sample_id"]) for sample in samples] != expected_ids:
        errors.append("Sample IDs/order do not match the selected manifest")
    if len(set(expected_ids)) != len(expected_ids):
        errors.append("Duplicate sample IDs")

    for sample in samples:
        sid = str(sample["sample_id"])
        local: list[str] = []
        length = int(sample["valid_length"])
        expected_shapes = {
            "timestamps": (max_len, 2),
            "text": (max_len, int(meta["text_dim"])),
            "audio": (max_len, int(meta["audio_dim"])),
            "vision": (max_len, int(meta["vision_dim"])),
            "vision_22": (max_len, int(meta["vision_22_dim"])),
        }
        for key, shape in expected_shapes.items():
            array = np.asarray(sample[key])
            if array.shape != shape:
                local.append(f"{key}.shape={array.shape}, expected={shape}")
            elif not np.isfinite(array).all():
                local.append(f"{key} has NaN/Inf")
        for key in ("sequence_mask", "text_mask", "audio_mask", "vision_mask"):
            array = np.asarray(sample[key])
            if array.shape != (max_len,) or not set(np.unique(array)).issubset({0, 1}):
                local.append(f"invalid {key}")
        expected_sequence = np.zeros(max_len, dtype=np.uint8)
        expected_sequence[:length] = 1
        if not np.array_equal(sample["sequence_mask"], expected_sequence):
            local.append("sequence_mask does not encode valid_length")
        if not np.array_equal(sample["text_mask"], expected_sequence):
            local.append("text_mask does not match sequence_mask")
        if any(str(word) for word in sample["words"][length:]):
            local.append("non-empty padding word")
        if not np.allclose(sample["timestamps"][length:], 0):
            local.append("non-zero timestamp padding")
        for feature_key, mask_key in (("text", "text_mask"), ("audio", "audio_mask"), ("vision", "vision_mask")):
            values = np.asarray(sample[feature_key])
            mask = np.asarray(sample[mask_key]).astype(bool)
            if not np.allclose(values[~mask], 0):
                local.append(f"{feature_key} is non-zero where {mask_key}=0")
        for mask_key, count_key in (("audio_mask", "audio_frame_counts"), ("vision_mask", "vision_frame_counts")):
            mask = np.asarray(sample[mask_key])
            counts = np.asarray(sample[count_key])
            if counts.shape != (max_len,) or np.any(counts < 0):
                local.append(f"invalid {count_key}")
            elif not np.array_equal(mask, (counts > 0).astype(np.uint8)):
                local.append(f"{mask_key} != ({count_key} > 0)")
        real_times = np.asarray(sample["timestamps"][:length], dtype=np.float64)
        if length and (
            np.any(real_times[:, 0] < 0)
            or np.any(real_times[:, 1] <= real_times[:, 0])
            or np.any(real_times[:, 1] > float(sample["duration"]) + 0.25)
        ):
            local.append("invalid real word timestamp")
        if length > 1 and np.any(real_times[1:, 0] < real_times[:-1, 1] - 1.0e-5):
            local.append("word timestamps overlap or are out of order")
        if local:
            errors.extend(f"{sid}: {message}" for message in local)
        sample_results.append({"sample_id": sid, "passed": not local, "errors": local})

    status_counts = dict(Counter(str(sample["whisperx_status"]) for sample in samples))
    vision_counts = dict(Counter(str(sample["vision_status"]) for sample in samples))
    return {
        "overall_pass": not errors,
        "validated_samples": len(samples),
        "passed_samples": sum(bool(item["passed"]) for item in sample_results),
        "failed_samples": sum(not bool(item["passed"]) for item in sample_results),
        "whisperx_status_counts": status_counts,
        "vision_status_counts": vision_counts,
        "overall_shapes": {
            "text": [len(samples), max_len, int(meta["text_dim"])],
            "audio": [len(samples), max_len, int(meta["audio_dim"])],
            "vision": [len(samples), max_len, int(meta["vision_dim"])],
        },
        "errors": errors,
        "samples": sample_results,
    }


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = read_config(resolve_project_path(args.config))
    # Resolve relative data paths against the project, while leaving the
    # original YAML untouched for the MFA pipeline.
    for key in ("work_dir", "output_dir", "huggingface_home"):
        config["paths"][key] = str(resolve_project_path(config["paths"][key]))

    whisperx_dir = resolve_project_path(args.whisperx_dir)
    text_cache_dir = resolve_project_path(args.text_cache_dir)
    output_path = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary)
    report_path = resolve_project_path(args.validation_report)
    official_path = resolve_project_path(
        Path(config["paths"]["output_dir"]) / str(config["problem1"]["dataset_file"])
    )
    if output_path == official_path:
        raise ValueError(
            f"Refusing to overwrite the official-text/MFA dataset: {official_path}. "
            f"Use a distinct WhisperX output such as outputs/{DATASET_FILENAME}."
        )
    official_summary = resolve_project_path(
        Path(config["paths"]["output_dir"]) / str(config["problem1"]["summary_file"])
    )
    if summary_path == official_summary:
        raise ValueError(f"Refusing to overwrite the official-text/MFA summary: {official_summary}")

    manifest = load_manifest(config)
    if args.sample_id:
        requested = list(dict.fromkeys(args.sample_id))
        expected_ids = requested
    else:
        expected_ids = manifest.sort_values("feature_index")["sample_id"].astype(str).tolist()
    dataset, summary = build_dataset(
        config, whisperx_dir, text_cache_dir, args.sample_id, args.force_text
    )
    report = validate_dataset(dataset, expected_ids)
    if not report["overall_pass"]:
        preview = "\n".join(report["errors"][:20])
        raise RuntimeError(f"WhisperX aligned dataset validation failed:\n{preview}")

    print(
        f"VALIDATION_OK: {report['passed_samples']}/{report['validated_samples']} samples; "
        f"shapes={report['overall_shapes']}",
        flush=True,
    )
    if args.check_only:
        return
    atomic_pickle(output_path, dataset)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved dataset: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved validation report: {report_path}")


if __name__ == "__main__":
    main()
