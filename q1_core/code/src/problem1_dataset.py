from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import loadmat

from .prepare_manifest import load_manifest
from .utils import check_finite, safe_sample_name
from .vision_35 import audit_openface_au_fields, extract_visual35_from_csv


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _matlab_round_positive(value: float) -> int:
    """Match MATLAB round for the positive sample-index values used here."""
    return int(math.floor(value + 0.5))


def covarep_frame_times(
    wav_path: Path,
    frame_count: int,
    hop_seconds: float,
    first_sample_fraction: float = 0.5,
) -> np.ndarray:
    """Reconstruct timestamps used by official COVAREP_feature_extraction.m.

    COVAREP's default wrapper uses 1-based indices
    round((hop/2)*fs):round(hop*fs):length(x), and converts an index to
    seconds with (index-1)/fs.
    """
    info = sf.info(wav_path)
    fs = int(info.samplerate)
    first_index = _matlab_round_positive(hop_seconds * first_sample_fraction * fs)
    hop_samples = _matlab_round_positive(hop_seconds * fs)
    if first_index < 1 or hop_samples < 1:
        raise ValueError(f"Invalid COVAREP timing parameters for {wav_path}")
    expected = 0 if info.frames < first_index else (info.frames - first_index) // hop_samples + 1
    if expected != int(frame_count):
        raise ValueError(
            f"COVAREP frame count/timing mismatch for {wav_path.name}: "
            f"MAT={frame_count}, expected={expected}, samples={info.frames}, fs={fs}"
        )
    indices_1based = first_index + np.arange(frame_count, dtype=np.int64) * hop_samples
    return ((indices_1based - 1) / float(fs)).astype(np.float64)


def _mat_string(value: Any) -> str:
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).reshape(-1).tolist())
        value = value.reshape(-1)[0]
    return str(value)


def load_covarep_mat(
    mat_path: Path,
    feature_key: str,
    names_key: str,
    expected_dim: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    payload = loadmat(mat_path)
    public_keys = [key for key in payload if not key.startswith("__")]
    if feature_key not in payload:
        arrays = {
            key: tuple(np.asarray(value).shape)
            for key, value in payload.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
        raise KeyError(f"{mat_path.name}: missing key {feature_key!r}; arrays={arrays}")
    features = np.asarray(payload[feature_key])
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"{mat_path.name}: {feature_key} shape is {features.shape}, expected [T,{expected_dim}]"
        )
    features = features.astype(np.float32)
    check_finite(f"COVAREP {mat_path.name}", features)
    if names_key not in payload:
        raise KeyError(f"{mat_path.name}: missing feature-name key {names_key!r}")
    names = [_mat_string(value) for value in np.asarray(payload[names_key], dtype=object).reshape(-1)]
    if len(names) != expected_dim:
        raise ValueError(f"{mat_path.name}: found {len(names)} feature names, expected {expected_dim}")
    return features, names, public_keys


def pool_points_by_word_windows(
    frame_times: np.ndarray,
    frame_features: np.ndarray,
    windows: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool points using the required start <= t < end convention."""
    times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    features = np.asarray(frame_features, dtype=np.float32)
    intervals = np.asarray(windows, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(times):
        raise ValueError("Frame timestamps and features have inconsistent shapes")
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("Word windows must have shape [L,2]")
    valid_frames = np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid).astype(bool)
    if len(valid_frames) != len(times):
        raise ValueError("Frame validity mask has the wrong length")
    output = np.zeros((len(intervals), features.shape[1]), dtype=np.float32)
    mask = np.zeros(len(intervals), dtype=np.uint8)
    counts = np.zeros(len(intervals), dtype=np.int32)
    for index, (start, end) in enumerate(intervals):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        selected = (times >= start) & (times < end) & valid_frames
        count = int(selected.sum())
        if count:
            output[index] = features[selected].mean(axis=0, dtype=np.float64).astype(np.float32)
            mask[index] = 1
            counts[index] = count
    check_finite("word-pooled features", output)
    return output, mask, counts


def load_mfa_reference(config: dict[str, Any], sample_id: str) -> tuple[list[str], np.ndarray]:
    path = Path(config["paths"]["work_dir"]) / "mfa" / "word_timestamps" / (
        safe_sample_name(sample_id) + ".json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("words", [])
    words = [str(item["word"]) for item in items]
    # Keep the JSON boundary values in float64 during pooling. Casting 0.2 to
    # float32 before a strict t < end comparison can accidentally include a
    # frame at exactly 0.2 because the stored boundary becomes slightly larger.
    timestamps = np.asarray([[float(item["start"]), float(item["end"])] for item in items], dtype=np.float64)
    if timestamps.size == 0:
        timestamps = np.zeros((0, 2), dtype=np.float64)
    if timestamps.shape != (len(words), 2):
        raise ValueError(f"Malformed MFA timestamps for {sample_id}: {timestamps.shape}")
    return words, timestamps


def wav_statistics(path: Path) -> dict[str, float | int]:
    signal, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    mono = signal.mean(axis=1) if len(signal) else np.zeros(0, dtype=np.float64)
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    maximum = float(np.max(np.abs(mono))) if len(mono) else 0.0
    return {
        "sample_rate": int(sample_rate),
        "num_samples": int(len(mono)),
        "duration": float(len(mono) / sample_rate),
        "rms": rms,
        "max_abs": maximum,
    }


def _pad_matrix(values: np.ndarray, max_len: int, width: int) -> np.ndarray:
    result = np.zeros((max_len, width), dtype=np.float32)
    length = min(len(values), max_len)
    result[:length] = np.asarray(values[:length], dtype=np.float32)
    return result


def _pad_vector(values: np.ndarray, max_len: int, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.zeros(max_len, dtype=dtype)
    length = min(len(values), max_len)
    result[:length] = np.asarray(values[:length], dtype=dtype)
    return result


def _load_vision_on_mfa_timeline(
    config: dict[str, Any],
    sample_id: str,
    mfa_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    work = Path(config["paths"]["work_dir"])
    stem = safe_sample_name(sample_id)
    existing = _load_npz(work / "aligned" / f"{stem}.npz")
    max_len = int(config["project"]["max_len"])
    valid_length = min(len(mfa_timestamps), max_len)
    tolerance = float(config["problem1"].get("timestamp_tolerance", 1.0e-6))
    existing_match = (
        int(existing["valid_length"]) == valid_length
        and existing["timestamps"].shape == (max_len, 2)
        and np.allclose(existing["timestamps"][:valid_length], mfa_timestamps[:valid_length], atol=tolerance, rtol=0)
    )
    if existing_match:
        return (
            np.asarray(existing["vision"], dtype=np.float32),
            np.asarray(existing["vision_mask"], dtype=np.uint8),
            np.asarray(existing["vision_frame_counts"], dtype=np.int32),
            {
                "vision_timestamp_match_before_correction": True,
                "vision_alignment_source": "existing_word_level_alignment",
            },
        )

    # Do not rerun OpenFace. Correct only the time-window aggregation using the
    # already cached frame-level OpenFace NPZ and the authoritative MFA JSON.
    frame_data = _load_npz(work / "features" / "vision" / f"{stem}.npz")
    raw, mask, counts = pool_points_by_word_windows(
        frame_data["times"], frame_data["features"], mfa_timestamps, frame_data.get("valid")
    )
    dim = int(config["problem1"]["vision_legacy_dim"])
    if raw.shape[1] != dim:
        raise ValueError(f"{sample_id}: cached OpenFace dimension {raw.shape[1]} != {dim}")
    return (
        _pad_matrix(raw, max_len, dim),
        _pad_vector(mask, max_len, np.uint8),
        _pad_vector(counts, max_len, np.int32),
        {
            "vision_timestamp_match_before_correction": False,
            "vision_alignment_source": "recomputed_from_cached_openface_frames",
        },
    )


def build_one_sample(config: dict[str, Any], row: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(row["sample_id"])
    stem = safe_sample_name(sample_id)
    work = Path(config["paths"]["work_dir"])
    settings = config["problem1"]
    max_len = int(config["project"]["max_len"])
    text_dim = int(settings["text_expected_dim"])
    audio_dim = int(settings["covarep_expected_dim"])
    vision_dim = int(settings["vision_expected_dim"])
    vision_legacy_dim = int(settings["vision_legacy_dim"])

    words, timestamps = load_mfa_reference(config, sample_id)
    original_word_count = len(words)
    valid_length = min(original_word_count, max_len)
    truncated = original_word_count > max_len

    text_data = _load_npz(work / "features" / "text" / f"{stem}.npz")
    bert_words = [str(value) for value in text_data["words"].tolist()]
    mfa_bert_match = words == bert_words
    if not mfa_bert_match:
        mismatch = next(
            (i for i, pair in enumerate(zip(words, bert_words)) if pair[0] != pair[1]),
            min(len(words), len(bert_words)),
        )
        raise ValueError(
            f"MFA_BERT_WORD_MISMATCH {sample_id}: index={mismatch}, "
            f"MFA={words[mismatch:mismatch+3]}, BERT={bert_words[mismatch:mismatch+3]}"
        )
    text_raw = np.asarray(text_data["features"], dtype=np.float32)
    if text_raw.shape != (original_word_count, text_dim):
        raise ValueError(f"{sample_id}: BERT shape {text_raw.shape}, expected {(original_word_count, text_dim)}")
    bert_timestamps = np.asarray(text_data["timestamps"], dtype=np.float32)
    bert_timestamp_match = bert_timestamps.shape == timestamps.shape and np.allclose(
        bert_timestamps, timestamps, atol=float(settings.get("timestamp_tolerance", 1.0e-6)), rtol=0
    )

    text = _pad_matrix(text_raw, max_len, text_dim)
    sequence_mask = np.zeros(max_len, dtype=np.uint8)
    sequence_mask[:valid_length] = 1
    text_mask = sequence_mask.copy()
    padded_timestamps = np.zeros((max_len, 2), dtype=np.float32)
    padded_timestamps[:valid_length] = timestamps[:valid_length]
    padded_words = words[:valid_length] + [""] * (max_len - valid_length)

    wav_path = work / "audio" / f"{stem}.wav"
    mat_path = work / "audio" / f"{stem}.mat"
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    audio_feature_names: list[str] | None = None
    covarep_public_keys: list[str] = []
    if mat_path.is_file():
        audio_frames, audio_feature_names, covarep_public_keys = load_covarep_mat(
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
        audio_status = "covarep"
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

    vision_22, _, _, vision_info = _load_vision_on_mfa_timeline(
        config, sample_id, timestamps
    )
    if vision_22.shape != (max_len, vision_legacy_dim):
        raise ValueError(
            f"{sample_id}: legacy vision shape {vision_22.shape}, "
            f"expected {(max_len, vision_legacy_dim)}"
        )
    vision35_data = extract_visual35_from_csv(
        config, sample_id, config["_vision35_audit"]
    )
    vision_raw, vision_mask_raw, vision_counts_raw = pool_points_by_word_windows(
        vision35_data["times"],
        vision35_data["features"],
        timestamps,
        vision35_data["valid"],
    )
    vision = _pad_matrix(vision_raw, max_len, vision_dim)
    vision_mask = _pad_vector(vision_mask_raw, max_len, np.uint8)
    vision_counts = _pad_vector(vision_counts_raw, max_len, np.int32)
    if vision.shape != (max_len, vision_dim):
        raise ValueError(f"{sample_id}: vision shape {vision.shape}, expected {(max_len, vision_dim)}")
    real_vision = vision_mask[:valid_length]
    if int(real_vision.sum()) == 0:
        vision_status = "missing"
    elif int(real_vision.sum()) < valid_length:
        vision_status = "partial"
    else:
        vision_status = "available"

    for name, values in (("text", text), ("audio", audio), ("vision", vision)):
        check_finite(f"{sample_id} {name}", values)

    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "video_id": str(row["video_id"]),
        "clip_id": str(row["clip_id"]),
        "raw_text": str(row["raw_text"]),
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
        "alignment_status": "success",
        "mfa_bert_match": mfa_bert_match,
        "mfa_bert_timestamp_match": bool(bert_timestamp_match),
        **vision_info,
        "regression_label": float(row["regression_label"]),
        "classification_label": int(row["classification_label"]),
        "annotation": str(row["annotation"]),
    }
    if silence_stats is not None:
        sample["silence_check"] = silence_stats

    text_valid = int(text_mask[:valid_length].sum())
    audio_valid = int(audio_mask[:valid_length].sum())
    vision_valid = int(vision_mask[:valid_length].sum())
    summary = {
        "sample_id": sample_id,
        "video_id": str(row["video_id"]),
        "clip_id": str(row["clip_id"]),
        "duration": float(row["duration_s"]),
        "original_word_count": original_word_count,
        "valid_length": valid_length,
        "truncated": truncated,
        "MFA_BERT_match": mfa_bert_match,
        "MFA_BERT_timestamp_match": bool(bert_timestamp_match),
        "vision_timestamp_match_before_correction": vision_info[
            "vision_timestamp_match_before_correction"
        ],
        "vision_alignment_source": vision_info["vision_alignment_source"],
        "text_shape": str(tuple(text.shape)),
        "audio_shape": str(tuple(audio.shape)),
        "vision_shape": str(tuple(vision.shape)),
        "vision_22_shape": str(tuple(vision_22.shape)),
        "text_valid_windows": text_valid,
        "audio_valid_windows": audio_valid,
        "vision_valid_windows": vision_valid,
        "audio_valid_ratio": audio_valid / valid_length if valid_length else 0.0,
        "vision_valid_ratio": vision_valid / valid_length if valid_length else 0.0,
        "audio_status": audio_status,
        "vision_status": vision_status,
        "alignment_status": "success",
        "regression_label": float(row["regression_label"]),
        "classification_label": int(row["classification_label"]),
        "annotation": str(row["annotation"]),
    }
    sample["_build_audio_feature_names"] = audio_feature_names
    sample["_build_covarep_public_keys"] = covarep_public_keys
    return sample, summary


def _select_feature_names(samples: list[dict[str, Any]], field: str) -> list[str]:
    variants = [sample[field] for sample in samples if sample.get(field)]
    if not variants:
        return []
    reference = list(variants[0])
    if any(list(value) != reference for value in variants[1:]):
        raise ValueError(f"Inconsistent feature-name variants for {field}")
    return reference


def build_problem1_dataset(
    config: dict[str, Any], sample_ids: Iterable[str] | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = load_manifest(config)
    if sample_ids is not None:
        requested = list(sample_ids)
        missing = sorted(set(requested) - set(manifest["sample_id"].astype(str)))
        if missing:
            raise KeyError(f"Unknown sample IDs: {missing}")
        order = {sample_id: index for index, sample_id in enumerate(requested)}
        manifest = manifest[manifest["sample_id"].isin(requested)].copy()
        manifest["_requested_order"] = manifest["sample_id"].map(order)
        manifest = manifest.sort_values("_requested_order")

    vision_audit = audit_openface_au_fields(config, manifest["sample_id"].astype(str).tolist())
    config["_vision35_audit"] = vision_audit

    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_non_silent: list[str] = []
    for _, row in manifest.iterrows():
        sample, summary = build_one_sample(config, row)
        if sample["audio_status"] == "covarep_missing":
            missing_non_silent.append(sample["sample_id"])
        samples.append(sample)
        summaries.append(summary)

    audio_feature_names = _select_feature_names(samples, "_build_audio_feature_names")
    covarep_keys = sorted(
        {key for sample in samples for key in sample.get("_build_covarep_public_keys", [])}
    )
    for sample in samples:
        sample.pop("_build_audio_feature_names", None)
        sample.pop("_build_covarep_public_keys", None)

    legacy_vision_name_variants: list[list[str]] = []
    for sample in samples:
        vision_path = (
            Path(config["paths"]["work_dir"])
            / "features"
            / "vision"
            / f"{safe_sample_name(sample['sample_id'])}.npz"
        )
        with np.load(vision_path, allow_pickle=False) as vision_data:
            legacy_vision_name_variants.append(
                [str(value) for value in vision_data["feature_names"].tolist()]
            )
    legacy_vision_feature_names = (
        legacy_vision_name_variants[0] if legacy_vision_name_variants else []
    )
    if any(value != legacy_vision_feature_names for value in legacy_vision_name_variants[1:]):
        raise ValueError("Inconsistent OpenFace feature-name variants")

    mapping = {str(key): int(value) for key, value in config["data"]["classification_mapping"].items()}
    dataset = {
        "samples": samples,
        "meta": {
            "num_samples": len(samples),
            "max_len": int(config["project"]["max_len"]),
            "alignment_unit": "word",
            "alignment_reference": "Montreal Forced Aligner word timestamps",
            "text_extractor": "google-bert/bert-base-uncased",
            "text_dim": int(config["problem1"]["text_expected_dim"]),
            "audio_extractor": "COVAREP",
            "audio_dim": int(config["problem1"]["covarep_expected_dim"]),
            "audio_feature_names": audio_feature_names,
            "covarep_mat_keys": covarep_keys,
            "covarep_feature_key": str(config["problem1"]["covarep_feature_key"]),
            "covarep_names_key": str(config["problem1"]["covarep_names_key"]),
            "covarep_hop_seconds": float(config["problem1"]["covarep_hop_seconds"]),
            "covarep_timestamp_rule": (
                "(round(0.005*fs)-1)/fs + k*round(0.01*fs)/fs; k=0..T-1"
            ),
            "vision_extractor": "OpenFace 2.2",
            "vision_dim": int(config["problem1"]["vision_expected_dim"]),
            "vision_definition": "17 AU intensity + 18 AU presence",
            "vision_feature_names": list(vision_audit["feature_names"]),
            "vision_is_facet": False,
            "vision_22_dim": int(config["problem1"]["vision_legacy_dim"]),
            "vision_22_definition": "17 AU intensity + 3 head pose + 2 gaze angle",
            "vision_22_feature_names": legacy_vision_feature_names,
            "openface_csv_schema_consistent": bool(vision_audit["schema_consistent"]),
            "openface_csv_count": int(vision_audit["num_csv"]),
            "padding_strategy": "zero padding; empty word; timestamp [0,0]",
            "truncation_strategy": "first 50 valid words",
            "classification_mapping": mapping,
            "notes": "Self-generated word-level aligned multimodal features for problem 1.",
        },
    }
    if missing_non_silent:
        output = Path(config["paths"]["output_dir"]) / "missing_covarep_samples.txt"
        output.write_text("\n".join(missing_non_silent) + "\n", encoding="utf-8")
    return dataset, pd.DataFrame.from_records(summaries)


def save_problem1_dataset(
    config: dict[str, Any],
    dataset: dict[str, Any],
    summary: pd.DataFrame,
    dataset_path: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    output = Path(config["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path or output / str(config["problem1"]["dataset_file"])
    summary_path = summary_path or output / str(config["problem1"]["summary_file"])
    with dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return dataset_path, summary_path


def _draw_alignment_figure(sample: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    length = int(sample["valid_length"])
    windows = sample["timestamps"][:length]
    words = sample["words"][:length]
    starts = windows[:, 0]
    widths = np.maximum(windows[:, 1] - windows[:, 0], 1.0e-4)
    audio_counts = sample["audio_frame_counts"][:length]
    vision_counts = sample["vision_frame_counts"][:length]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.0, 0.65]},
        constrained_layout=True,
    )
    for index, (word, (start, end)) in enumerate(zip(words, windows)):
        color = "#D9EAF7" if index % 2 == 0 else "#E8F3DF"
        axes[0].add_patch(Rectangle((start, 0), end - start, 1, facecolor=color, edgecolor="#506070"))
        axes[0].text(
            (start + end) / 2,
            0.5,
            word,
            ha="center",
            va="center",
            rotation=45,
            fontsize=8,
            parse_math=False,
        )
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_ylabel("MFA words")

    axes[1].bar(starts, audio_counts, width=widths, align="edge", color="#2878B5", edgecolor="white")
    axes[1].set_ylabel("COVAREP\nframes")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(starts, vision_counts, width=widths, align="edge", color="#D95F02", edgecolor="white")
    axes[2].set_ylabel("OpenFace\nframes")
    axes[2].grid(axis="y", alpha=0.25)

    masks = np.vstack(
        [
            sample["sequence_mask"][:length],
            sample["audio_mask"][:length],
            sample["vision_mask"][:length],
        ]
    )
    for row_index, row in enumerate(masks):
        for index, value in enumerate(row):
            axes[3].add_patch(
                Rectangle(
                    (starts[index], row_index),
                    widths[index],
                    1,
                    facecolor="#2CA25F" if value else "#D9D9D9",
                    edgecolor="white",
                )
            )
    axes[3].set_ylim(3, 0)
    axes[3].set_yticks([0.5, 1.5, 2.5], ["sequence", "audio", "vision"])
    axes[3].set_xlabel("Original video time (seconds)")
    axes[3].set_ylabel("Masks")
    if length:
        axes[3].set_xlim(max(0.0, float(starts[0]) - 0.05), float(windows[-1, 1]) + 0.05)
    fig.suptitle(f"{title}\n{sample['sample_id']}", fontsize=13, parse_math=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_problem1_figures(config: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Path]:
    samples = dataset["samples"]
    figures = Path(config["paths"]["output_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    normal_candidates = [
        sample
        for sample in samples
        if not sample["truncated"]
        and sample["mfa_bert_match"]
        and sample["audio_status"] == "covarep"
        and 5 <= sample["valid_length"] <= 22
    ]
    if not normal_candidates:
        normal_candidates = list(samples)
    normal = max(
        normal_candidates,
        key=lambda value: (
            float(value["audio_mask"][: value["valid_length"]].mean()),
            float(value["vision_mask"][: value["valid_length"]].mean()),
        ),
    )
    paths = {"alignment_example": figures / "alignment_example.png"}
    _draw_alignment_figure(normal, paths["alignment_example"], "Typical word-level multimodal alignment")

    partial = [sample for sample in samples if sample["vision_status"] == "partial"]
    if partial:
        chosen = min(
            partial,
            key=lambda value: float(value["vision_mask"][: value["valid_length"]].mean()),
        )
        paths["missing_vision_example"] = figures / "missing_vision_example.png"
        _draw_alignment_figure(
            chosen,
            paths["missing_vision_example"],
            "Real words with partially missing reliable visual observations",
        )

    silent = [sample for sample in samples if sample["audio_status"] == "silent"]
    if silent:
        paths["silent_audio_example"] = figures / "silent_audio_example.png"
        _draw_alignment_figure(
            silent[0],
            paths["silent_audio_example"],
            "Silent audio: modality absence is distinct from sequence padding",
        )
    return paths
