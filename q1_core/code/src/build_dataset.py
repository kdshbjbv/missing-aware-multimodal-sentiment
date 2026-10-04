from __future__ import annotations

import pickle
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .align_modalities import aligned_feature_path
from .utils import config_fingerprint, get_logger, package_version


def _scalar_text(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def _read_aligned(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_dataset(config: dict[str, Any], manifest: pd.DataFrame) -> tuple[Path, Path]:
    logger = get_logger()
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    available: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        path = aligned_feature_path(config, sid)
        if not path.exists():
            errors[sid] = "aligned feature file missing"
            continue
        try:
            available[sid] = _read_aligned(path)
        except Exception as exc:
            errors[sid] = str(exc)
    if not available:
        raise RuntimeError("No aligned samples are available; dataset dimensions cannot be established")

    reference = next(iter(available.values()))
    max_len = int(config["project"]["max_len"])
    text_dim = int(reference["text"].shape[1])
    audio_dim = int(reference["audio"].shape[1])
    vision_dim = int(reference["vision"].shape[1])
    include_failed = bool(config["output"].get("include_failed_samples", True))
    samples: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in manifest.sort_values("feature_index").iterrows():
        sid = str(row["sample_id"])
        aligned = available.get(sid)
        status = "success"
        error_message = ""
        if aligned is None:
            status = "failed"
            error_message = errors.get(sid, "unknown error")
            if not include_failed:
                continue
            aligned = {
                "words": np.full(max_len, "<PAD>", dtype="<U5"),
                "timestamps": np.full((max_len, 2), -1.0, dtype=np.float32),
                "text": np.zeros((max_len, text_dim), dtype=np.float32),
                "audio": np.zeros((max_len, audio_dim), dtype=np.float32),
                "vision": np.zeros((max_len, vision_dim), dtype=np.float32),
                "sequence_mask": np.zeros(max_len, dtype=np.uint8),
                "text_mask": np.zeros(max_len, dtype=np.uint8),
                "audio_mask": np.zeros(max_len, dtype=np.uint8),
                "vision_mask": np.zeros(max_len, dtype=np.uint8),
                "valid_length": np.asarray(0, dtype=np.int16),
                "original_length": np.asarray(0, dtype=np.int16),
                "truncated_count": np.asarray(0, dtype=np.int16),
                "text_extractor": np.asarray("missing", dtype=np.str_),
                "audio_extractor": np.asarray("missing", dtype=np.str_),
                "vision_extractor": np.asarray("missing", dtype=np.str_),
            }
        valid_length = int(np.asarray(aligned["valid_length"]).item())
        sample = {
            "sample_id": sid,
            "video_id": str(row["video_id"]),
            "clip_id": str(row["clip_id"]),
            "raw_text": str(row["raw_text"]),
            "words": aligned["words"],
            "text": aligned["text"].astype(np.float32),
            "audio": aligned["audio"].astype(np.float32),
            "vision": aligned["vision"].astype(np.float32),
            "sequence_mask": aligned["sequence_mask"].astype(np.uint8),
            "text_mask": aligned["text_mask"].astype(np.uint8),
            "audio_mask": aligned["audio_mask"].astype(np.uint8),
            "vision_mask": aligned["vision_mask"].astype(np.uint8),
            "timestamps": aligned["timestamps"].astype(np.float32),
            "valid_length": valid_length,
            "duration": float(row["duration_s"]),
            "label": float(row["regression_label"]),
            "annotation": str(row["annotation"]),
            "classification_label": int(row["classification_label"]),
            "status": status,
            "error": error_message,
        }
        samples.append(sample)
        sequence_denominator = max(1, valid_length)
        summary_rows.append(
            {
                "feature_index": int(row["feature_index"]),
                "sample_id": sid,
                "video_id": str(row["video_id"]),
                "clip_id": str(row["clip_id"]),
                "duration_s": float(row["duration_s"]),
                "fps": float(row["fps"]),
                "frame_count": int(row["frame_count"]),
                "text_dim": text_dim,
                "audio_dim": audio_dim,
                "vision_dim": vision_dim,
                "max_len": max_len,
                "original_length": int(np.asarray(aligned["original_length"]).item()),
                "valid_length": valid_length,
                "padding_length": max_len - valid_length,
                "truncated_count": int(np.asarray(aligned["truncated_count"]).item()),
                "alignment_granularity": "MFA word interval",
                "text_valid_rate": float(aligned["text_mask"].sum()) / sequence_denominator,
                "audio_valid_rate": float(aligned["audio_mask"].sum()) / sequence_denominator,
                "vision_valid_rate": float(aligned["vision_mask"].sum()) / sequence_denominator,
                "text_extractor": _scalar_text(aligned["text_extractor"]),
                "audio_extractor": _scalar_text(aligned["audio_extractor"]),
                "vision_extractor": _scalar_text(aligned["vision_extractor"]),
                "regression_label": float(row["regression_label"]),
                "annotation": str(row["annotation"]),
                "processing_status": status,
                "error_message": error_message,
            }
        )
    dataset = {
        "samples": samples,
        "meta": {
            "schema_version": "1.0",
            "num_samples": len(samples),
            "max_len": max_len,
            "text_dim": text_dim,
            "audio_dim": audio_dim,
            "vision_dim": vision_dim,
            "timestamp_unit": "second",
            "timestamp_padding": float(config["alignment"].get("timestamp_padding_value", -1.0)),
            "alignment": "Montreal Forced Aligner word intervals with frame-level mean pooling",
            "overlength_strategy": config["alignment"].get("overlength_strategy", "truncate"),
            "classification_mapping": config["data"]["classification_mapping"],
            "config_fingerprint": config_fingerprint(config),
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "torch", "transformers", "opensmile", "textgrid")
            },
            "failed_samples": errors,
        },
    }
    dataset_path = output_dir / config["output"]["dataset_file"]
    with dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary_path = output_dir / config["output"]["summary_file"]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("Saved dataset with %d samples to %s", len(samples), dataset_path)
    logger.info("Saved feature summary to %s", summary_path)
    return dataset_path, summary_path
