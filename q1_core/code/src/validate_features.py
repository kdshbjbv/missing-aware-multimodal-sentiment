from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger


def validate_dataset(config: dict[str, Any], manifest: pd.DataFrame, strict: bool | None = None) -> dict[str, Any]:
    logger = get_logger()
    strict = config["project"].get("strict_final_validation", True) if strict is None else strict
    dataset_path = Path(config["paths"]["output_dir"]) / config["output"]["dataset_file"]
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    errors: list[str] = []
    warnings: list[str] = []
    samples = dataset.get("samples", [])
    max_len = int(config["project"]["max_len"])
    if len(samples) != len(manifest):
        errors.append(f"sample count mismatch: dataset={len(samples)}, manifest={len(manifest)}")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        errors.append("sample_id is not unique")
    expected_ids = manifest.sort_values("feature_index")["sample_id"].astype(str).tolist()
    if ids != expected_ids:
        errors.append("dataset sample order does not match manifest feature_index")
    for index, sample in enumerate(samples):
        sid = sample["sample_id"]
        if sample.get("status") != "success":
            message = f"{sid}: processing status={sample.get('status')} ({sample.get('error', '')})"
            (errors if strict else warnings).append(message)
        text = np.asarray(sample["text"])
        audio = np.asarray(sample["audio"])
        vision = np.asarray(sample["vision"])
        timestamps = np.asarray(sample["timestamps"])
        masks = {
            name: np.asarray(sample[name])
            for name in ("sequence_mask", "text_mask", "audio_mask", "vision_mask")
        }
        for name, array in (("text", text), ("audio", audio), ("vision", vision)):
            if array.ndim != 2 or array.shape[0] != max_len:
                errors.append(f"{sid}: {name} shape is {array.shape}, expected [{max_len}, d]")
            if not np.isfinite(array).all():
                errors.append(f"{sid}: {name} contains NaN/Inf")
        if timestamps.shape != (max_len, 2):
            errors.append(f"{sid}: timestamps shape is {timestamps.shape}")
        for name, mask in masks.items():
            if mask.shape != (max_len,):
                errors.append(f"{sid}: {name} shape is {mask.shape}")
            if not np.isin(mask, (0, 1)).all():
                errors.append(f"{sid}: {name} contains values other than 0/1")
        valid_length = int(sample["valid_length"])
        if not 0 <= valid_length <= max_len:
            errors.append(f"{sid}: invalid valid_length={valid_length}")
            continue
        if int(masks["sequence_mask"].sum()) != valid_length:
            errors.append(f"{sid}: sequence_mask sum differs from valid_length")
        valid_times = timestamps[:valid_length]
        if valid_length and ((valid_times[:, 0] < 0).any() or (valid_times[:, 1] <= valid_times[:, 0]).any()):
            errors.append(f"{sid}: invalid active timestamp interval")
        if valid_length > 1 and (np.diff(valid_times[:, 0]) < -1e-6).any():
            errors.append(f"{sid}: timestamps are not monotonic")
        padding_value = float(config["alignment"].get("timestamp_padding_value", -1.0))
        if valid_length < max_len and not np.allclose(timestamps[valid_length:], padding_value):
            errors.append(f"{sid}: timestamp padding is inconsistent")
        for name, array in (("text", text), ("audio", audio), ("vision", vision)):
            if valid_length < max_len and not np.allclose(array[valid_length:], 0.0):
                errors.append(f"{sid}: {name} padding is not zero")
        row = manifest.iloc[index]
        if abs(float(sample["label"]) - float(row["regression_label"])) > 1e-6:
            errors.append(f"{sid}: regression label mismatch")
        if str(sample["annotation"]) != str(row["annotation"]):
            errors.append(f"{sid}: annotation mismatch")
    report = {
        "passed": not errors,
        "strict": bool(strict),
        "num_manifest_samples": len(manifest),
        "num_dataset_samples": len(samples),
        "num_success": sum(sample.get("status") == "success" for sample in samples),
        "errors": errors,
        "warnings": warnings,
    }
    report_path = Path(config["paths"]["output_dir"]) / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        logger.error("Dataset validation failed with %d errors", len(errors))
        if strict:
            raise ValueError(f"Validation failed; see {report_path}")
    else:
        logger.info("Dataset validation passed for %d samples", len(samples))
    return report
