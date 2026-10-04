from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import check_finite, safe_sample_name


AU_INTENSITY = re.compile(r"^AU\d+_r$")
AU_PRESENCE = re.compile(r"^AU\d+_c$")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def source_csv_for_sample(config: dict[str, Any], sample_id: str) -> Path:
    """Resolve the exact raw CSV from the existing traceable OpenFace cache."""
    path = (
        Path(config["paths"]["work_dir"])
        / "features"
        / "vision"
        / f"{safe_sample_name(sample_id)}.npz"
    )
    data = _load_npz(path)
    source = Path(str(data["source_csv"]))
    if not source.is_file():
        raise FileNotFoundError(f"OpenFace source CSV recorded by {path} is missing: {source}")
    return source


def audit_openface_au_fields(
    config: dict[str, Any], sample_ids: Iterable[str]
) -> dict[str, Any]:
    """Audit all selected raw CSV schemas before allowing 35-D extraction."""
    sample_ids = list(sample_ids)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs supplied to OpenFace audit")
    schema: tuple[str, ...] | None = None
    intensity: tuple[str, ...] | None = None
    presence: tuple[str, ...] | None = None
    source_by_sample: dict[str, str] = {}
    for sample_id in sample_ids:
        source = source_csv_for_sample(config, sample_id)
        columns = tuple(str(value).strip() for value in pd.read_csv(source, nrows=0, skipinitialspace=True).columns)
        current_intensity = tuple(value for value in columns if AU_INTENSITY.fullmatch(value))
        current_presence = tuple(value for value in columns if AU_PRESENCE.fullmatch(value))
        if schema is None:
            schema = columns
            intensity = current_intensity
            presence = current_presence
        elif columns != schema or current_intensity != intensity or current_presence != presence:
            raise ValueError(
                f"OPENFACE_SCHEMA_MISMATCH {sample_id}: "
                f"AU_r={list(current_intensity)}, AU_c={list(current_presence)}"
            )
        required = {"timestamp", "success", "confidence"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"{sample_id}: OpenFace CSV missing columns {missing}")
        source_by_sample[sample_id] = str(source)

    intensity = intensity or tuple()
    presence = presence or tuple()
    if len(intensity) != 17 or len(presence) != 18:
        raise ValueError(
            "OPENFACE_35D_DEFINITION_UNAVAILABLE: "
            f"actual AU_r={len(intensity)} {list(intensity)}, "
            f"AU_c={len(presence)} {list(presence)}"
        )
    names = list(intensity + presence)
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError(f"OpenFace 35-D feature names are not unique: {names}")
    return {
        "num_csv": len(sample_ids),
        "schema_consistent": True,
        "total_columns": len(schema or ()),
        "au_intensity_names": list(intensity),
        "au_presence_names": list(presence),
        "feature_names": names,
        "source_by_sample": source_by_sample,
    }


def vision35_feature_path(config: dict[str, Any], sample_id: str) -> Path:
    return (
        Path(config["paths"]["work_dir"])
        / "features"
        / "vision_35"
        / f"{safe_sample_name(sample_id)}.npz"
    )


def extract_visual35_from_csv(
    config: dict[str, Any],
    sample_id: str,
    audit: dict[str, Any],
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Read existing OpenFace CSV; this never invokes the OpenFace binary."""
    destination = vision35_feature_path(config, sample_id)
    expected_names = [str(value) for value in audit["feature_names"]]
    if destination.is_file() and not force:
        cached = _load_npz(destination)
        cached_names = [str(value) for value in cached["feature_names"].tolist()]
        if cached_names == expected_names and cached["features"].shape[1] == 35:
            return cached

    source = Path(str(audit["source_by_sample"][sample_id]))
    frame = pd.read_csv(source, skipinitialspace=True)
    frame.columns = [str(value).strip() for value in frame.columns]
    current_r = [value for value in frame.columns if AU_INTENSITY.fullmatch(value)]
    current_c = [value for value in frame.columns if AU_PRESENCE.fullmatch(value)]
    current_names = current_r + current_c
    if current_names != expected_names:
        raise ValueError(f"OPENFACE_SCHEMA_CHANGED {sample_id}: {current_names}")

    features = frame[expected_names].to_numpy(dtype=np.float32)
    check_finite(f"OpenFace 35-D {sample_id}", features)
    confidence = frame["confidence"].to_numpy(dtype=np.float32)
    success = frame["success"].to_numpy(dtype=np.float32)
    threshold = float(config["vision"].get("confidence_threshold", 0.8))
    valid = ((success > 0) & (confidence >= threshold)).astype(np.uint8)
    times = frame["timestamp"].to_numpy(dtype=np.float32)
    check_finite(f"OpenFace timestamps {sample_id}", times)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        times=times,
        features=features,
        valid=valid,
        confidence=confidence,
        feature_names=np.asarray(expected_names, dtype=np.str_),
        extractor=np.asarray("OpenFace 2.2", dtype=np.str_),
        definition=np.asarray("17 AU intensity + 18 AU presence", dtype=np.str_),
        source_csv=np.asarray(str(source), dtype=np.str_),
    )
    return _load_npz(destination)
