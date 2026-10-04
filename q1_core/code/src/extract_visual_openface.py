from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import check_finite, get_logger, resolve_executable, run_command, safe_sample_name


AU_INTENSITY = re.compile(r"^AU\d+_r$")
AU_PRESENCE = re.compile(r"^AU\d+_c$")


def vision_feature_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "features" / "vision" / f"{safe_sample_name(sample_id)}.npz"


def selected_openface_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    columns = [str(name).strip() for name in frame.columns]
    selected = [name for name in columns if AU_INTENSITY.match(name)]
    if config["vision"].get("include_au_presence", False):
        selected.extend(name for name in columns if AU_PRESENCE.match(name))
    selected.extend(name for name in ("pose_Rx", "pose_Ry", "pose_Rz") if name in columns)
    if config["vision"].get("include_translation", False):
        selected.extend(name for name in ("pose_Tx", "pose_Ty", "pose_Tz") if name in columns)
    selected.extend(name for name in ("gaze_angle_x", "gaze_angle_y") if name in columns)
    return list(dict.fromkeys(selected))


def extract_one_visual(config: dict[str, Any], row: pd.Series, force: bool = False) -> Path:
    logger = get_logger()
    sid = str(row["sample_id"])
    destination = vision_feature_path(config, sid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.debug("Visual feature cache hit: %s", destination)
        return destination
    binary = resolve_executable(config["paths"]["openface_feature_extraction"])
    if binary is None:
        raise RuntimeError(
            "OpenFace FeatureExtraction was not found. Build OpenFace 2.2 and set "
            "paths.openface_feature_extraction in the YAML file. No visual features were fabricated."
        )
    video_path = Path(str(row["video_path"]))
    raw_output = Path(config["paths"]["work_dir"]) / "openface" / safe_sample_name(sid)
    raw_output.mkdir(parents=True, exist_ok=True)
    csv_path = raw_output / f"{video_path.stem}.csv"
    if force or not csv_path.exists():
        command = [
            binary,
            "-f",
            str(video_path),
            "-out_dir",
            str(raw_output),
            "-aus",
            "-pose",
            "-gaze",
            "-q",
        ]
        command.extend(str(value) for value in config["vision"].get("extra_openface_args", []))
        environment = os.environ.copy()
        library_dir = str(config["paths"].get("openface_library_dir", "")).strip()
        if library_dir and os.name != "nt":
            previous = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = library_dir + (os.pathsep + previous if previous else "")
        result = run_command(command, check=False, env=environment)
        if result.returncode != 0:
            raise RuntimeError(f"OpenFace failed for {sid}: {result.stderr}")
    if not csv_path.exists():
        candidates = list(raw_output.glob("*.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"OpenFace CSV was not found for {sid} in {raw_output}")
        csv_path = candidates[0]
    frame = pd.read_csv(csv_path, skipinitialspace=True)
    frame.columns = [str(name).strip() for name in frame.columns]
    if "timestamp" not in frame.columns:
        raise ValueError(f"OpenFace CSV lacks timestamp column: {csv_path}")
    feature_columns = selected_openface_columns(frame, config)
    if not feature_columns:
        raise ValueError(f"No requested AU/pose/gaze columns found in {csv_path}")
    confidence = frame["confidence"].to_numpy(dtype=np.float32) if "confidence" in frame else np.ones(len(frame))
    success = frame["success"].to_numpy(dtype=np.float32) if "success" in frame else np.ones(len(frame))
    threshold = float(config["vision"].get("confidence_threshold", 0.8))
    valid = ((success > 0) & (confidence >= threshold)).astype(np.uint8)
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    features[~np.isfinite(features)] = 0.0
    check_finite("OpenFace frame features", features)
    np.savez_compressed(
        destination,
        times=frame["timestamp"].to_numpy(dtype=np.float32),
        features=features,
        valid=valid,
        confidence=confidence,
        feature_names=np.asarray(feature_columns, dtype=np.str_),
        extractor=np.asarray("OpenFace 2.2 AU intensity + head pose + gaze", dtype=np.str_),
        source_csv=np.asarray(str(csv_path), dtype=np.str_),
    )
    logger.info(
        "Extracted OpenFace features: %s (%d dimensions, %.1f%% valid frames)",
        sid,
        len(feature_columns),
        100.0 * float(valid.mean()) if len(valid) else 0.0,
    )
    return destination
