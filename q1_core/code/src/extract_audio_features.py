from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extract_audio import audio_output_path
from .utils import check_finite, get_logger, run_command, safe_sample_name


def audio_feature_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "features" / "audio" / f"{safe_sample_name(sample_id)}.npz"


def _extract_covarep(config: dict[str, Any], row: pd.Series, wav_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    template = str(config["paths"].get("covarep_command", "")).strip()
    if not template:
        raise RuntimeError(
            "COVAREP command is not configured. COVAREP requires a working MATLAB/Octave integration; "
            "the pipeline will use the explicitly labelled openSMILE fallback when configured."
        )
    work_output = Path(config["paths"]["work_dir"]) / "features" / "audio" / "covarep_raw"
    work_output.mkdir(parents=True, exist_ok=True)
    csv_path = work_output / f"{safe_sample_name(str(row['sample_id']))}.csv"
    command = shlex.split(template.format(wav=str(wav_path), output=str(csv_path)))
    result = run_command(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"COVAREP command failed: {result.stderr}")
    if not csv_path.exists():
        raise FileNotFoundError(f"COVAREP did not create {csv_path}")
    frame = pd.read_csv(csv_path)
    time_candidates = [name for name in frame.columns if str(name).lower() in {"time", "timestamp", "seconds"}]
    if not time_candidates:
        raise ValueError("COVAREP CSV must contain a time or timestamp column")
    time_column = time_candidates[0]
    feature_columns = [name for name in frame.columns if name != time_column]
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    times = frame[time_column].to_numpy(dtype=np.float32)
    if features.shape[1] != 74:
        raise ValueError(
            f"Configured COVAREP extractor returned {features.shape[1]} dimensions, not 74. "
            "The output will not be mislabeled as COVAREP-74."
        )
    return times, features, [str(name) for name in feature_columns], "COVAREP 74-dimensional acoustic features"


def _extract_opensmile(config: dict[str, Any], wav_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    try:
        import opensmile
    except ImportError as exc:
        raise RuntimeError("openSMILE fallback requested but the opensmile Python package is unavailable") from exc
    feature_set_name = config["audio"].get("opensmile_feature_set", "eGeMAPSv02")
    feature_level_name = config["audio"].get("opensmile_feature_level", "LowLevelDescriptors")
    try:
        feature_set = getattr(opensmile.FeatureSet, feature_set_name)
        feature_level = getattr(opensmile.FeatureLevel, feature_level_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported openSMILE setting: {feature_set_name}/{feature_level_name}"
        ) from exc
    smile = opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
    frame = smile.process_file(str(wav_path))
    if frame.empty:
        raise ValueError(f"openSMILE produced no frames for {wav_path}")
    index_names = list(frame.index.names)
    if "start" in index_names and "end" in index_names:
        starts = frame.index.get_level_values("start").total_seconds().to_numpy(dtype=np.float32)
        ends = frame.index.get_level_values("end").total_seconds().to_numpy(dtype=np.float32)
        times = (starts + ends) / 2.0
    else:
        times = np.arange(len(frame), dtype=np.float32) * 0.01
    features = frame.to_numpy(dtype=np.float32)
    names = [str(name) for name in frame.columns]
    extractor_name = f"openSMILE {feature_set_name} {feature_level_name} ({features.shape[1]} dimensions)"
    return times, features, names, extractor_name


def extract_one_audio_features(config: dict[str, Any], row: pd.Series, force: bool = False) -> Path:
    logger = get_logger()
    sid = str(row["sample_id"])
    destination = audio_feature_path(config, sid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.debug("Acoustic feature cache hit: %s", destination)
        return destination
    wav_path = audio_output_path(config, sid)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file is missing for {sid}: {wav_path}")
    primary = str(config["audio"].get("primary_extractor", "covarep")).lower()
    fallback = str(config["audio"].get("fallback_extractor", "opensmile")).lower()
    error: Exception | None = None
    if primary == "covarep":
        try:
            times, features, names, extractor_name = _extract_covarep(config, row, wav_path)
        except Exception as exc:
            error = exc
            if fallback != "opensmile":
                raise
            logger.warning("COVAREP unavailable for %s: %s. Using openSMILE fallback.", sid, exc)
            times, features, names, extractor_name = _extract_opensmile(config, wav_path)
    elif primary == "opensmile":
        times, features, names, extractor_name = _extract_opensmile(config, wav_path)
    else:
        raise ValueError(f"Unsupported acoustic extractor: {primary}")
    check_finite("audio frame features", features)
    np.savez_compressed(
        destination,
        times=np.asarray(times, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        valid=np.ones(len(times), dtype=np.uint8),
        feature_names=np.asarray(names, dtype=np.str_),
        extractor=np.asarray(extractor_name, dtype=np.str_),
        fallback_reason=np.asarray(str(error) if error else "", dtype=np.str_),
    )
    logger.info("Extracted acoustic features: %s -> %s", sid, extractor_name)
    return destination
