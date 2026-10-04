from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extract_audio_features import audio_feature_path
from .extract_text_bert import text_feature_path
from .extract_visual_openface import vision_feature_path
from .utils import check_finite, get_logger, safe_sample_name


def aligned_feature_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "aligned" / f"{safe_sample_name(sample_id)}.npz"


def pool_points_by_windows(
    frame_times: np.ndarray,
    frame_features: np.ndarray,
    windows: np.ndarray,
    valid: np.ndarray | None = None,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    frame_features = np.asarray(frame_features, dtype=np.float32)
    windows = np.asarray(windows, dtype=np.float64)
    if frame_features.ndim != 2 or len(frame_features) != len(frame_times):
        raise ValueError("Frame times and features have inconsistent shapes")
    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError("windows must have shape [L, 2]")
    valid_array = np.ones(len(frame_times), dtype=bool) if valid is None else np.asarray(valid).astype(bool)
    output = np.zeros((len(windows), frame_features.shape[1]), dtype=np.float32)
    mask = np.zeros(len(windows), dtype=np.uint8)
    counts = np.zeros(len(windows), dtype=np.int32)
    for index, (start, end) in enumerate(windows):
        if end <= start:
            continue
        in_window = (frame_times >= start) & (frame_times < end) & valid_array
        if index == len(windows) - 1:
            in_window = (frame_times >= start) & (frame_times <= end) & valid_array
        selected = frame_features[in_window]
        if selected.size == 0:
            continue
        if method == "mean":
            output[index] = selected.mean(axis=0)
        elif method == "median":
            output[index] = np.median(selected, axis=0)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")
        mask[index] = 1
        counts[index] = len(selected)
    check_finite("pooled features", output)
    return output, mask, counts


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def align_one_sample(config: dict[str, Any], row: pd.Series, force: bool = False) -> Path:
    logger = get_logger()
    sid = str(row["sample_id"])
    destination = aligned_feature_path(config, sid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.debug("Aligned feature cache hit: %s", destination)
        return destination

    text_data = _load_npz(text_feature_path(config, sid))
    audio_data = _load_npz(audio_feature_path(config, sid))
    vision_data = _load_npz(vision_feature_path(config, sid))
    words = text_data["words"].astype(np.str_)
    timestamps = text_data["timestamps"].astype(np.float32)
    text = text_data["features"].astype(np.float32)
    if len(words) != len(timestamps) or len(text) != len(timestamps):
        raise ValueError(f"Text/MFA length mismatch for {sid}")

    method = str(config["alignment"].get("pooling", "mean"))
    audio, audio_mask, audio_counts = pool_points_by_windows(
        audio_data["times"], audio_data["features"], timestamps, audio_data.get("valid"), method
    )
    vision, vision_mask, vision_counts = pool_points_by_windows(
        vision_data["times"], vision_data["features"], timestamps, vision_data.get("valid"), method
    )
    text_mask = np.ones(len(words), dtype=np.uint8)

    max_len = int(config["project"]["max_len"])
    original_length = len(words)
    truncated_count = max(0, original_length - max_len)
    if truncated_count:
        strategy = str(config["alignment"].get("overlength_strategy", "truncate"))
        if strategy != "truncate":
            raise ValueError(f"Unsupported overlength strategy: {strategy}")
        logger.warning("Truncating %s from %d to %d word positions", sid, original_length, max_len)
    valid_length = min(original_length, max_len)

    def pad_matrix(array: np.ndarray, width: int, fill: float = 0.0) -> np.ndarray:
        output = np.full((max_len, width), fill, dtype=array.dtype)
        output[:valid_length] = array[:valid_length]
        return output

    def pad_vector(array: np.ndarray, fill: int = 0) -> np.ndarray:
        output = np.full(max_len, fill, dtype=array.dtype)
        output[:valid_length] = array[:valid_length]
        return output

    padded_words = np.full(max_len, "<PAD>", dtype=f"<U{max(5, max((len(word) for word in words), default=5))}")
    padded_words[:valid_length] = words[:valid_length]
    timestamp_fill = float(config["alignment"].get("timestamp_padding_value", -1.0))
    padded_timestamps = np.full((max_len, 2), timestamp_fill, dtype=np.float32)
    padded_timestamps[:valid_length] = timestamps[:valid_length]
    sequence_mask = np.zeros(max_len, dtype=np.uint8)
    sequence_mask[:valid_length] = 1

    text = pad_matrix(text, text.shape[1])
    audio = pad_matrix(audio, audio.shape[1])
    vision = pad_matrix(vision, vision.shape[1])
    text_mask = pad_vector(text_mask)
    audio_mask = pad_vector(audio_mask)
    vision_mask = pad_vector(vision_mask)
    audio_counts = pad_vector(audio_counts)
    vision_counts = pad_vector(vision_counts)
    for name, array in (("text", text), ("audio", audio), ("vision", vision)):
        check_finite(name, array)

    np.savez_compressed(
        destination,
        sample_id=np.asarray(sid, dtype=np.str_),
        words=padded_words,
        timestamps=padded_timestamps,
        text=text,
        audio=audio,
        vision=vision,
        sequence_mask=sequence_mask,
        text_mask=text_mask,
        audio_mask=audio_mask,
        vision_mask=vision_mask,
        audio_frame_counts=audio_counts,
        vision_frame_counts=vision_counts,
        valid_length=np.asarray(valid_length, dtype=np.int16),
        original_length=np.asarray(original_length, dtype=np.int16),
        truncated_count=np.asarray(truncated_count, dtype=np.int16),
        text_extractor=text_data["extractor"],
        audio_extractor=audio_data["extractor"],
        vision_extractor=vision_data["extractor"],
        audio_feature_names=audio_data["feature_names"],
        vision_feature_names=vision_data["feature_names"],
    )
    logger.info(
        "Aligned %s: L=%d, text=%s, audio=%s, vision=%s",
        sid,
        valid_length,
        tuple(text.shape),
        tuple(audio.shape),
        tuple(vision.shape),
    )
    return destination
