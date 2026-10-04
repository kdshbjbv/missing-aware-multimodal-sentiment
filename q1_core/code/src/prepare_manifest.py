from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import file_sha256, get_logger, resolve_executable, run_command, sample_id


REQUIRED_COLUMNS = {"video_id", "clip_id", "text", "label", "annotation"}


def normalize_clip_id(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("clip_id is empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    fps_text = video.get("avg_frame_rate", "0/1") or "0/1"
    try:
        fps = float(Fraction(fps_text))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    frame_count = video.get("nb_frames")
    if frame_count in (None, "N/A") and duration > 0 and fps > 0:
        frame_count = round(duration * fps)
    return {
        "duration_s": duration,
        "fps": fps,
        "frame_count": int(frame_count or 0),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "video_codec": video.get("codec_name", ""),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name", ""),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": int(audio.get("channels", 0) or 0),
    }


def prepare_manifest(config: dict[str, Any], compute_hashes: bool = False) -> pd.DataFrame:
    logger = get_logger()
    dataset_root = Path(config["paths"]["dataset_root"])
    label_path = Path(config["paths"]["label_file"])
    output_path = Path(config["paths"]["work_dir"]) / config["output"]["manifest_file"]
    deliverable_path = Path(config["paths"]["output_dir"]) / config["output"]["manifest_file"]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    ffprobe = resolve_executable(config["paths"]["ffprobe"])
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for the audit stage but was not found")

    frame = pd.read_excel(
        label_path,
        sheet_name=config["data"].get("label_sheet", "label"),
        dtype={"video_id": str},
    )
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Label file is missing columns: {sorted(missing_columns)}")
    frame = frame.loc[:, ["video_id", "clip_id", "text", "label", "annotation"]].copy()
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame["clip_id"] = frame["clip_id"].map(normalize_clip_id)
    frame["text"] = frame["text"].astype(str)
    separator = config["data"].get("sample_id_separator", "$_$")
    frame.insert(0, "sample_id", [sample_id(v, c, separator) for v, c in zip(frame.video_id, frame.clip_id)])
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs: {duplicates}")

    records: list[dict[str, Any]] = []
    label_pairs = set(zip(frame.video_id, frame.clip_id))
    video_pairs = {(path.parent.name, path.stem) for path in dataset_root.rglob("*.mp4")}
    missing_videos = sorted(label_pairs - video_pairs)
    extra_videos = sorted(video_pairs - label_pairs)
    if missing_videos or extra_videos:
        raise ValueError(f"Label/video mismatch. Missing={missing_videos}, extra={extra_videos}")

    for feature_index, row in enumerate(frame.itertuples(index=False)):
        video_path = dataset_root / row.video_id / f"{row.clip_id}.mp4"
        metadata = probe_video(video_path, ffprobe)
        record = {
            "feature_index": feature_index,
            "sample_id": row.sample_id,
            "video_id": row.video_id,
            "clip_id": row.clip_id,
            "video_path": str(video_path),
            "video_relpath": str(video_path.relative_to(dataset_root)),
            "raw_text": row.text,
            "regression_label": float(row.label),
            "annotation": str(row.annotation),
            "classification_label": int(config["data"]["classification_mapping"][str(row.annotation)]),
            "source_size_bytes": video_path.stat().st_size,
            **metadata,
        }
        record["source_sha256"] = file_sha256(video_path) if compute_hashes else ""
        records.append(record)
    result = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(deliverable_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Dataset audit passed: %d labels, %d videos, %d unique sample IDs",
        len(result),
        len(video_pairs),
        result.sample_id.nunique(),
    )
    logger.info("Saved manifest deliverable to %s", deliverable_path)
    return result


def load_manifest(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["paths"]["work_dir"]) / config["output"]["manifest_file"]
    if not path.exists():
        return prepare_manifest(config)
    return pd.read_csv(path, dtype={"video_id": str, "clip_id": str}, encoding="utf-8-sig")
