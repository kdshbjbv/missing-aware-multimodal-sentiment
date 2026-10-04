from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import get_logger, resolve_executable, run_command, safe_sample_name


def audio_output_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "audio" / f"{safe_sample_name(sample_id)}.wav"


def extract_one_audio(config: dict[str, Any], row: pd.Series, force: bool = False) -> Path:
    logger = get_logger()
    ffmpeg = resolve_executable(config["paths"]["ffmpeg"])
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found")
    destination = audio_output_path(config, row["sample_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 44 and not force:
        logger.debug("Audio cache hit: %s", destination)
        return destination
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(row["video_path"]),
        "-vn",
        "-ac",
        str(config["audio"]["channels"]),
        "-ar",
        str(config["audio"]["sample_rate"]),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    result = run_command(command, check=False)
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"ffmpeg failed for {row['sample_id']}: {result.stderr.strip()}")
    logger.info("Extracted audio: %s", row["sample_id"])
    return destination
