"""Run the original Q1 WhisperX route with explicit external requirements."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_dir", type=Path)
    source.add_argument("--input_video", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/feature_extraction.yaml"))
    parser.add_argument("--covarep_dir", type=Path, default=Path("precomputed/covarep"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    config = (ROOT / args.config).resolve()
    covarep = (ROOT / args.covarep_dir).resolve()
    if not config.is_file():
        parser.error(f"Config is missing: {config}")
    if not covarep.is_dir() or not list(covarep.glob("*.mat")):
        parser.error("Q1 requires precomputed COVAREP-74 MAT files; provide --covarep_dir. Other acoustic features are not E7a-compatible.")
    for executable in ("ffmpeg", "ffprobe", "FeatureExtraction"):
        if shutil.which(executable) is None:
            parser.error(f"Required external executable is missing: {executable}")
    if args.input_video:
        video = (ROOT / args.input_video).resolve()
        if not video.is_file() or video.suffix.lower() != ".mp4":
            parser.error("--input_video must name an existing MP4")
        source_root = output / "staged_input"
        staged = source_root / video.parent.name / video.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, staged)
    else:
        source_root = (ROOT / args.input_dir).resolve()
        if not source_root.is_dir():
            parser.error(f"Input directory is missing: {source_root}")
        children = [item for item in source_root.iterdir() if item.is_dir()]
        if len(children) == 1 and not list(source_root.glob("*/*.mp4")):
            source_root = children[0]
    if not list(source_root.glob("*/*.mp4")):
        parser.error("Expected video_id/clip_id.mp4 files below the input root")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "versionB.py", "--config", str(config),
        "--dataset-root", str(source_root), "--covarep-dir", str(covarep),
        "--work-dir", str(output / "work"), "--output", str(output / "features.pkl"),
        "--summary", str(output / "feature_summary.csv"),
        "--validation-report", str(output / "validation.json"),
        "--manifest-output", str(output / "manifest.csv"),
        "--asr-summary", str(output / "asr_summary.csv"),
        "--device", args.device,
        "--compute-type", "int8" if args.device == "cpu" else "auto",
    ]
    return subprocess.call(command, cwd=ROOT / "q1_core" / "code")


if __name__ == "__main__":
    raise SystemExit(main())
