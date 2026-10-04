"""Raw-video entry point when all Q1 external tools and models are installed."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--feature_config", type=Path, default=Path("configs/feature_extraction.yaml"))
    parser.add_argument("--model_config", type=Path, default=Path("configs/e7a_inference.yaml"))
    parser.add_argument("--covarep_dir", type=Path, default=Path("precomputed/covarep"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    feature_output = output / "features"
    command = [
        sys.executable, str(ROOT / "scripts" / "extract_features.py"),
        "--output_dir", str(feature_output), "--config", str(args.feature_config),
        "--covarep_dir", str(args.covarep_dir), "--device", args.device,
    ]
    command += ["--input_video", str(args.input_video)] if args.input_video else ["--input_dir", str(args.input_dir)]
    extraction = subprocess.call(command, cwd=ROOT)
    if extraction:
        return extraction
    return subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "predict_features.py"),
         "--features", str(feature_output / "features.pkl"),
         "--output_dir", str(output / "prediction"), "--config", str(args.model_config)],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
