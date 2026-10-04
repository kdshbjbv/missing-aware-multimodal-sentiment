"""Report whether core tests or the full Q1 media pipeline can run."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("core", "video"), default="core")
    args = parser.parse_args()
    missing = []
    for module in ("numpy", "torch", "transformers", "yaml", "pytest"):
        if importlib.util.find_spec(module) is None:
            missing.append(f"Python package: {module}")
    if args.mode == "video":
        for executable in ("ffmpeg", "ffprobe", "FeatureExtraction"):
            if shutil.which(executable) is None:
                missing.append(f"Executable: {executable}")
        for module in ("whisperx", "soundfile", "librosa"):
            if importlib.util.find_spec(module) is None:
                missing.append(f"Python package: {module}")
        if not (ROOT / "precomputed" / "covarep").is_dir():
            missing.append("Precomputed COVAREP MAT directory")
        if not (ROOT / "pretrained" / "bert-base-uncased").is_dir():
            missing.append("Complete BERT model directory")
    print(f"Python {sys.version.split()[0]}; mode={args.mode}")
    if missing:
        for item in missing:
            print(f"MISSING: {item}")
        return 1
    print("INSTALLATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
