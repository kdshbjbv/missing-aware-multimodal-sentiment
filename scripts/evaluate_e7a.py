"""Public wrapper around the unchanged E7a evaluator."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/attachment2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--config", type=Path, default=Path("configs/e7a_inference.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    data = (ROOT / args.data_dir / "aligned_50.pkl").resolve()
    config = (ROOT / args.config).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    for path in (data, config, checkpoint):
        if not path.is_file():
            parser.error(f"Required file is missing: {path}")
    return subprocess.call(
        [sys.executable, "evaluate.py", "--data_path", str(data), "--config", str(config),
         "--checkpoint", str(checkpoint), "--split", args.split,
         "--output_dir", str((ROOT / args.output_dir).resolve())],
        cwd=ROOT / "e7a_core",
    )


if __name__ == "__main__":
    raise SystemExit(main())
