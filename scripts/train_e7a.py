"""Public wrapper around the unchanged two-stage E7a trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/attachment2"))
    parser.add_argument("--config", type=Path, default=Path("configs/e7a_train.yaml"))
    parser.add_argument("--seed", type=int, choices=(42, 17, 2026), required=True)
    args = parser.parse_args()
    data = (ROOT / args.data_dir / "aligned_50.pkl").resolve()
    config = (ROOT / args.config).resolve()
    if not data.is_file():
        parser.error(f"Aligned attachment 2 is missing: {data}")
    if not config.is_file():
        parser.error(f"Config is missing: {config}")
    return subprocess.call(
        [sys.executable, "train.py", "--data_path", str(data), "--config", str(config), "--seed", str(args.seed)],
        cwd=ROOT / "e7a_core",
    )


if __name__ == "__main__":
    raise SystemExit(main())
