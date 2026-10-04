"""Check the aligned attachment 2 training interface and save a summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_attachment2 import summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/attachment2"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/attachment2_prepared"))
    parser.add_argument("--config", type=Path, default=Path("configs/e7a_train.yaml"))
    args = parser.parse_args()
    path = ROOT / args.input / "aligned_50.pkl"
    if not path.is_file() or not (ROOT / args.config).is_file():
        parser.error("Aligned pickle or config is missing")
    report = summary(path)
    if set(report["splits"]) != {"train", "valid", "test"}:
        raise ValueError("Expected original train/valid/test splits")
    destination = ROOT / args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "schema_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated original splits; summary: {destination / 'schema_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
