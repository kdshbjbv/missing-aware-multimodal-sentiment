from __future__ import annotations

import argparse
from pathlib import Path

from src.config import read_config
from src.problem1_dataset import (
    build_problem1_dataset,
    generate_problem1_figures,
    save_problem1_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the final MFA-aligned BERT/COVAREP/OpenFace problem-1 dataset."
    )
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--sample-id", action="append", help="Build/check only selected sample IDs")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build in memory and print sample diagnostics without writing deliverables",
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    dataset, summary = build_problem1_dataset(config, args.sample_id)
    print(summary.to_string(index=False))
    if args.check_only:
        print(f"CHECK_ONLY_OK: {len(dataset['samples'])} sample(s)")
        return
    dataset_path, summary_path = save_problem1_dataset(config, dataset, summary)
    print(f"Saved dataset: {dataset_path}")
    print(f"Saved summary: {summary_path}")
    if not args.skip_figures:
        for name, path in generate_problem1_figures(config, dataset).items():
            print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
