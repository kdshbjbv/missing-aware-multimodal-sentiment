from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.calibration import logits_to_probabilities, mean_ensemble_logits
from src.metrics import task_metrics
from src.utils import resolve_path, save_json


MODES = ("complete", "broad", "attachment3")
LOGIT_COLUMNS = ("logit_class_0", "logit_class_1", "logit_class_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen E7a: equal-weight average of three members' raw logits."
    )
    parser.add_argument(
        "--evaluation_dirs", nargs="+", required=True,
        help="Directories containing <split>_<mode>_predictions.csv files.",
    )
    parser.add_argument(
        "--member_names", nargs="+", default=None,
        help="Optional names matching evaluation_dirs.",
    )
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], default="valid"
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES),
        help="Evaluation masking modes to ensemble.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_mode_frames(
    directories: list[Path], mode: str, split: str
) -> list[pd.DataFrame]:
    required = {
        "sample_id", "true_class", "true_regression", "pred_regression",
        *LOGIT_COLUMNS,
    }
    frames = []
    expected_ids = None
    expected_true = None
    for directory in directories:
        path = directory / f"{split}_{mode}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ensemble prediction file: {path}")
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        ids = frame["sample_id"].astype(str).tolist()
        true = frame["true_class"].astype(int).tolist()
        if expected_ids is None:
            expected_ids, expected_true = ids, true
        elif ids != expected_ids or true != expected_true:
            raise ValueError(
                f"All ensemble members must contain identical ordered samples for {mode}"
            )
        frames.append(frame)
    return frames


def main() -> int:
    args = parse_args()
    directories = [resolve_path(value) for value in args.evaluation_dirs]
    names = args.member_names or [directory.parent.name for directory in directories]
    if len(names) != len(directories):
        raise ValueError("member_names must match evaluation_dirs length")
    if len(directories) != 3:
        raise ValueError("Frozen E7a ensembling requires exactly three directories")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_biases = [0.0, 0.0, 0.0]

    summary = {
        "method": "equal_weight_mean_raw_logits_no_class_bias",
        "split": args.split,
        "class_biases": class_biases,
        "members": [
            {"name": name, "evaluation_dir": str(directory)}
            for name, directory in zip(names, directories)
        ],
        "modes": {},
    }
    for mode in args.modes:
        frames = load_mode_frames(directories, mode, args.split)
        ensemble_logits = mean_ensemble_logits(
            [frame.loc[:, LOGIT_COLUMNS].to_numpy(dtype=np.float64) for frame in frames]
        )
        _, probabilities, predictions = logits_to_probabilities(ensemble_logits)
        true_class = frames[0]["true_class"].to_numpy(dtype=np.int64)
        true_regression = frames[0]["true_regression"].to_numpy(dtype=np.float64)
        pred_regression = np.mean(
            np.stack(
                [frame["pred_regression"].to_numpy(dtype=np.float64) for frame in frames],
                axis=0,
            ),
            axis=0,
        )
        metrics = task_metrics(
            true_class, predictions, true_regression, pred_regression
        )
        summary["modes"][mode] = metrics
        prediction_frame = pd.DataFrame(
            {
                "sample_id": frames[0]["sample_id"].astype(str),
                "true_class": true_class,
                "pred_class": predictions,
                "true_regression": true_regression,
                "pred_regression": pred_regression,
                **{
                    f"logit_class_{index}": ensemble_logits[:, index]
                    for index in range(3)
                },
                **{
                    f"prob_class_{index}": probabilities[:, index]
                    for index in range(3)
                },
            }
        )
        prediction_frame.to_csv(
            output_dir / f"{args.split}_{mode}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_json(
            {
                "mask_mode": mode,
                "split": args.split,
                "ensemble_method": "equal_weight_mean_raw_logits_no_class_bias",
                "member_names": names,
                "class_biases": class_biases,
                **metrics,
            },
            output_dir / f"{args.split}_{mode}_metrics.json",
        )
    save_json(summary, output_dir / "ensemble_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
