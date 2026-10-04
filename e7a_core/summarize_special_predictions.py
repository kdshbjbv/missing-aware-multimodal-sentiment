from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize final attachment 3/4 predictions."
    )
    parser.add_argument("--q2", required=True, help="Attachment 3 q2_predictions.csv")
    parser.add_argument("--q3", required=True, help="Attachment 4 q3_predictions.csv")
    return parser.parse_args()


def summarize(path: Path, expected_rows: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "sample_id", "pred_class_index", "pred_class_name",
        "pred_regression", "pred_regression_clipped",
        "prob_Negative", "prob_Neutral", "prob_Positive",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"{path} expected {expected_rows} rows, got {len(frame)}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate sample IDs")
    probabilities = frame[
        ["prob_Negative", "prob_Neutral", "prob_Positive"]
    ].to_numpy(dtype=np.float64)
    regression = frame["pred_regression"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or not np.isfinite(regression).all():
        raise ValueError(f"{path} contains non-finite predictions")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{path} probabilities do not sum to one")
    member_columns = [
        column for column in frame.columns
        if column.endswith("_pred_class_index")
        and column not in {"pred_class_index", "raw_pred_class_index"}
    ]
    unanimous = None
    if member_columns:
        member_predictions = frame[member_columns].to_numpy(dtype=np.int64)
        unanimous = float(
            np.mean(np.all(member_predictions == member_predictions[:, :1], axis=1))
        )
    result = {
        "file": str(path.resolve()),
        "rows": int(len(frame)),
        "unique_ids": int(frame["sample_id"].astype(str).nunique()),
        "predicted_class_counts": {
            str(key): int(value)
            for key, value in frame["pred_class_index"].value_counts().sort_index().items()
        },
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "regression_min": float(regression.min()),
        "regression_max": float(regression.max()),
        "regression_clipped_count": int(frame["regression_was_clipped"].sum()),
        "member_unanimous_fraction": unanimous,
    }
    if "complete_three_modality" in frame:
        complete = frame["complete_three_modality"].astype(bool)
        result["complete_three_modality_count"] = int(complete.sum())
        result["incomplete_sample_ids"] = frame.loc[
            ~complete, "sample_id"
        ].astype(str).tolist()
    return result


def main() -> int:
    args = parse_args()
    report = {
        "attachment3": summarize(Path(args.q2), 30),
        "attachment4": summarize(Path(args.q3), 20),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
