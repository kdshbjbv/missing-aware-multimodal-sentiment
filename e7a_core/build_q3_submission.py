from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import resolve_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge attachment-4 predictions and model-level explanations into submission CSVs."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--regression_shapley", required=True)
    parser.add_argument("--regression_ig", required=True)
    parser.add_argument("--class_shapley", required=True)
    parser.add_argument("--class_ig", required=True)
    parser.add_argument("--class_biases", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_csv(path: str, name: str) -> pd.DataFrame:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {name}: {resolved}")
    return pd.read_csv(resolved, dtype={"sample_id": str})


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    if "sample_id" not in frame:
        raise ValueError(f"{name} lacks sample_id")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate sample_id rows")


def top_positions(frame: pd.DataFrame, top_k: int) -> dict[str, str]:
    required = {"sample_id", "modality", "aligned_position", "mean_member_absolute_importance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IG file lacks columns: {sorted(missing)}")
    result = {}
    for sample_id, group in frame.groupby("sample_id", sort=False):
        ordered = group.sort_values("mean_member_absolute_importance", ascending=False).head(top_k)
        result[str(sample_id)] = ";".join(
            f"{row.modality}:{int(row.aligned_position)}:{float(row.mean_member_absolute_importance):.6g}"
            for row in ordered.itertuples()
        )
    return result


def prefixed_shapley(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        column for column in (
            "sample_id", "phi_T", "phi_A", "phi_V", "additivity_error",
            "primary_modality_by_abs_phi",
        )
        if column in frame.columns
    ]
    result = frame.loc[:, keep].copy()
    return result.rename(
        columns={column: f"{prefix}_{column}" for column in keep if column != "sample_id"}
    )


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    predictions = read_csv(args.predictions, "predictions")
    diagnostics = read_csv(args.diagnostics, "diagnostics")
    regression_shapley = read_csv(args.regression_shapley, "regression Shapley")
    regression_ig = read_csv(args.regression_ig, "regression IG")
    class_shapley = read_csv(args.class_shapley, "class-logit Shapley")
    class_ig = read_csv(args.class_ig, "class-logit IG")
    assert_unique(predictions, "predictions")
    assert_unique(diagnostics, "diagnostics")
    if len(predictions) != 20:
        raise ValueError(f"Attachment 4 must contain 20 predictions, got {len(predictions)}")

    summary = predictions.merge(diagnostics, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_diagnostic"))
    summary = summary.merge(prefixed_shapley(regression_shapley, "regression"), on="sample_id", how="left", validate="one_to_one")
    summary = summary.merge(prefixed_shapley(class_shapley, "class_logit"), on="sample_id", how="left", validate="one_to_one")
    regression_top = top_positions(regression_ig, args.top_k)
    class_top = top_positions(class_ig, args.top_k)
    summary["regression_top_aligned_evidence"] = summary["sample_id"].map(regression_top).fillna("")
    summary["class_logit_top_aligned_evidence"] = summary["sample_id"].map(class_top).fillna("")

    if any(abs(float(value)) > 1e-12 for value in args.class_biases):
        raise ValueError("Frozen E7a submission requires class_biases=[0,0,0]")
    label_names = ("Negative", "Neutral", "Positive")
    probability_columns = []
    for class_index, class_name in enumerate(label_names):
        probability_column = f"prob_{class_name}"
        if probability_column in summary:
            probability_columns.append(probability_column)
    if len(probability_columns) == 3:
        summary["max_class_probability"] = summary.loc[:, probability_columns].max(axis=1)
    member_columns = [
        column for column in summary
        if column.endswith("_pred_class_index")
        and column not in {"pred_class_index", "raw_pred_class_index"}
    ]
    if member_columns:
        member_votes = summary.loc[:, member_columns].to_numpy(dtype=np.int64)
        summary["member_unanimous"] = np.all(member_votes == member_votes[:, [0]], axis=1)
        summary["member_agreement_count"] = np.max(
            np.stack([(member_votes == label).sum(axis=1) for label in range(3)], axis=1),
            axis=1,
        )
    else:
        summary["member_unanimous"] = False
        summary["member_agreement_count"] = np.nan
    summary["formal_explanation_status"] = np.where(
        summary["complete_three_modality"].astype(bool),
        "regression_and_class_logit_explained",
        "prediction_and_missing_diagnosis_only",
    )
    summary["evidence_scope"] = "aligned_position_only"

    regression_long = regression_ig.copy()
    regression_long.insert(1, "explanation_target", "ensemble_mean_regression")
    class_long = class_ig.copy()
    class_long.insert(1, "explanation_target", "final_predicted_class_raw_logit")
    evidence = pd.concat([regression_long, class_long], ignore_index=True, sort=False)

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "attachment4_predictions_and_explanations.csv"
    evidence_path = output / "attachment4_local_evidence_long.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    validation = {
        "summary_file": str(summary_path),
        "evidence_file": str(evidence_path),
        "summary_rows": int(len(summary)),
        "unique_sample_ids": int(summary["sample_id"].nunique()),
        "complete_explained_rows": int((summary["formal_explanation_status"] == "regression_and_class_logit_explained").sum()),
        "limited_rows": int((summary["formal_explanation_status"] == "prediction_and_missing_diagnosis_only").sum()),
        "limited_sample_ids": summary.loc[
            summary["formal_explanation_status"] == "prediction_and_missing_diagnosis_only", "sample_id"
        ].astype(str).tolist(),
        "local_evidence_rows": int(len(evidence)),
        "class_biases": args.class_biases,
        "mapping_scope": "aligned positions only; no claim of original media timestamps or frames",
    }
    save_json(validation, output / "attachment4_submission_validation.json")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
