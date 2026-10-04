from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize attachment 4 ensemble explanations."
    )
    parser.add_argument("--input_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    shapley = pd.read_csv(root / "q3_modality_shapley.csv", dtype={"sample_id": str})
    ig = pd.read_csv(root / "q3_position_ig.csv", dtype={"sample_id": str})
    gate = pd.read_csv(root / "q3_gate_weights.csv", dtype={"sample_id": str})
    faith = pd.read_csv(root / "q3_faithfulness.csv", dtype={"sample_id": str})
    with (root / "q3_explanation_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_explained = int(manifest["explained_complete_samples"])
    shapley_ids = set(shapley["sample_id"].astype(str))
    ig_ids = set(ig["sample_id"].astype(str))
    faith_ids = set(faith["sample_id"].astype(str))
    if not (len(shapley) == expected_explained == len(shapley_ids)):
        raise ValueError("Shapley output does not contain one row per complete sample")
    if ig_ids != shapley_ids or faith_ids != shapley_ids:
        raise ValueError("IG/faithfulness sample IDs do not match Shapley IDs")
    if gate["sample_id"].astype(str).nunique() != int(manifest["sample_count"]):
        raise ValueError("Gate output does not cover every attachment 4 sample")
    grouped_ig_error = ig.groupby("sample_id")[
        "mean_ig_completeness_error"
    ].first()
    top = faith[faith["selection"] == "top"].set_index("sample_id")
    random_mean = faith[faith["selection"] == "random"].groupby("sample_id")[
        "absolute_change"
    ].mean()
    comparison = top["absolute_change"].to_frame("top").join(
        random_mean.rename("random")
    )
    result = {
        "input_dir": str(root),
        "total_samples": int(manifest["sample_count"]),
        "explained_complete_samples": expected_explained,
        "blocked_incomplete_sample_ids": manifest["blocked_incomplete_sample_ids"],
        "primary_modality_counts": {
            str(key): int(value)
            for key, value in shapley["primary_modality_by_abs_phi"].value_counts().items()
        },
        "max_abs_shapley_additivity_error": float(
            shapley["additivity_error"].abs().max()
        ),
        "mean_abs_ig_completeness_error": float(grouped_ig_error.abs().mean()),
        "max_abs_ig_completeness_error": float(grouped_ig_error.abs().max()),
        "mean_top_occlusion_change": float(comparison["top"].mean()),
        "mean_random_occlusion_change": float(comparison["random"].mean()),
        "top_exceeds_random_fraction": float(
            np.mean(comparison["top"] > comparison["random"])
        ),
        "gate_sample_count": int(gate["sample_id"].astype(str).nunique()),
        "figure_count": len(list((root / "figures").glob("*.png"))),
        "explanation_card_count": len(
            list((root / "explanation_cards").glob("*.md"))
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
