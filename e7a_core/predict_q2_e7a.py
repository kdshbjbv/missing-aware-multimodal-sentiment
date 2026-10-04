from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.calibration import logits_to_probabilities
from src.data import Attachment3Dataset
from src.ensemble_inference import (
    aggregate_member_outputs,
    collect_member_outputs,
    validate_members,
)
from src.masks import mask_intervals
from src.utils import (
    environment_manifest,
    load_config,
    resolve_path,
    save_json,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen E7a raw logits ensemble inference on attachment 3."
    )
    parser.add_argument("--config", default="configs/e7a_member.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--member_names", nargs="+", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    checkpoint_paths = [resolve_path(value) for value in args.checkpoints]
    member_names = validate_members(checkpoint_paths, args.member_names)
    # E7a is frozen as an uncalibrated raw logits ensemble with zero class bias.
    class_biases = [0.0, 0.0, 0.0]
    data_dir = resolve_path(args.data_dir or config["data"]["attachment3_aligned"])
    dataset = Attachment3Dataset(data_dir)
    collected = collect_member_outputs(
        dataset,
        config,
        checkpoint_paths,
        member_names,
        device,
        args.batch_size,
    )
    aggregate = aggregate_member_outputs(
        collected["member_logits"],
        collected["member_regression"],
    )
    label_mapping = {
        int(key): value for key, value in config["data"]["label_mapping"].items()
    }
    rows = []
    for index, sample_id in enumerate(collected["sample_ids"]):
        row = {
            "sample_id": sample_id,
            "pred_class_index": int(aggregate["predictions"][index]),
            "pred_class_name": label_mapping[int(aggregate["predictions"][index])],
            "raw_pred_class_index": int(aggregate["raw_predictions"][index]),
            "pred_regression": float(aggregate["mean_regression"][index]),
            "pred_regression_clipped": float(
                np.clip(aggregate["mean_regression"][index], -3.0, 3.0)
            ),
            "regression_was_clipped": bool(
                aggregate["mean_regression"][index] < -3.0
                or aggregate["mean_regression"][index] > 3.0
            ),
        }
        for class_index in range(3):
            class_name = label_mapping[class_index]
            row[f"ensemble_logit_{class_name}"] = float(
                aggregate["mean_logits"][index, class_index]
            )
            row[f"raw_prob_{class_name}"] = float(
                aggregate["raw_probabilities"][index, class_index]
            )
            row[f"prob_{class_name}"] = float(
                aggregate["probabilities"][index, class_index]
            )
        for member_index, member_name in enumerate(member_names):
            member_logits = collected["member_logits"][member_index, index]
            _, member_probability, member_prediction = logits_to_probabilities(
                member_logits[None, :]
            )
            row[f"{member_name}_pred_class_index"] = int(member_prediction[0])
            row[f"{member_name}_pred_regression"] = float(
                collected["member_regression"][member_index, index]
            )
            row[f"{member_name}_confidence"] = float(member_probability[0].max())
        rows.append(row)

    diagnostics = []
    for index in range(len(dataset)):
        item = dataset[index]
        observed = item["observed_mask"].numpy()
        applicable = item["valid_mask_by_modality"].numpy()
        diagnostics.append(
            {
                "sample_id": item["sample_id"],
                "valid_length": int(item["valid_mask"].sum()),
                "text_missing_intervals": str(
                    mask_intervals(applicable[:, 0] & ~observed[:, 0])
                ),
                "audio_missing_intervals": str(
                    mask_intervals(applicable[:, 1] & ~observed[:, 1])
                ),
                "vision_missing_intervals": str(
                    mask_intervals(applicable[:, 2] & ~observed[:, 2])
                ),
                "any_member_used_null_prior": bool(
                    collected["member_null_prior"][:, index].any()
                ),
                "text_missing_encoding_status": (
                    "no explicit marker observed; valid tokens treated as observed"
                ),
            }
        )

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output / "q2_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(diagnostics).to_csv(
        output / "q2_missing_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    save_json(
        {
            "method": "equal_weight_mean_raw_logits_no_class_bias",
            "members": collected["members"],
            "class_biases": class_biases,
            "regression_method": "equal_weight_mean",
            "data_dir": str(data_dir),
            "sample_count": len(dataset),
            "input_order_preserved": True,
            "environment": environment_manifest(),
        },
        output / "q2_run_manifest.json",
    )
    print(output / "q2_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
