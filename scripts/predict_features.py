"""Adapt Q1 features and run the frozen three-member E7a ensemble."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e7a_core.src.ensemble_inference import aggregate_member_outputs, collect_member_outputs
from e7a_core.src.data import load_pickle_compat
from e7a_core.src.utils import load_config, resolve_model_config, select_device
from src.data.q1_to_e7a_adapter import adapt_q1_sample


class ConvertedDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "text_bert": torch.as_tensor(row["text_bert"], dtype=torch.long),
            "audio": torch.as_tensor(row["audio"], dtype=torch.float32),
            "vision": torch.as_tensor(row["vision"], dtype=torch.float32),
            "valid_mask": torch.as_tensor(row["valid_mask"], dtype=torch.bool),
            "valid_mask_by_modality": torch.as_tensor(row["valid_mask_by_modality"], dtype=torch.bool),
            "observed_mask": torch.as_tensor(row["observed_mask"], dtype=torch.bool),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/e7a_inference.yaml"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config((ROOT / args.config).resolve())
    feature_path = (ROOT / args.features).resolve()
    if not feature_path.is_file():
        parser.error(f"Q1 feature pickle is missing: {feature_path}")
    from transformers import AutoTokenizer

    model_config = resolve_model_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["pretrained_name"],
        use_fast=True,
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    payload = load_pickle_compat(feature_path)
    source_rows = payload["samples"]
    converted = [adapt_q1_sample(sample, tokenizer) for sample in source_rows]
    if len({row["sample_id"] for row in converted}) != len(converted):
        raise ValueError("Duplicate Q1 sample IDs")
    dataset = ConvertedDataset(converted)
    checkpoints = [ROOT / "checkpoints" / f"e7a_seed{seed}_compact.pt" for seed in (42, 17, 2026)]
    if any(not path.is_file() for path in checkpoints):
        parser.error("All three compact E7a checkpoints are required")
    outputs = collect_member_outputs(
        dataset, config, checkpoints, ["seed42", "seed17", "seed2026"],
        select_device(args.device), batch_size=8,
    )
    merged = aggregate_member_outputs(outputs["member_logits"], outputs["member_regression"])
    destination = (ROOT / args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class_id", "regression", "raw_logit_negative", "raw_logit_neutral", "raw_logit_positive", "prob_negative", "prob_neutral", "prob_positive", "observed_text", "observed_audio", "observed_vision", "token_truncated"])
        for index, row in enumerate(converted):
            present = row["observed_mask"].any(axis=0)
            writer.writerow([
                row["sample_id"], int(merged["predictions"][index]),
                float(merged["mean_regression"][index]),
                *map(float, merged["mean_logits"][index]),
                *map(float, merged["probabilities"][index]),
                *map(int, present), int(row["truncated"]),
            ])
    print(f"Predicted {len(converted)} samples: {destination / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
