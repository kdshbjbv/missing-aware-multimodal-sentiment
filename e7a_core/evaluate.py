from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data import Attachment2Dataset
from src.engine import run_epoch
from src.masks import ArtificialMaskGenerator
from src.model import UnifiedSentimentModel
from src.utils import (
    load_config,
    resolve_model_config,
    resolve_path,
    save_json,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint on attachment 2.")
    parser.add_argument("--config", default="configs/e7a_member.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], default="valid"
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--mask_mode",
        choices=["auto", "complete", "broad", "attachment3"],
        default="auto",
        help="Evaluation protocol. auto preserves the checkpoint-stage behavior.",
    )
    return parser.parse_args()


def build_mask_generator(mask_cfg: dict, seed: int) -> ArtificialMaskGenerator:
    return ArtificialMaskGenerator(
        complete_probability=float(mask_cfg["complete_sample_probability"]),
        missing_ratios=tuple(mask_cfg["missing_ratios"]),
        combinations=tuple(mask_cfg["modality_combinations"]),
        seed=seed,
        synchronize_modalities=bool(mask_cfg.get("synchronize_modalities", False)),
        missing_ratio_probabilities=mask_cfg.get("missing_ratio_probabilities"),
        max_block_length=int(mask_cfg.get("max_block_length", 15)),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]), True)
    device = select_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UnifiedSentimentModel(**resolve_model_config(config)).to(device)
    compact = bool(checkpoint.get("compact_frozen_text_encoder", False))
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=not compact
    )
    if compact:
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("text_encoder.")
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                "Compact checkpoint mismatch: "
                f"unexpected={unexpected}, invalid_missing={invalid_missing}"
            )
    model.eval()
    data_path = resolve_path(args.data_path or config["data"]["attachment2"])
    dataset = Attachment2Dataset(data_path, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    stage = int(checkpoint.get("stage", 2))
    variant = str(checkpoint.get("variant", "E3"))
    mask_mode = args.mask_mode
    if mask_mode == "auto":
        mask_mode = "broad" if stage == 2 else "complete"
    mask_generator = None
    artificial_masking = mask_mode != "complete"
    if mask_mode == "broad":
        mask_cfg = config.get("evaluation", {}).get(
            "broad_masking", config["masking"]
        )
        mask_generator = build_mask_generator(
            mask_cfg, int(mask_cfg["valid_replay_seed"])
        )
    elif mask_mode == "attachment3":
        mask_cfg = config["evaluation"]["attachment3_masking"]
        mask_generator = build_mask_generator(
            mask_cfg, int(mask_cfg["valid_replay_seed"])
        )
    metrics = run_epoch(
        model, loader, device, config["loss"], stage, variant, 0,
        mask_generator=mask_generator,
        artificial_masking=artificial_masking,
    )
    output = resolve_path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    scalar = {key: value for key, value in metrics.items() if key != "predictions"}
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_stage": stage,
        "variant": variant,
        "split": args.split,
        "mask_mode": mask_mode,
        **scalar,
    }
    save_json(report, output / f"{args.split}_{mask_mode}_metrics.json")
    pd.DataFrame(metrics["predictions"]).to_csv(
        output / f"{args.split}_{mask_mode}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
