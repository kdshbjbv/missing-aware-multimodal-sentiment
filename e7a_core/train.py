from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data import Attachment2Dataset
from src.engine import run_epoch
from src.masks import (
    ArtificialMaskGenerator,
    HybridArtificialMaskGenerator,
    MaskGenerator,
)
from src.model import UnifiedSentimentModel
from src.utils import (
    create_logger,
    environment_manifest,
    load_config,
    make_run_dir,
    resolve_path,
    resolve_model_config,
    save_json,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified Q2/Q3 model in two stages.")
    parser.add_argument("--config", default="configs/e7a_member.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--variant", choices=["E0", "E1", "E2", "E3"], default="E3")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random-seed override saved into the resolved run config.",
    )
    return parser.parse_args()


def freeze_stage2_targets(model: UnifiedSentimentModel, enabled: bool) -> None:
    if not enabled:
        return
    for module in (
        model.text_projection,
        model.audio_projection,
        model.vision_projection,
    ):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def configure_stage_trainability(
    model: UnifiedSentimentModel, stage: int, training: Dict[str, Any]
) -> None:
    if stage != 2:
        return
    freeze_stage2_targets(
        model, bool(training["freeze_projections_in_stage2"])
    )
    if bool(training.get("freeze_bert_in_stage2", False)):
        model.freeze_text_encoder()


def _build_single_mask_generator(
    mask_config: Dict[str, Any], seed: int, complete_probability: float
) -> ArtificialMaskGenerator:
    return ArtificialMaskGenerator(
        complete_probability=complete_probability,
        missing_ratios=tuple(mask_config["missing_ratios"]),
        combinations=tuple(mask_config["modality_combinations"]),
        seed=seed,
        synchronize_modalities=bool(
            mask_config.get("synchronize_modalities", False)
        ),
        missing_ratio_probabilities=mask_config.get(
            "missing_ratio_probabilities"
        ),
        max_block_length=int(mask_config.get("max_block_length", 15)),
    )


def build_mask_generator(mask_config: Dict[str, Any], seed: int) -> MaskGenerator:
    profile_configs = mask_config.get("profiles")
    if not profile_configs:
        return _build_single_mask_generator(
            mask_config,
            seed,
            float(mask_config["complete_sample_probability"]),
        )
    profiles = []
    profile_probabilities = []
    for index, profile_config in enumerate(profile_configs):
        profiles.append(
            _build_single_mask_generator(
                profile_config,
                seed + 1009 * (index + 1),
                complete_probability=0.0,
            )
        )
        profile_probabilities.append(float(profile_config["probability"]))
    return HybridArtificialMaskGenerator(
        profiles=tuple(profiles),
        profile_probabilities=tuple(profile_probabilities),
        complete_probability=float(mask_config["complete_sample_probability"]),
        seed=seed,
    )


def save_checkpoint(
    path: Path,
    model: UnifiedSentimentModel,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    stage: int,
    epoch: int,
    variant: str,
    metrics: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stage": stage,
            "epoch": epoch,
            "variant": variant,
            "validation_metrics": {
                key: value for key, value in metrics.items() if key != "predictions"
            },
            "label_mapping": config["data"]["label_mapping"],
            "pretrained_name": config["model"]["pretrained_name"],
            "freeze_bert": config["model"]["freeze_bert"],
            "random_seed": config["seed"],
        },
        path,
    )


def build_optimizer(
    model: UnifiedSentimentModel, training: Dict[str, Any]
) -> torch.optim.Optimizer:
    base_learning_rate = float(training["learning_rate"])
    bert_learning_rate = float(
        training.get("bert_learning_rate", base_learning_rate)
    )
    bert_parameter_ids = {
        id(parameter)
        for parameter in model.text_encoder.parameters()
        if parameter.requires_grad
    }
    bert_parameters = [
        parameter
        for parameter in model.text_encoder.parameters()
        if parameter.requires_grad
    ]
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in bert_parameter_ids
    ]
    parameter_groups = []
    if other_parameters:
        parameter_groups.append(
            {
                "params": other_parameters,
                "lr": base_learning_rate,
                "group_name": "task_model",
            }
        )
    if bert_parameters:
        parameter_groups.append(
            {
                "params": bert_parameters,
                "lr": bert_learning_rate,
                "group_name": "bert",
            }
        )
    if not parameter_groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )


def plot_stage_log(log_path: Path, figure_path: Path) -> None:
    frame = pd.read_csv(log_path)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for split, group in frame.groupby("split"):
        axes[0].plot(group["epoch"], group["total_loss"], marker="o", label=split)
        axes[1].plot(group["epoch"], group["macro_f1"], marker="o", label=split)
        axes[2].plot(group["epoch"], group["mae"], marker="o", label=split)
    axes[0].set_title("Total loss")
    axes[1].set_title("Macro-F1")
    axes[2].set_title("MAE")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def train_stage(
    stage: int,
    model: UnifiedSentimentModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    variant: str,
    run_dir: Path,
    logger: Any,
) -> Path:
    training = config["training"]
    configure_stage_trainability(model, stage, training)
    optimizer = build_optimizer(model, training)
    bert_trainable = sum(
        parameter.numel()
        for parameter in model.text_encoder.parameters()
        if parameter.requires_grad
    )
    total_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        "stage=%d trainable_bert_parameters=%d trainable_other_parameters=%d "
        "bert_lr=%.2e other_lr=%.2e",
        stage,
        bert_trainable,
        total_trainable - bert_trainable,
        float(training.get("bert_learning_rate", training["learning_rate"])),
        float(training["learning_rate"]),
    )
    amp_enabled = bool(training["use_amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    mask_config = config["masking"]
    train_masks = build_mask_generator(mask_config, int(config["seed"]))
    valid_masks = build_mask_generator(
        mask_config, int(mask_config["valid_replay_seed"])
    )
    epochs = int(training[f"stage{stage}_epochs"])
    selection_name = str(training.get("selection_metric", "valid_total_loss"))
    selection_key_by_name = {
        "valid_total_loss": "total_loss",
        "valid_accuracy": "accuracy",
        "valid_macro_f1": "macro_f1",
    }
    if selection_name not in selection_key_by_name:
        raise ValueError(
            "training.selection_metric must be one of "
            f"{sorted(selection_key_by_name)}, got {selection_name!r}"
        )
    selection_key = selection_key_by_name[selection_name]
    checkpoint_keys = ("total_loss", "accuracy", "macro_f1")
    best_scores: Dict[str, tuple[float, ...] | None] = {
        key: None for key in checkpoint_keys
    }
    best_paths = {
        key: run_dir / "checkpoints" / f"best_stage{stage}_{variant}_by_{key}.pt"
        for key in checkpoint_keys
    }

    def ranking(key: str, metrics: Dict[str, Any]) -> tuple[float, ...]:
        if key == "total_loss":
            return (-float(metrics["total_loss"]),)
        if key == "accuracy":
            return (
                float(metrics["accuracy"]),
                float(metrics["macro_f1"]),
                -float(metrics["total_loss"]),
            )
        return (
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            -float(metrics["total_loss"]),
        )

    stale = 0
    last_path = run_dir / "checkpoints" / f"last_stage{stage}_{variant}.pt"
    log_path = run_dir / "logs" / f"stage{stage}_{variant}.csv"
    fieldnames = [
        "stage", "epoch", "split", "total_loss", "classification_loss",
        "regression_loss", "reconstruction_loss", "accuracy", "macro_f1",
        "balanced_accuracy", "precision_class_0", "recall_class_0",
        "f1_class_0", "precision_class_1", "recall_class_1", "f1_class_1",
        "precision_class_2", "recall_class_2", "f1_class_2",
        "mae", "pearson", "rmse",
    ]
    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device, config["loss"], stage, variant, epoch,
                mask_generator=train_masks if stage == 2 else None,
                optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
                accumulation_steps=int(training["gradient_accumulation_steps"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            valid_metrics = run_epoch(
                model, valid_loader, device, config["loss"], stage, variant, 0,
                mask_generator=valid_masks if stage == 2 else None,
                amp_enabled=amp_enabled,
            )
            for split, metrics in (("train", train_metrics), ("valid", valid_metrics)):
                writer.writerow({
                    "stage": stage, "epoch": epoch, "split": split,
                    **{key: metrics[key] for key in fieldnames[3:]},
                })
            handle.flush()
            logger.info(
                "stage=%d variant=%s epoch=%d train_loss=%.5f valid_loss=%.5f "
                "acc=%.4f macro_f1=%.4f recall_0=%.4f recall_1=%.4f "
                "recall_2=%.4f pred_counts=%s mae=%.4f pearson=%s",
                stage, variant, epoch, train_metrics["total_loss"],
                valid_metrics["total_loss"], valid_metrics["accuracy"],
                valid_metrics["macro_f1"], valid_metrics["recall_class_0"],
                valid_metrics["recall_class_1"], valid_metrics["recall_class_2"],
                valid_metrics["predicted_class_counts"], valid_metrics["mae"],
                "NA" if valid_metrics["pearson"] is None else f"{valid_metrics['pearson']:.4f}",
            )
            if bool(training.get("save_last_checkpoint", True)):
                save_checkpoint(
                    last_path, model, optimizer, config, stage, epoch, variant, valid_metrics
                )
            selection_improved = False
            for key in checkpoint_keys:
                score = ranking(key, valid_metrics)
                if best_scores[key] is None or score > best_scores[key]:
                    best_scores[key] = score
                    save_checkpoint(
                        best_paths[key], model, optimizer, config,
                        stage, epoch, variant, valid_metrics,
                    )
                    if key == selection_key:
                        selection_improved = True
            if selection_improved:
                stale = 0
            else:
                stale += 1
                if stale >= int(training["patience"]):
                    logger.info(
                        "Early stopping stage %d on %s", stage, selection_name
                    )
                    break
    best_path = best_paths[selection_key]
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return best_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    set_seed(int(config["seed"]), bool(config["training"]["deterministic"]))
    device = select_device(args.device)
    run_dir = make_run_dir(config["output"]["root"], args.run_name)
    logger = create_logger(run_dir / "logs" / "train.log", f"unified-{run_dir.name}")
    save_json(environment_manifest(), run_dir / "run_manifest.json")
    save_json(config, run_dir / "resolved_config.json")
    data_path = resolve_path(args.data_path or config["data"]["attachment2"])
    train_set = Attachment2Dataset(data_path, "train")
    valid_set = Attachment2Dataset(data_path, "valid")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    model = UnifiedSentimentModel(**resolve_model_config(config)).to(device)
    stage1 = train_stage(1, model, train_loader, valid_loader, config, device, args.variant, run_dir, logger)
    logger.info("Best stage I checkpoint: %s", stage1)
    plot_stage_log(
        run_dir / "logs" / f"stage1_{args.variant}.csv",
        run_dir / "figures" / f"stage1_{args.variant}_curves.png",
    )
    stage2 = train_stage(2, model, train_loader, valid_loader, config, device, args.variant, run_dir, logger)
    logger.info("Best stage II checkpoint: %s", stage2)
    plot_stage_log(
        run_dir / "logs" / f"stage2_{args.variant}.csv",
        run_dir / "figures" / f"stage2_{args.variant}_curves.png",
    )
    validation_rows = []
    for path in (stage1, stage2):
        saved = torch.load(path, map_location="cpu")
        validation_rows.append(
            {
                "stage": saved["stage"],
                "variant": saved["variant"],
                "epoch": saved["epoch"],
                **saved["validation_metrics"],
            }
        )
    pd.DataFrame(validation_rows).to_csv(
        run_dir / "results" / "validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(
        {"run_dir": str(run_dir), "stage1_checkpoint": str(stage1), "stage2_checkpoint": str(stage2)},
        run_dir / "training_summary.json",
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
