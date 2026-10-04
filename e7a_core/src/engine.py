from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import multitask_loss
from .masks import ArtificialMaskGenerator
from .metrics import task_metrics


TENSOR_KEYS = (
    "text_bert",
    "audio",
    "vision",
    "valid_mask",
    "valid_mask_by_modality",
    "observed_mask",
    "classification_label",
    "regression_label",
)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result = dict(batch)
    for key in TENSOR_KEYS:
        if key in result:
            result[key] = result[key].to(device, non_blocking=True)
    return result


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: Dict[str, float],
    stage: int,
    variant: str,
    epoch: int,
    mask_generator: Optional[ArtificialMaskGenerator] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_enabled: bool = False,
    accumulation_steps: int = 1,
    gradient_clip_norm: Optional[float] = None,
    artificial_masking: Optional[bool] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    # Stage II may freeze the three projections to define a stable H* target.
    # Keep their dropout disabled even after model.train(True) recurses.
    if stage == 2:
        for module in (
            model.text_projection,
            model.audio_projection,
            model.vision_projection,
        ):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    enable_gate = variant in {"E1", "E3"}
    enable_imputation = stage == 2 and variant in {"E2", "E3"}
    use_artificial_masking = stage == 2 if artificial_masking is None else artificial_masking
    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "regression_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    count = 0
    true_cls: list[int] = []
    pred_cls: list[int] = []
    true_reg: list[float] = []
    pred_reg: list[float] = []
    sample_ids: list[str] = []
    logits_rows: list[list[float]] = []
    probability_rows: list[list[float]] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = nullcontext() if training else torch.no_grad()
    with context:
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            artificial = None
            clean_target = None
            if use_artificial_masking:
                if mask_generator is None:
                    raise ValueError("Artificial masking requires ArtificialMaskGenerator")
                artificial = mask_generator.generate(
                    batch["valid_mask_by_modality"],
                    batch["observed_mask"],
                    raw_batch["sample_id"],
                    epoch,
                )
                if variant in {"E2", "E3"}:
                    with torch.no_grad():
                        clean_target = model.encode_projected(
                            batch["text_bert"], batch["audio"], batch["vision"]
                        )
            autocast = (
                torch.cuda.amp.autocast(enabled=True) if amp_enabled else nullcontext()
            )
            with autocast:
                outputs = model(
                    batch["text_bert"],
                    batch["audio"],
                    batch["vision"],
                    batch["valid_mask"],
                    batch["observed_mask"],
                    artificial_mask=artificial,
                    enable_imputation=enable_imputation,
                    valid_mask_by_modality=batch["valid_mask_by_modality"],
                    enable_gate=enable_gate,
                )
                losses = multitask_loss(
                    outputs,
                    batch["classification_label"],
                    batch["regression_label"],
                    clean_target,
                    lambda_cls=float(loss_config["lambda_cls"]),
                    lambda_reg=float(loss_config["lambda_reg"]),
                    lambda_rec=(
                        float(loss_config["lambda_rec"])
                        if stage == 2 and variant in {"E2", "E3"}
                        else 0.0
                    ),
                    huber_delta=float(loss_config["huber_delta"]),
                    class_weights=loss_config.get("class_weights"),
                    label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
                )
                scaled_loss = losses["total_loss"] / accumulation_steps
            if not torch.isfinite(losses["total_loss"]):
                raise FloatingPointError("Non-finite training/evaluation loss")
            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if gradient_clip_norm is not None:
                        if scaler is not None and amp_enabled:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            batch_size = len(batch["classification_label"])
            count += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_size
            true_cls.extend(batch["classification_label"].detach().cpu().tolist())
            pred_cls.extend(outputs["cls_logits"].argmax(-1).detach().cpu().tolist())
            true_reg.extend(batch["regression_label"].detach().cpu().tolist())
            pred_reg.extend(outputs["reg_pred"].squeeze(-1).detach().float().cpu().tolist())
            sample_ids.extend(map(str, raw_batch["sample_id"]))
            detached_logits = outputs["cls_logits"].detach().float().cpu()
            logits_rows.extend(detached_logits.tolist())
            probability_rows.extend(torch.softmax(detached_logits, dim=-1).tolist())
    result: Dict[str, Any] = {key: value / count for key, value in totals.items()}
    result.update(task_metrics(true_cls, pred_cls, true_reg, pred_reg))
    result["predictions"] = {
        "sample_id": sample_ids,
        "true_class": true_cls,
        "pred_class": pred_cls,
        "true_regression": true_reg,
        "pred_regression": pred_reg,
    }
    if logits_rows:
        for class_index in range(len(logits_rows[0])):
            result["predictions"][f"logit_class_{class_index}"] = [
                row[class_index] for row in logits_rows
            ]
            result["predictions"][f"prob_class_{class_index}"] = [
                row[class_index] for row in probability_rows
            ]
    return result
