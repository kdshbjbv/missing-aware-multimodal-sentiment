from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .calibration import logits_to_probabilities, mean_ensemble_logits
from .model import UnifiedSentimentModel
from .utils import resolve_model_config


def validate_members(
    checkpoint_paths: Sequence[Path], member_names: Sequence[str] | None
) -> list[str]:
    if len(checkpoint_paths) != 3:
        raise ValueError("Frozen E7a inference requires exactly three checkpoints")
    names = (
        [str(value) for value in member_names]
        if member_names is not None
        else [path.parent.parent.name for path in checkpoint_paths]
    )
    if len(names) != len(checkpoint_paths):
        raise ValueError("member_names must match checkpoints length")
    if len(set(names)) != len(names):
        raise ValueError("member_names must be unique")
    return names


def load_inference_model(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[UnifiedSentimentModel, dict[str, Any]]:
    # Keep checkpoint tensors on CPU while copying one member at a time to
    # the target device; this avoids a second full state dict on the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if str(checkpoint.get("variant", "E3")) != "E3":
        raise ValueError(f"{checkpoint_path} is not a formal E3 checkpoint")
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
    metadata = {
        "checkpoint": str(checkpoint_path),
        "stage": int(checkpoint.get("stage", 2)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "variant": str(checkpoint.get("variant", "E3")),
        "random_seed": int(checkpoint.get("random_seed", -1)),
        "compact_frozen_text_encoder": compact,
    }
    del checkpoint
    return model, metadata


def collect_member_outputs(
    dataset: Dataset,
    config: dict[str, Any],
    checkpoint_paths: Sequence[Path],
    member_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    expected_ids: list[str] | None = None
    member_logits: list[np.ndarray] = []
    member_regression: list[np.ndarray] = []
    member_null_prior: list[np.ndarray] = []
    checkpoint_metadata: list[dict[str, Any]] = []

    for checkpoint_path, member_name in zip(checkpoint_paths, member_names):
        model, metadata = load_inference_model(
            config, checkpoint_path, device
        )
        ids: list[str] = []
        logits_parts: list[np.ndarray] = []
        regression_parts: list[np.ndarray] = []
        null_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                tensors = {
                    key: batch[key].to(device)
                    for key in (
                        "text_bert", "audio", "vision", "valid_mask",
                        "valid_mask_by_modality", "observed_mask",
                    )
                }
                outputs = model(
                    tensors["text_bert"],
                    tensors["audio"],
                    tensors["vision"],
                    tensors["valid_mask"],
                    tensors["observed_mask"],
                    artificial_mask=None,
                    enable_imputation=True,
                    valid_mask_by_modality=tensors["valid_mask_by_modality"],
                    enable_gate=True,
                )
                ids.extend(str(value) for value in batch["sample_id"])
                logits_parts.append(
                    outputs["cls_logits"].detach().cpu().numpy().astype(np.float64)
                )
                regression_parts.append(
                    outputs["reg_pred"].detach().cpu().numpy()[:, 0].astype(np.float64)
                )
                null_parts.append(
                    outputs["used_null_prior"].detach().cpu().numpy().astype(bool)
                )
        logits = np.concatenate(logits_parts, axis=0)
        regression = np.concatenate(regression_parts, axis=0)
        null_prior = np.concatenate(null_parts, axis=0)
        if not np.isfinite(logits).all() or not np.isfinite(regression).all():
            raise FloatingPointError(f"Non-finite output from {checkpoint_path}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise RuntimeError("Ensemble members produced different sample ordering")
        member_logits.append(logits)
        member_regression.append(regression)
        member_null_prior.append(null_prior)
        checkpoint_metadata.append(
            {
                "name": member_name,
                **metadata,
            }
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    member_seeds = sorted(item["random_seed"] for item in checkpoint_metadata)
    if member_seeds != [17, 42, 2026]:
        raise ValueError(
            "Frozen E7a checkpoints must have random seeds 42, 17 and 2026; "
            f"got {member_seeds}"
        )
    if expected_ids is None or len(expected_ids) != len(dataset):
        raise RuntimeError("Inference row count does not match dataset")
    return {
        "sample_ids": expected_ids,
        "member_logits": np.stack(member_logits, axis=0),
        "member_regression": np.stack(member_regression, axis=0),
        "member_null_prior": np.stack(member_null_prior, axis=0),
        "members": checkpoint_metadata,
    }


def aggregate_member_outputs(
    member_logits: np.ndarray,
    member_regression: np.ndarray,
) -> dict[str, np.ndarray]:
    """Frozen E7a aggregation: equal-weight raw logits and regression means."""
    logits = np.asarray(member_logits, dtype=np.float64)
    regression = np.asarray(member_regression, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[2] != 3:
        raise ValueError("member_logits must have shape [members,samples,3]")
    if regression.shape != logits.shape[:2]:
        raise ValueError("member_regression must have shape [members,samples]")
    mean_logits = mean_ensemble_logits([values for values in logits])
    adjusted_logits, probabilities, predictions = logits_to_probabilities(mean_logits)
    return {
        "mean_logits": mean_logits,
        "adjusted_logits": adjusted_logits,
        "raw_probabilities": probabilities,
        "raw_predictions": predictions,
        "probabilities": probabilities,
        "predictions": predictions,
        "mean_regression": regression.mean(axis=0),
    }
