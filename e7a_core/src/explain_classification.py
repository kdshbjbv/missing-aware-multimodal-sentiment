from __future__ import annotations

import itertools
import math
from typing import Any, Dict

import torch


MODALITIES = ("T", "A", "V")


def class_logit_prediction(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
    target_class: int,
    class_bias: float = 0.0,
) -> torch.Tensor:
    output = model.forward_from_projected(
        projected,
        valid_mask,
        observed_mask,
        enable_imputation=False,
        valid_mask_by_modality=applicable_mask,
        return_explain_features=True,
    )
    return output["cls_logits"][:, int(target_class)] + float(class_bias)


def exact_modality_shapley_class_logit(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
    target_class: int,
    class_bias: float = 0.0,
) -> Dict[str, Any]:
    """Exact three-player Shapley values for one predicted-class raw logit."""
    if projected.shape[0] != 1:
        raise ValueError("Exact Shapley helper expects batch size 1")
    values: Dict[str, float] = {}
    indices = range(3)
    with torch.no_grad():
        for size in range(4):
            for subset in itertools.combinations(indices, size):
                key = "".join(MODALITIES[i] for i in subset) or "empty"
                keep = torch.zeros_like(applicable_mask)
                if subset:
                    keep[..., list(subset)] = applicable_mask[..., list(subset)]
                values[key] = float(
                    class_logit_prediction(
                        model, projected, valid_mask, keep, applicable_mask,
                        target_class, class_bias,
                    ).item()
                )
    shapley: Dict[str, float] = {}
    for modality in indices:
        contribution = 0.0
        others = [index for index in indices if index != modality]
        for size in range(3):
            for subset in itertools.combinations(others, size):
                with_modality = tuple(sorted((*subset, modality)))
                key_without = "".join(MODALITIES[i] for i in subset) or "empty"
                key_with = "".join(MODALITIES[i] for i in with_modality)
                weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
                contribution += weight * (values[key_with] - values[key_without])
        shapley[MODALITIES[modality]] = contribution
    error = sum(shapley.values()) - (values["TAV"] - values["empty"])
    return {
        "target_class": int(target_class),
        "class_bias": float(class_bias),
        "coalition_values": values,
        "shapley": shapley,
        "additivity_error": error,
    }


def integrated_gradients_projected_class_logit(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
    target_class: int,
    class_bias: float = 0.0,
    steps: int = 32,
) -> Dict[str, torch.Tensor]:
    """Projected-feature IG for the final predicted-class raw logit."""
    if steps < 2:
        raise ValueError("IG requires at least two integration steps")
    baseline = torch.zeros_like(projected)
    total_gradient = torch.zeros_like(projected)
    for alpha in torch.linspace(0.0, 1.0, steps, device=projected.device):
        interpolated = (baseline + alpha * (projected - baseline)).detach().requires_grad_(True)
        value = class_logit_prediction(
            model, interpolated, valid_mask, applicable_mask, applicable_mask,
            target_class, class_bias,
        ).sum()
        gradient = torch.autograd.grad(value, interpolated)[0]
        total_gradient += gradient
    ig = (projected - baseline) * total_gradient / steps
    with torch.no_grad():
        full = class_logit_prediction(
            model, projected, valid_mask, applicable_mask, applicable_mask,
            target_class, class_bias,
        )
        base = class_logit_prediction(
            model, baseline, valid_mask, applicable_mask, applicable_mask,
            target_class, class_bias,
        )
    signed_sum = ig.sum(dim=(1, 2, 3))
    completeness_error = signed_sum - (full - base)
    return {
        "ig": ig,
        "absolute_by_position": ig.abs().sum(dim=-1),
        "signed_by_position": ig.sum(dim=-1),
        "full_value": full,
        "baseline_value": base,
        "completeness_error": completeness_error,
    }
