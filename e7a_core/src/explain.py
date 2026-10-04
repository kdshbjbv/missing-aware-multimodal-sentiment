from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch


MODALITIES = ("T", "A", "V")


def _prediction(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
) -> torch.Tensor:
    return model.forward_from_projected(
        projected,
        valid_mask,
        observed_mask,
        enable_imputation=False,
        valid_mask_by_modality=applicable_mask,
        return_explain_features=True,
    )["reg_pred"].squeeze(-1)


def exact_modality_shapley(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
) -> Dict[str, Any]:
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
                    _prediction(model, projected, valid_mask, keep, applicable_mask).item()
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
    return {"coalition_values": values, "shapley": shapley, "additivity_error": error}


def integrated_gradients_projected(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
    steps: int = 32,
) -> Dict[str, torch.Tensor]:
    if steps < 2:
        raise ValueError("IG requires at least two integration steps")
    baseline = torch.zeros_like(projected)
    total_gradient = torch.zeros_like(projected)
    for alpha in torch.linspace(0.0, 1.0, steps, device=projected.device):
        interpolated = (baseline + alpha * (projected - baseline)).detach().requires_grad_(True)
        value = _prediction(
            model, interpolated, valid_mask, applicable_mask, applicable_mask
        ).sum()
        gradient = torch.autograd.grad(value, interpolated)[0]
        total_gradient += gradient
    ig = (projected - baseline) * total_gradient / steps
    with torch.no_grad():
        full = _prediction(model, projected, valid_mask, applicable_mask, applicable_mask)
        base = _prediction(model, baseline, valid_mask, applicable_mask, applicable_mask)
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


def faithfulness_occlusion(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable_mask: torch.Tensor,
    importance: torch.Tensor,
    topk_fraction: float,
    random_repeats: int,
    seed: int,
) -> list[Dict[str, float | int | str]]:
    if projected.shape[0] != 1:
        raise ValueError("Faithfulness helper expects batch size 1")
    legal = applicable_mask[0]
    candidates = torch.nonzero(legal, as_tuple=False)
    k = max(1, int(round(len(candidates) * topk_fraction)))
    candidate_scores = importance[0][legal]
    top_indices = candidates[torch.topk(candidate_scores, k=min(k, len(candidates))).indices]
    with torch.no_grad():
        full = float(_prediction(model, projected, valid_mask, applicable_mask, applicable_mask).item())

    def removed_value(indices: torch.Tensor) -> float:
        observed = applicable_mask.clone()
        modified = projected.clone()
        observed[0, indices[:, 0], indices[:, 1]] = False
        modified[0, indices[:, 0], indices[:, 1]] = 0.0
        with torch.no_grad():
            return float(_prediction(model, modified, valid_mask, observed, applicable_mask).item())

    rows: list[Dict[str, float | int | str]] = []
    top_value = removed_value(top_indices)
    rows.append(
        {
            "selection": "top",
            "repeat": 0,
            "k": int(len(top_indices)),
            "full_value": full,
            "removed_value": top_value,
            "absolute_change": abs(full - top_value),
            "signed_change": top_value - full,
            "seed": seed,
        }
    )
    rng = np.random.default_rng(seed)
    for repeat in range(random_repeats):
        picked = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        random_value = removed_value(candidates[picked])
        rows.append(
            {
                "selection": "random",
                "repeat": repeat,
                "k": int(min(k, len(candidates))),
                "full_value": full,
                "removed_value": random_value,
                "absolute_change": abs(full - random_value),
                "signed_change": random_value - full,
                "seed": seed,
            }
        )
    return rows
