from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    classification_target: torch.Tensor,
    regression_target: torch.Tensor,
    clean_projected_target: Optional[torch.Tensor],
    lambda_cls: float = 1.0,
    lambda_reg: float = 1.0,
    lambda_rec: float = 0.1,
    huber_delta: float = 1.0,
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Frozen E7a loss: classification + regression + reconstruction."""
    regression = outputs["reg_pred"].squeeze(-1)
    if regression.shape != regression_target.shape:
        raise RuntimeError(
            f"Regression broadcasting forbidden: {regression.shape} vs {regression_target.shape}"
        )

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(
            class_weights,
            dtype=outputs["cls_logits"].dtype,
            device=outputs["cls_logits"].device,
        )
        expected = outputs["cls_logits"].shape[-1]
        if weight_tensor.shape != (expected,):
            raise ValueError(
                "class_weights must contain one value per class: "
                f"expected {expected}, got {weight_tensor.numel()}"
            )

    cls_loss = nn.functional.cross_entropy(
        outputs["cls_logits"],
        classification_target,
        weight=weight_tensor,
        label_smoothing=float(label_smoothing),
    )
    reg_loss = nn.functional.huber_loss(
        regression, regression_target, delta=huber_delta
    )

    rec_mask = outputs["artificial_mask"] & outputs["observed_mask"]
    if clean_projected_target is not None and torch.any(rec_mask):
        difference = outputs["reconstructed"] - clean_projected_target.detach()
        rec_loss = difference.pow(2)[rec_mask].mean()
    else:
        rec_loss = outputs["cls_logits"].sum() * 0.0

    total = lambda_cls * cls_loss + lambda_reg * reg_loss + lambda_rec * rec_loss
    return {
        "total_loss": total,
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
        "reconstruction_loss": rec_loss,
        "reconstruction_positions": rec_mask.sum(),
    }
