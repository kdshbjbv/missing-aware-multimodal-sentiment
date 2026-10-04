from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


def safe_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a = np.asarray(y_true, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def classification_metrics(
    true_cls: Sequence[int], pred_cls: Sequence[int]
) -> Dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_cls,
        pred_cls,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(true_cls, pred_cls, labels=labels)
    true_counts = np.bincount(np.asarray(true_cls, dtype=np.int64), minlength=3)
    predicted_counts = np.bincount(np.asarray(pred_cls, dtype=np.int64), minlength=3)
    result = {
        "accuracy": float(accuracy_score(true_cls, pred_cls)),
        "macro_f1": float(
            f1_score(true_cls, pred_cls, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "true_class_counts": true_counts.astype(int).tolist(),
        "predicted_class_counts": predicted_counts.astype(int).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    for index, label in enumerate(labels):
        result[f"precision_class_{label}"] = float(precision[index])
        result[f"recall_class_{label}"] = float(recall[index])
        result[f"f1_class_{label}"] = float(per_class_f1[index])
    return result


def task_metrics(
    true_cls: Sequence[int],
    pred_cls: Sequence[int],
    true_reg: Sequence[float],
    pred_reg: Sequence[float],
) -> Dict[str, Any]:
    result = classification_metrics(true_cls, pred_cls)
    result.update(
        {
            "mae": float(mean_absolute_error(true_reg, pred_reg)),
            "pearson": safe_pearson(true_reg, pred_reg),
            "rmse": float(np.sqrt(mean_squared_error(true_reg, pred_reg))),
        }
    )
    return result
