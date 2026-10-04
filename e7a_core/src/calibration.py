from __future__ import annotations

from typing import Sequence

import numpy as np


def logits_to_probabilities(
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert frozen, uncalibrated E7a logits to probabilities and labels."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predictions = values.argmax(axis=1).astype(np.int64)
    return values, probabilities, predictions


def mean_ensemble_logits(logit_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight arithmetic mean used by the frozen E7a ensemble."""
    if not logit_matrices:
        raise ValueError("At least one logit matrix is required")
    matrices = [np.asarray(values, dtype=np.float64) for values in logit_matrices]
    expected_shape = matrices[0].shape
    if len(expected_shape) != 2:
        raise ValueError("Each logit matrix must have shape [samples, classes]")
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("All ensemble logit matrices must have identical shapes")
    if not all(np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Ensemble logits must be finite")
    return np.mean(np.stack(matrices, axis=0), axis=0)
