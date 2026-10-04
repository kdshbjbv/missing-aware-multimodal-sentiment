from __future__ import annotations

import numpy as np

from src.problem1_dataset import pool_points_by_word_windows


def test_pooling_uses_half_open_word_windows() -> None:
    times = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    windows = np.asarray([[0.0, 0.2], [0.2, 0.3]], dtype=np.float64)
    pooled, mask, counts = pool_points_by_word_windows(times, features, windows)
    np.testing.assert_allclose(pooled[:, 0], [0.5, 2.0])
    np.testing.assert_array_equal(mask, [1, 1])
    np.testing.assert_array_equal(counts, [2, 1])


def test_pooling_preserves_missing_modality_window() -> None:
    times = np.asarray([0.05, 0.15], dtype=np.float64)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.asarray([0, 1], dtype=np.uint8)
    windows = np.asarray([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    pooled, mask, counts = pool_points_by_word_windows(times, features, windows, valid)
    np.testing.assert_array_equal(mask, [0, 1, 0])
    np.testing.assert_array_equal(counts, [0, 1, 0])
    np.testing.assert_array_equal(pooled[0], [0.0, 0.0])
    np.testing.assert_array_equal(pooled[2], [0.0, 0.0])
