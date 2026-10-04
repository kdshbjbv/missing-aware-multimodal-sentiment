import numpy as np

from src.align_modalities import pool_points_by_windows


def test_pool_points_by_windows_mean_and_mask():
    times = np.array([0.1, 0.2, 0.7, 1.2], dtype=np.float32)
    features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    windows = np.array([[0.0, 0.5], [0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    pooled, mask, counts = pool_points_by_windows(times, features, windows)
    np.testing.assert_allclose(pooled[0], [2, 3])
    np.testing.assert_allclose(pooled[1], [5, 6])
    np.testing.assert_allclose(pooled[2], [0, 0])
    np.testing.assert_array_equal(mask, [1, 1, 0])
    np.testing.assert_array_equal(counts, [2, 1, 0])


def test_invalid_frames_are_excluded():
    times = np.array([0.1, 0.2], dtype=np.float32)
    features = np.array([[1.0], [100.0]], dtype=np.float32)
    windows = np.array([[0.0, 0.5]], dtype=np.float32)
    pooled, mask, counts = pool_points_by_windows(times, features, windows, valid=np.array([1, 0]))
    np.testing.assert_allclose(pooled, [[1.0]])
    np.testing.assert_array_equal(mask, [1])
    np.testing.assert_array_equal(counts, [1])
