from __future__ import annotations

import numpy as np
import torch

from e7a_core.src.ensemble_inference import aggregate_member_outputs
from e7a_core.src.masks import ArtificialMaskGenerator, mask_intervals


def test_three_member_mean_uses_unbiased_raw_logits():
    logits = np.array([
        [[0.0, 1.0, 3.0]],
        [[0.0, 1.0, 3.0]],
        [[4.0, 1.0, 0.0]],
    ])
    regression = np.array([[1.0], [2.0], [3.0]])
    result = aggregate_member_outputs(logits, regression)
    np.testing.assert_allclose(result["mean_logits"], [[4 / 3, 1, 2]])
    assert int(result["predictions"][0]) == 2
    np.testing.assert_allclose(result["mean_regression"], [2.0])


def test_continuous_missing_blocks_respect_observations():
    applicable = torch.ones((1, 20, 3), dtype=torch.bool)
    observed = applicable.clone()
    observed[:, 5:8, 1] = False
    generator = ArtificialMaskGenerator(
        complete_probability=0,
        missing_ratios=(0.5,),
        combinations=("A",),
        seed=42,
        max_block_length=5,
    )
    mask = generator.generate(applicable, observed, ["synthetic"], 1)
    assert not torch.any(mask & ~observed)
    assert not mask[0, :, 0].any() and not mask[0, :, 2].any()
    assert mask[0, :, 1].sum() > 0
    assert all(end - start + 1 <= 5 for start, end in mask_intervals(mask[0, :, 1].numpy()))
