from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


MODALITY_INDEX = {"T": 0, "A": 1, "V": 2}


def _stable_seed(base_seed: int, sample_id: str, epoch: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{epoch}|{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _choose_contiguous(
    candidates: np.ndarray,
    target: int,
    rng: np.random.Generator,
    max_run_length: int = 15,
) -> np.ndarray:
    chosen = np.zeros_like(candidates, dtype=bool)
    indices = np.flatnonzero(candidates)
    if len(indices) == 0 or target <= 0:
        return chosen
    target = min(target, len(indices))
    remaining = set(indices.tolist())
    while remaining and chosen.sum() < target:
        starts = np.array(sorted(remaining))
        start = int(rng.choice(starts))
        max_run = min(target - int(chosen.sum()), max_run_length)
        run = int(rng.integers(1, max_run + 1))
        for position in range(start, start + run):
            if position in remaining:
                chosen[position] = True
                remaining.remove(position)
                if chosen.sum() >= target:
                    break
        if start in remaining:
            remaining.remove(start)
    return chosen


@dataclass
class ArtificialMaskGenerator:
    complete_probability: float = 0.15
    missing_ratios: Sequence[float] = (0.1, 0.2, 0.4, 0.6)
    combinations: Sequence[str] = ("T", "A", "V", "TA", "TV", "AV", "TAV")
    seed: int = 42
    synchronize_modalities: bool = False
    missing_ratio_probabilities: Sequence[float] | None = None
    max_block_length: int = 15

    def __post_init__(self) -> None:
        if not 0.0 <= self.complete_probability <= 1.0:
            raise ValueError("complete_probability must be between 0 and 1")
        if not self.missing_ratios:
            raise ValueError("missing_ratios must not be empty")
        if any(float(ratio) <= 0.0 or float(ratio) > 1.0 for ratio in self.missing_ratios):
            raise ValueError("missing_ratios must be in (0, 1]")
        if self.max_block_length <= 0:
            raise ValueError("max_block_length must be positive")
        if self.missing_ratio_probabilities is not None:
            probabilities = np.asarray(self.missing_ratio_probabilities, dtype=float)
            if len(probabilities) != len(self.missing_ratios):
                raise ValueError(
                    "missing_ratio_probabilities must match missing_ratios length"
                )
            if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
                raise ValueError(
                    "missing_ratio_probabilities must be non-negative with positive sum"
                )

    def generate(
        self,
        applicable_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_ids: Iterable[str],
        epoch: int,
    ) -> torch.Tensor:
        eligible = (applicable_mask & observed_mask).detach().cpu().numpy()
        output = np.zeros_like(eligible, dtype=bool)
        for row, sample_id in enumerate(sample_ids):
            rng = np.random.default_rng(_stable_seed(self.seed, str(sample_id), epoch))
            if rng.random() < self.complete_probability:
                continue
            combination = str(rng.choice(self.combinations))
            if self.missing_ratio_probabilities is None:
                ratio = float(rng.choice(self.missing_ratios))
            else:
                probabilities = np.asarray(
                    self.missing_ratio_probabilities, dtype=float
                )
                probabilities = probabilities / probabilities.sum()
                ratio = float(rng.choice(self.missing_ratios, p=probabilities))
            if self.synchronize_modalities and len(combination) > 1:
                modality_indices = [MODALITY_INDEX[code] for code in combination]
                common_candidates = np.logical_and.reduce(
                    [eligible[row, :, modality] for modality in modality_indices]
                )
                common_count = int(common_candidates.sum())
                target = max(1, int(round(common_count * ratio))) if common_count else 0
                shared = _choose_contiguous(
                    common_candidates, target, rng, self.max_block_length
                )
                for modality in modality_indices:
                    output[row, :, modality] = shared
                continue
            for code in combination:
                modality = MODALITY_INDEX[code]
                count = int(eligible[row, :, modality].sum())
                target = max(1, int(round(count * ratio))) if count else 0
                output[row, :, modality] = _choose_contiguous(
                    eligible[row, :, modality], target, rng, self.max_block_length
                )
        result = torch.as_tensor(output, device=applicable_mask.device, dtype=torch.bool)
        if torch.any(result & ~(applicable_mask & observed_mask)):
            raise RuntimeError("Artificial mask violated A <= P_mod * O")
        return result


@dataclass
class HybridArtificialMaskGenerator:
    """Choose one masking profile per non-complete sample deterministically."""

    profiles: Sequence[ArtificialMaskGenerator]
    profile_probabilities: Sequence[float]
    complete_probability: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.complete_probability <= 1.0:
            raise ValueError("complete_probability must be between 0 and 1")
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        probabilities = np.asarray(self.profile_probabilities, dtype=float)
        if len(probabilities) != len(self.profiles):
            raise ValueError("profile_probabilities must match profiles length")
        if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
            raise ValueError(
                "profile_probabilities must be non-negative with positive sum"
            )

    def generate(
        self,
        applicable_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_ids: Iterable[str],
        epoch: int,
    ) -> torch.Tensor:
        ids = [str(sample_id) for sample_id in sample_ids]
        if len(ids) != int(applicable_mask.shape[0]):
            raise ValueError("sample_ids length must match batch size")
        probabilities = np.asarray(self.profile_probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum()
        assignments: list[list[int]] = [[] for _ in self.profiles]
        for row, sample_id in enumerate(ids):
            rng = np.random.default_rng(_stable_seed(self.seed, sample_id, epoch))
            if rng.random() < self.complete_probability:
                continue
            profile_index = int(rng.choice(len(self.profiles), p=probabilities))
            assignments[profile_index].append(row)

        output = torch.zeros_like(applicable_mask, dtype=torch.bool)
        for profile, rows in zip(self.profiles, assignments):
            if not rows:
                continue
            row_index = torch.as_tensor(
                rows, device=applicable_mask.device, dtype=torch.long
            )
            profile_mask = profile.generate(
                applicable_mask.index_select(0, row_index),
                observed_mask.index_select(0, row_index),
                [ids[row] for row in rows],
                epoch,
            )
            output.index_copy_(0, row_index, profile_mask)
        if torch.any(output & ~(applicable_mask & observed_mask)):
            raise RuntimeError("Hybrid artificial mask violated A <= P_mod * O")
        return output


MaskGenerator = ArtificialMaskGenerator | HybridArtificialMaskGenerator


def mask_intervals(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool).tolist() + [False]
    result: list[list[int]] = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append([start, index - 1])
            start = None
    return result
