from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .align_modalities import aligned_feature_path
from .utils import get_logger


def plot_alignment_example(config: dict[str, Any], row: pd.Series, output_path: Path | None = None) -> Path:
    sid = str(row["sample_id"])
    path = aligned_feature_path(config, sid)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        valid_length = int(data["valid_length"].item())
        words = data["words"][:valid_length].astype(str)
        timestamps = data["timestamps"][:valid_length]
        audio = data["audio"][:valid_length]
        vision = data["vision"][:valid_length]
        audio_mask = data["audio_mask"][:valid_length].astype(bool)
        vision_mask = data["vision_mask"][:valid_length].astype(bool)
    if valid_length == 0:
        raise ValueError(f"No valid positions to plot for {sid}")
    centers = timestamps.mean(axis=1)
    audio_strength = np.linalg.norm(audio, axis=1)
    vision_strength = np.linalg.norm(vision, axis=1)
    if audio_strength.max() > 0:
        audio_strength = audio_strength / audio_strength.max()
    if vision_strength.max() > 0:
        vision_strength = vision_strength / vision_strength.max()

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1]})
    for index, (word, (start, end)) in enumerate(zip(words, timestamps)):
        axes[0].broken_barh([(start, end - start)], (0.15, 0.7), facecolors="#4C78A8", alpha=0.8)
        axes[0].text((start + end) / 2, 0.5, word, rotation=45, ha="center", va="center", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Text words")
    # Matplotlib treats dollar signs as math delimiters; MOSEI sample IDs use
    # "$_$", so escape them before rendering the title.
    display_sid = sid.replace("$", r"\$")
    axes[0].set_title(f"Word-level multimodal alignment: {display_sid}")
    axes[1].plot(centers, audio_strength, color="#F58518", marker="o", linewidth=1.5, label="Audio feature norm")
    axes[1].scatter(centers[~audio_mask], np.zeros((~audio_mask).sum()), color="red", marker="x", label="Missing audio")
    axes[1].set_ylabel("Audio\n(normalized)")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend(loc="upper right")
    axes[2].plot(centers, vision_strength, color="#54A24B", marker="o", linewidth=1.5, label="Visual feature norm")
    axes[2].scatter(centers[~vision_mask], np.zeros((~vision_mask).sum()), color="red", marker="x", label="Missing face")
    axes[2].set_ylabel("Vision\n(normalized)")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_xlabel("Original video time (seconds)")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    destination = output_path or (Path(config["paths"]["output_dir"]) / config["output"]["figure_file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Saved alignment example for %s to %s", sid, destination)
    return destination
