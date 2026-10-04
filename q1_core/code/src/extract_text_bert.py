from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .align_text_mfa import load_word_timestamps
from .utils import check_finite, get_logger, safe_sample_name


class BertWordExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        # Keep downloaded model artifacts inside the writable project tree.
        os.environ["HF_HOME"] = str(config["paths"]["huggingface_home"])
        endpoint = str(config["text"].get("hf_endpoint", "")).strip()
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        # The mirror serves ordinary HTTP files; disabling Xet avoids a second,
        # often firewalled download endpoint on competition servers.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("PyTorch and transformers are required for BERT extraction") from exc
        self.torch = torch
        requested = config["text"].get("device", "auto")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        model_name = config["text"]["model_name"]
        local_only = bool(config["text"].get("local_files_only", False))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self.model.eval().to(self.device)
        self.model_name = model_name
        self.hidden_size = int(self.model.config.hidden_size)
        self.max_model_tokens = int(config["text"].get("max_model_tokens", 512))
        self.batch_size = max(1, int(config["text"].get("batch_size", 16)))
        if self.device == "cuda":
            index = int(config["text"].get("device_index", 0))
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid text.device_index={index}; device_count={torch.cuda.device_count()}"
                )
            torch.cuda.set_device(index)
            properties = torch.cuda.get_device_properties(index)
            get_logger().info(
                "BERT runtime: CUDA device=%d name=%s memory_mib=%d batch_size=%d",
                index,
                properties.name,
                round(properties.total_memory / 1024**2),
                self.batch_size,
            )
        else:
            get_logger().info("BERT runtime: device=%s batch_size=%d", self.device, self.batch_size)

    def encode_words(self, words: Sequence[str]) -> np.ndarray:
        return self.encode_word_batches([words])[0]

    def encode_word_batches(self, word_batches: Sequence[Sequence[str]]) -> list[np.ndarray]:
        """Encode multiple word sequences in one padded forward pass.

        Version A contains short utterances. Batching them substantially improves
        GPU occupancy without changing the per-word mean-subword representation.
        """
        if not word_batches:
            return []
        if any(not words for words in word_batches):
            raise ValueError("BERT batch contains an empty word sequence")
        encoded = self.tokenizer(
            [list(words) for words in word_batches],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_tokens,
            add_special_tokens=True,
        )
        word_id_batches = [encoded.word_ids(batch_index=index) for index in range(len(word_batches))]
        tensors = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
        with self.torch.inference_mode():
            hidden = self.model(**tensors).last_hidden_state.detach().float().cpu().numpy()
        results: list[np.ndarray] = []
        for batch_index, words in enumerate(word_batches):
            word_ids = word_id_batches[batch_index]
            pooled: list[np.ndarray] = []
            for word_index in range(len(words)):
                token_indices = [index for index, value in enumerate(word_ids) if value == word_index]
                if not token_indices:
                    raise ValueError(
                        f"Tokenizer produced no subword for word index {word_index}: "
                        f"{words[word_index]}"
                    )
                pooled.append(hidden[batch_index, token_indices].mean(axis=0))
            result = np.asarray(pooled, dtype=np.float32)
            check_finite("BERT word features", result)
            results.append(result)
        return results


def text_feature_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "features" / "text" / f"{safe_sample_name(sample_id)}.npz"


def extract_one_text(
    config: dict[str, Any],
    row: pd.Series,
    extractor: BertWordExtractor,
    force: bool = False,
) -> Path:
    logger = get_logger()
    destination = text_feature_path(config, str(row["sample_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.debug("Text cache hit: %s", destination)
        return destination
    word_items = load_word_timestamps(config, str(row["sample_id"]))
    words = [str(item["word"]) for item in word_items]
    timestamps = np.asarray([[item["start"], item["end"]] for item in word_items], dtype=np.float32)
    features = extractor.encode_words(words)
    if len(features) != len(timestamps):
        raise ValueError(
            f"BERT/MFA length mismatch for {row['sample_id']}: {len(features)} vs {len(timestamps)}"
        )
    np.savez_compressed(
        destination,
        words=np.asarray(words, dtype=np.str_),
        timestamps=timestamps,
        features=features,
        feature_names=np.asarray([f"bert_{index:03d}" for index in range(features.shape[1])], dtype=np.str_),
        extractor=np.asarray(extractor.model_name, dtype=np.str_),
    )
    logger.info("Extracted BERT word features: %s (%d words)", row["sample_id"], len(words))
    return destination


def extract_many_texts(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    extractor: BertWordExtractor,
    force: bool = False,
) -> list[Path]:
    """Extract BERT features with cache-aware multi-utterance GPU batches."""
    logger = get_logger()
    completed: list[Path] = []
    pending: list[tuple[str, Path, list[str], np.ndarray]] = []
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        destination = text_feature_path(config, sample_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            completed.append(destination)
            continue
        word_items = load_word_timestamps(config, sample_id)
        words = [str(item["word"]) for item in word_items]
        timestamps = np.asarray(
            [[item["start"], item["end"]] for item in word_items], dtype=np.float32
        )
        pending.append((sample_id, destination, words, timestamps))

    logger.info(
        "BERT extraction plan: cached=%d pending=%d batch_size=%d device=%s",
        len(completed),
        len(pending),
        extractor.batch_size,
        extractor.device,
    )
    for start in range(0, len(pending), extractor.batch_size):
        batch = pending[start : start + extractor.batch_size]
        feature_batches = extractor.encode_word_batches([item[2] for item in batch])
        for (sample_id, destination, words, timestamps), features in zip(batch, feature_batches):
            if len(features) != len(timestamps):
                raise ValueError(
                    f"BERT/MFA length mismatch for {sample_id}: "
                    f"{len(features)} vs {len(timestamps)}"
                )
            np.savez_compressed(
                destination,
                words=np.asarray(words, dtype=np.str_),
                timestamps=timestamps,
                features=features,
                feature_names=np.asarray(
                    [f"bert_{index:03d}" for index in range(features.shape[1])],
                    dtype=np.str_,
                ),
                extractor=np.asarray(extractor.model_name, dtype=np.str_),
            )
            completed.append(destination)
        logger.info(
            "BERT GPU batch [%d/%d]: %d utterances",
            min(start + len(batch), len(pending)),
            len(pending),
            len(batch),
        )
    return completed
