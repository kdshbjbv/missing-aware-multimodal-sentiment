"""Convert a Q1 word aligned sample to the E7a token aligned input.

Q1's 768 dimensional word embeddings are intentionally not fed to E7a.
E7a runs its own frozen BERT over token IDs.  The tokenizer supplied here
must be the same bert-base-uncased tokenizer used by that encoder.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from e7a_core.src.data import _normalise_split


def adapt_q1_sample(sample: Mapping[str, Any], tokenizer: Any, max_length: int = 50) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("Q1 sample_id is required")
    if max_length != 50:
        raise ValueError("The supplied E7a checkpoints require 50 positions")
    words = np.asarray(sample["words"])
    sequence_mask = np.asarray(sample["sequence_mask"], dtype=bool)
    audio = np.asarray(sample["audio"], dtype=np.float32)
    vision = np.asarray(sample["vision"], dtype=np.float32)
    audio_mask = np.asarray(sample["audio_mask"], dtype=bool)
    vision_mask = np.asarray(sample["vision_mask"], dtype=bool)
    text_mask = np.asarray(sample.get("text_mask", sequence_mask), dtype=bool)
    if words.shape != (50,) or sequence_mask.shape != (50,):
        raise ValueError("Q1 words and sequence_mask must have 50 positions")
    if audio.shape != (50, 74) or vision.shape != (50, 35):
        raise ValueError(f"Q1 Audio/Vision shape mismatch: {audio.shape}, {vision.shape}")
    if audio_mask.shape != (50,) or vision_mask.shape != (50,) or text_mask.shape != (50,):
        raise ValueError("Q1 Text/Audio/Vision masks must have 50 positions")
    if not np.array_equal(sequence_mask, np.arange(50) < sequence_mask.sum()):
        raise ValueError("Q1 valid positions must form a contiguous prefix")
    count = int(sequence_mask.sum())
    if count == 0:
        raise ValueError(f"Q1 sample {sample_id} has no valid words")
    valid_words = [str(word) if text_mask[index] else "[UNK]" for index, word in enumerate(words[:count])]
    encoded = tokenizer(
        valid_words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention = np.asarray(encoded["attention_mask"], dtype=np.int64)
    token_types = np.asarray(encoded["token_type_ids"], dtype=np.int64)
    word_ids = encoded.word_ids()
    if ids.shape != (50,) or len(word_ids) != 50:
        raise ValueError("Tokenizer did not return 50 token positions with word IDs")
    aligned_audio = np.zeros((50, 74), dtype=np.float32)
    aligned_vision = np.zeros((50, 35), dtype=np.float32)
    observed = np.zeros((50, 3), dtype=bool)
    for pos, word_index in enumerate(word_ids):
        if word_index is None or not attention[pos]:
            continue
        if not 0 <= word_index < count:
            raise ValueError("Tokenizer returned a word ID outside the Q1 sequence")
        if audio_mask[word_index]:
            aligned_audio[pos] = audio[word_index]
            observed[pos, 1] = True
        if vision_mask[word_index]:
            aligned_vision[pos] = vision[word_index]
            observed[pos, 2] = True
        observed[pos, 0] = text_mask[word_index]
    text_bert = np.stack((ids, attention, token_types))
    normalized = _normalise_split({
        "text_bert": text_bert,
        "audio": aligned_audio,
        "vision": aligned_vision,
    })
    # Explicit source masks are authoritative: a valid observed vector may be zero.
    normalized["observed_mask"] = observed[None]
    result = {key: value[0] for key, value in normalized.items()}
    result.update({
        "sample_id": sample_id,
        "raw_text": str(sample.get("raw_text", " ".join(valid_words))),
        "token_to_q1_word": [None if index is None else int(index) for index in word_ids],
        "source_word_count": count,
        "tokenized_word_count": len({i for i in word_ids if i is not None}),
        "truncated": len({i for i in word_ids if i is not None}) < count,
    })
    if sample.get("classification_label") is not None:
        result["classification_label"] = int(sample["classification_label"])
    regression = sample.get("regression_label", sample.get("label"))
    if regression is not None:
        result["regression_label"] = float(regression)
    return result
