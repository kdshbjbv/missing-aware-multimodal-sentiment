from __future__ import annotations

import numpy as np
import pytest
import torch

from e7a_core.src.model import UnifiedSentimentModel
from src.data.q1_to_e7a_adapter import adapt_q1_sample


class Encoded(dict):
    def __init__(self, *args, word_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class TinyTokenizer:
    def __call__(self, words, **kwargs):
        assert kwargs["is_split_into_words"]
        assert kwargs["max_length"] == 50
        # The first source word becomes two WordPieces.
        mapped = [None, 0, 0] + list(range(1, len(words))) + [None]
        mapped = mapped[:50]
        if len(mapped) == 50:
            mapped[-1] = None
        n = len(mapped)
        ids = [101] + [1000 + i for i in range(n - 2)] + [102]
        return Encoded(
            input_ids=ids + [0] * (50 - n),
            attention_mask=[1] * n + [0] * (50 - n),
            token_type_ids=[0] * 50,
            word_ids=mapped + [None] * (50 - n),
        )


class DummyEncoder(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        hidden = torch.zeros((*input_ids.shape, 768), dtype=torch.float32)
        return type("EncoderOutput", (), {"last_hidden_state": hidden})()


def sample():
    words = np.array(["hello", "world"] + ["<PAD>"] * 48)
    audio = np.zeros((50, 74), np.float32)
    vision = np.zeros((50, 35), np.float32)
    audio[0] = 1
    vision[1] = 1
    return {
        "sample_id": "synthetic-1",
        "words": words,
        "sequence_mask": np.array([1, 1] + [0] * 48),
        "audio": audio,
        "vision": vision,
        "audio_mask": np.array([1, 0] + [0] * 48),
        "vision_mask": np.array([0, 1] + [0] * 48),
    }


def test_q1_adapter_feeds_e7a():
    converted = adapt_q1_sample(sample(), TinyTokenizer())
    assert converted["text_bert"].shape == (3, 50)
    assert converted["audio"].shape == (50, 74)
    assert converted["vision"].shape == (50, 35)
    assert converted["observed_mask"][1:3, 1].all()
    assert converted["observed_mask"][3, 2]
    assert not converted["observed_mask"][0, 1:].any()
    model = UnifiedSentimentModel(text_encoder=DummyEncoder()).eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(converted["text_bert"])[None],
            torch.as_tensor(converted["audio"])[None],
            torch.as_tensor(converted["vision"])[None],
            torch.as_tensor(converted["valid_mask"])[None],
            torch.as_tensor(converted["observed_mask"])[None],
            valid_mask_by_modality=torch.as_tensor(converted["valid_mask_by_modality"])[None],
        )
    assert output["cls_logits"].shape == (1, 3)
    assert output["reg_pred"].shape == (1, 1)


def test_invalid_q1_sample_is_rejected():
    invalid = sample()
    invalid["vision"] = np.zeros((50, 34), np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        adapt_q1_sample(invalid, TinyTokenizer())
