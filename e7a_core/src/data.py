from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("T", "A", "V")


def load_pickle_compat(path: str | Path) -> Any:
    """Read NumPy 1.x/2.x pickles without modifying the source file."""
    path = Path(path)
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        for name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            module = getattr(np.core, name, None)
            if module is not None:
                sys.modules.setdefault(f"numpy._core.{name}", module)
    with path.open("rb") as handle:
        return pickle.load(handle)


def validate_text_bert(text_bert: np.ndarray, expected_length: int = 50) -> Dict[str, Any]:
    array = np.asarray(text_bert)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1] != 3 or array.shape[2] != expected_length:
        raise ValueError(f"Expected text_bert [N,3,{expected_length}], got {array.shape}")
    input_ids, attention, token_types = array[:, 0], array[:, 1], array[:, 2]
    if not np.all(input_ids == np.round(input_ids)):
        raise ValueError("text_bert input_ids contain non-integer values")
    if not np.isin(attention, [0, 1]).all():
        raise ValueError("text_bert attention channel is not binary")
    if not np.isin(token_types, [0, 1]).all():
        raise ValueError("text_bert token_type_ids channel is not binary")
    if not np.array_equal(attention.astype(bool), input_ids != 0):
        raise ValueError("attention_mask does not match non-PAD input_ids")
    if np.any(np.diff(attention, axis=1) > 0):
        raise ValueError("attention_mask is not a contiguous valid prefix")
    lengths = attention.sum(axis=1).astype(int)
    if np.any(lengths < 2):
        raise ValueError("Every sequence must contain at least CLS and SEP")
    last_tokens = input_ids[np.arange(len(input_ids)), lengths - 1]
    if not np.all(input_ids[:, 0] == 101) or not np.all(last_tokens == 102):
        raise ValueError("BERT CLS/SEP convention was not satisfied")
    return {
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "input_id_min": int(input_ids.min()),
        "input_id_max": int(input_ids.max()),
    }


def build_masks(
    text_bert: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build P_seq, P_mod and O after the verified BERT/padding audit.

    Audio/vision are structurally inapplicable at BERT CLS/SEP. Interior
    all-zero vectors are kept on the sequence axis but marked unobserved.
    """
    tb = np.asarray(text_bert)
    if tb.ndim == 2:
        tb = tb[None, ...]
    audio = np.asarray(audio)
    vision = np.asarray(vision)
    if audio.ndim == 2:
        audio = audio[None, ...]
    if vision.ndim == 2:
        vision = vision[None, ...]
    valid = tb[:, 1, :].astype(bool)
    batch, steps = valid.shape
    p_mod = np.repeat(valid[..., None], 3, axis=-1)
    lengths = valid.sum(axis=1).astype(int)
    p_mod[:, 0, 1:] = False
    p_mod[np.arange(batch), lengths - 1, 1:] = False
    observed = np.zeros((batch, steps, 3), dtype=bool)
    observed[..., 0] = p_mod[..., 0] & (tb[:, 0, :] != 0)
    observed[..., 1] = p_mod[..., 1] & ~np.all(audio == 0, axis=-1)
    observed[..., 2] = p_mod[..., 2] & ~np.all(vision == 0, axis=-1)
    return valid, p_mod, observed


def _normalise_split(split: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("text_bert", "audio", "vision")
    missing = [key for key in required if key not in split]
    if missing:
        raise KeyError(f"Missing fields: {missing}")
    text_bert = np.asarray(split["text_bert"])
    audio = np.asarray(split["audio"])
    vision = np.asarray(split["vision"])
    if text_bert.ndim == 2:
        text_bert = text_bert[None, ...]
    if audio.ndim == 2:
        audio = audio[None, ...]
    if vision.ndim == 2:
        vision = vision[None, ...]
    validate_text_bert(text_bert, text_bert.shape[-1])
    if audio.shape[:2] != (len(text_bert), text_bert.shape[-1]):
        raise ValueError(f"Audio shape mismatch: {audio.shape} vs {text_bert.shape}")
    if vision.shape[:2] != (len(text_bert), text_bert.shape[-1]):
        raise ValueError(f"Vision shape mismatch: {vision.shape} vs {text_bert.shape}")
    valid, p_mod, observed = build_masks(text_bert, audio, vision)
    return {
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "valid_mask": valid,
        "valid_mask_by_modality": p_mod,
        "observed_mask": observed,
    }


class Attachment2Dataset(Dataset):
    def __init__(self, path: str | Path, split: str) -> None:
        data = load_pickle_compat(path)
        if split not in ("train", "valid", "test") or split not in data:
            raise KeyError(f"Invalid or absent split: {split}")
        source = data[split]
        normalised = _normalise_split(source)
        self.__dict__.update(normalised)
        self.sample_ids = [str(value) for value in source["id"]]
        self.class_labels = np.asarray(source["classification_labels"], dtype=np.int64)
        self.reg_labels = np.asarray(source["regression_labels"], dtype=np.float32)
        self.raw_text = np.asarray(source.get("raw_text", [""] * len(self.sample_ids)))
        if not (len(self.sample_ids) == len(self.class_labels) == len(self.reg_labels)):
            raise ValueError("Attachment 2 sample counts are inconsistent")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_ids[index],
            "raw_text": str(self.raw_text[index]),
            "text_bert": torch.as_tensor(self.text_bert[index], dtype=torch.long),
            "audio": torch.as_tensor(self.audio[index], dtype=torch.float32),
            "vision": torch.as_tensor(self.vision[index], dtype=torch.float32),
            "valid_mask": torch.as_tensor(self.valid_mask[index], dtype=torch.bool),
            "valid_mask_by_modality": torch.as_tensor(
                self.valid_mask_by_modality[index], dtype=torch.bool
            ),
            "observed_mask": torch.as_tensor(self.observed_mask[index], dtype=torch.bool),
            "classification_label": torch.tensor(int(self.class_labels[index]), dtype=torch.long),
            "regression_label": torch.tensor(float(self.reg_labels[index]), dtype=torch.float32),
        }


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


class FileCollectionDataset(Dataset):
    def __init__(self, directory: str | Path, kind: str) -> None:
        self.kind = kind
        self.paths = sorted(Path(directory).glob("*.pkl"), key=natural_key)
        if not self.paths:
            raise FileNotFoundError(f"No pkl files found in {directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        obj = load_pickle_compat(path)
        if self.kind == "q2":
            source = obj["test"]
            sample_id = path.stem
        elif self.kind == "q3":
            source = obj
            sample_id = str(obj.get("id", path.stem))
        else:
            raise ValueError(f"Unknown collection kind: {self.kind}")
        normalised = _normalise_split(source)
        item = {
            "sample_id": sample_id,
            "raw_text": str(source.get("raw_text", "")),
            "source_path": str(path),
        }
        for key, value in normalised.items():
            value = value[0]
            dtype = torch.long if key == "text_bert" else (
                torch.bool if "mask" in key else torch.float32
            )
            item[key] = torch.as_tensor(value, dtype=dtype)
        return item


class Attachment3Dataset(FileCollectionDataset):
    def __init__(self, directory: str | Path) -> None:
        super().__init__(directory, "q2")


class Attachment4Dataset(FileCollectionDataset):
    def __init__(self, directory: str | Path) -> None:
        super().__init__(directory, "q3")
