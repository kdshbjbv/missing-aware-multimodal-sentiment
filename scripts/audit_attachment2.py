"""Summarize pickle schema and labels without exposing text or sample IDs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e7a_core.src.data import load_pickle_compat


def summary(path: Path) -> dict:
    data = load_pickle_compat(path)
    result = {"filename": path.name, "top_level_keys": list(data)}
    result["splits"] = {}
    for name in ("train", "valid", "test"):
        if name not in data:
            continue
        section = data[name]
        fields = {}
        for key, value in section.items():
            array = np.asarray(value)
            fields[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        labels = section.get("classification_labels")
        if labels is not None:
            classes, counts = np.unique(np.asarray(labels), return_counts=True)
            fields["classification_distribution"] = dict(zip(map(str, classes), map(int, counts)))
        result["splits"][name] = fields
    del data
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summary(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
