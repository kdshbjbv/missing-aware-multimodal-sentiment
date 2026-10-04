from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_config


def read_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    required = [
        ("paths", "dataset_root"),
        ("paths", "label_file"),
        ("paths", "work_dir"),
        ("paths", "output_dir"),
        ("project", "max_len"),
    ]
    missing = [f"{section}.{key}" for section, key in required if not config.get(section, {}).get(key)]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
    return config
