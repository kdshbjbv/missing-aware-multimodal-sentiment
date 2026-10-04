from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


LOGGER_NAME = "q1_multimodal"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    config["_project_root"] = str(config_path.parent.parent)
    resolve_config_paths(config)
    return config


def resolve_config_paths(config: dict[str, Any]) -> None:
    root = Path(config["_project_root"])
    path_keys = {
        "dataset_root",
        "label_file",
        "work_dir",
        "output_dir",
        "mfa_root_dir",
        "mfa_temp_dir",
        "huggingface_home",
        "openface_feature_extraction",
        "openface_library_dir",
    }
    for key in path_keys:
        value = config.get("paths", {}).get(key)
        if not value:
            continue
        candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if key == "openface_feature_extraction" and candidate.parent == Path("."):
            # A bare command is resolved through PATH by resolve_executable().
            config["paths"][key] = str(value)
            continue
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        config["paths"][key] = str(candidate)


def ensure_directories(config: dict[str, Any]) -> None:
    work = Path(config["paths"]["work_dir"])
    output = Path(config["paths"]["output_dir"])
    directories = [
        work,
        output,
        output / "figures",
        work / "audio",
        work / "mfa" / "corpus",
        work / "mfa" / "aligned",
        Path(config["paths"]["mfa_root_dir"]),
        Path(config["paths"]["mfa_temp_dir"]),
        Path(config["paths"]["huggingface_home"]),
        work / "features" / "text",
        work / "features" / "audio",
        work / "features" / "vision",
        work / "aligned",
        work / "openface",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(config: dict[str, Any], verbose: bool = False) -> logging.Logger:
    ensure_directories(config)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_path = Path(config["paths"]["output_dir"]) / config["output"]["log_file"]
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sample_id(video_id: str, clip_id: str, separator: str = "$_$") -> str:
    return f"{video_id}{separator}{clip_id}"


def safe_sample_name(value: str) -> str:
    return value.replace("$_$", "__").replace("/", "_").replace("\\", "_")


def resolve_executable(value: str | Path) -> str | None:
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(str(value))


def run_command(
    command: Iterable[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(part) for part in command]
    get_logger().debug("Running command: %s", " ".join(command))
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def check_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {bad} NaN/Inf values")


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None
