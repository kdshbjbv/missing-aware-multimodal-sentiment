from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
# This submission bundle is self-contained at E7a_model/.  Relative config,
# data, checkpoint and output paths are therefore resolved from that directory.
REPO_ROOT = PROJECT_DIR


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return portable model kwargs with a repository-relative path resolved."""
    model_config = dict(config["model"])
    pretrained_name = str(model_config["pretrained_name"])
    candidate = resolve_path(pretrained_name)
    if candidate.is_dir():
        model_config["pretrained_name"] = str(candidate)
    elif bool(model_config.get("local_files_only", False)):
        raise FileNotFoundError(
            "Offline pretrained model directory not found: "
            f"{candidate}. Copy the complete bert-base-uncased directory to this path."
        )
    return model_config


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, ensure_ascii=False, indent=2)


def make_run_dir(output_root: str | Path, name: Optional[str] = None) -> Path:
    root = resolve_path(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / (name or stamp)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    for child in ("checkpoints", "logs", "results", "figures", "explanation_cards"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def create_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def environment_manifest() -> Dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        memory_info = {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
        }
    except Exception:
        memory_info = None
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_summary = None
    if nvidia_smi:
        try:
            nvidia_summary = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except Exception as exc:
            nvidia_summary = f"unavailable: {type(exc).__name__}"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "system_memory": memory_info,
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_summary": nvidia_summary,
    }
