from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import read_config
from src.utils import package_version, resolve_executable


def main() -> int:
    config = read_config(PROJECT_ROOT / "configs" / "feature_config.yaml")
    packages = ["numpy", "pandas", "yaml", "openpyxl", "matplotlib", "torch", "transformers", "opensmile", "textgrid"]
    report = {
        "python": sys.version,
        "packages": {
            name: {"installed": bool(importlib.util.find_spec(name)), "version": package_version(name)}
            for name in packages
        },
        "executables": {
            "ffmpeg": resolve_executable(config["paths"]["ffmpeg"]),
            "ffprobe": resolve_executable(config["paths"]["ffprobe"]),
            "mfa": resolve_executable(config["paths"]["mfa"]),
            "openface": resolve_executable(config["paths"]["openface_feature_extraction"]),
            "git": shutil.which("git"),
        },
        "paths": {
            "dataset_root_exists": Path(config["paths"]["dataset_root"]).is_dir(),
            "label_file_exists": Path(config["paths"]["label_file"]).is_file(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = ["numpy", "pandas", "yaml", "openpyxl"]
    return 0 if all(report["packages"][name]["installed"] for name in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
