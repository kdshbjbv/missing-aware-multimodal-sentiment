from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_config, resolve_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify BERT can load with networking disabled.")
    parser.add_argument("--config", default="configs/e7a_member.yaml")
    return parser.parse_args()


def main() -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    config = load_config(parse_args().config)
    model_config = resolve_model_config(config)
    model_path = model_config["pretrained_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    encoded = tokenizer("offline model check", return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded).last_hidden_state
    print(
        json.dumps(
            {
                "status": "ok",
                "model_path": model_path,
                "model_class": type(model).__name__,
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
                "output_shape": list(output.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
