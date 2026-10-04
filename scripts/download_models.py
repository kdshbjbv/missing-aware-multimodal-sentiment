"""Download the public BERT encoder and tokenizer required by E7a.

The checkpoint files in this repository contain only the E7a task layers;
the frozen BERT encoder is obtained from its upstream model repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output_dir", type=Path, default=Path("pretrained/bert-base-uncased"))
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model)
    tokenizer.save_pretrained(output_dir)
    encoder.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    AutoModel.from_pretrained(output_dir, local_files_only=True)
    print(f"BERT_MODEL_READY: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
