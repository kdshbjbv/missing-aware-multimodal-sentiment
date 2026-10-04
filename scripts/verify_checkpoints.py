"""Check the three published compact E7a member checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e7a_core.src.model import UnifiedSentimentModel


class DummyEncoder(nn.Module):
    """Exercise the frozen E7a layers without downloading third-party BERT."""

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor):
        hidden = torch.zeros((*input_ids.shape, 768), dtype=torch.float32, device=input_ids.device)
        return type("EncoderOutput", (), {"last_hidden_state": hidden})()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    manifest = json.loads(Path("e7a_core/checkpoint_manifest.json").read_text(encoding="utf-8"))
    for member in manifest["members"]:
        path = args.checkpoint_dir / Path(member["bundle_path"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != member["sha256"] or path.stat().st_size != member["size_bytes"]:
            raise ValueError(f"Checkpoint hash or size mismatch: {path.name}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        state = bundle.get("model_state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Missing state_dict: {path.name}")
        if any("optimizer" in key.lower() for key in bundle):
            raise ValueError(f"Unexpected optimizer state: {path.name}")
        model = UnifiedSentimentModel(text_encoder=DummyEncoder())
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or [
            key for key in incompatible.missing_keys if not key.startswith("text_encoder.")
        ]:
            raise ValueError(f"State dict does not fit E7a: {path.name}")
        model.eval()
        with torch.no_grad():
            valid = torch.zeros((1, 50), dtype=torch.bool)
            valid[:, :3] = True
            observed = torch.zeros((1, 50, 3), dtype=torch.bool)
            observed[:, :3, 0] = True
            output = model(
                torch.zeros((1, 3, 50), dtype=torch.long),
                torch.zeros((1, 50, 74)),
                torch.zeros((1, 50, 35)),
                valid,
                observed,
            )
        if output["cls_logits"].shape != (1, 3) or output["reg_pred"].shape != (1, 1):
            raise ValueError(f"Unexpected model output shape: {path.name}")
        print(f"{path.name}: {path.stat().st_size} bytes, SHA-256 {digest}, {len(state)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
