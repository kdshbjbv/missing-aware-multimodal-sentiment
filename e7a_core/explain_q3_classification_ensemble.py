from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data import Attachment4Dataset  # noqa: E402
from src.ensemble_inference import load_inference_model, validate_members  # noqa: E402
from src.explain_classification import (  # noqa: E402
    class_logit_prediction,
    exact_modality_shapley_class_logit,
    integrated_gradients_projected_class_logit,
)
from src.model import MODALITIES  # noqa: E402
from src.utils import environment_manifest, load_config, resolve_path, save_json, select_device  # noqa: E402


COALITIONS = ("empty", "T", "A", "V", "TA", "TV", "AV", "TAV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a frozen ensemble's final predicted-class logit on attachment 4."
    )
    parser.add_argument("--config", default="configs/e7a_member.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--member_names", nargs="+", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--class_biases", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def heatmap(values: np.ndarray, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.2))
    image = axis.imshow(values.T, aspect="auto", cmap="magma")
    axis.set_yticks(range(3), MODALITIES)
    axis.set_xlabel("Aligned position")
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def batch_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in (
            "text_bert", "audio", "vision", "valid_mask",
            "valid_mask_by_modality", "observed_mask",
        )
    }


def removed_class_logit(
    model: torch.nn.Module,
    projected: torch.Tensor,
    valid_mask: torch.Tensor,
    applicable: torch.Tensor,
    target_class: int,
    class_bias: float,
    indices: np.ndarray,
) -> float:
    observed = applicable.clone()
    modified = projected.clone()
    if len(indices):
        positions = torch.as_tensor(indices[:, 0], device=projected.device)
        modalities = torch.as_tensor(indices[:, 1], device=projected.device)
        observed[0, positions, modalities] = False
        modified[0, positions, modalities] = 0.0
    with torch.no_grad():
        value = class_logit_prediction(
            model, modified, valid_mask, observed, applicable,
            target_class, class_bias,
        )
    return float(value[0])


def main() -> int:
    args = parse_args()
    if any(abs(float(value)) > 1e-12 for value in args.class_biases):
        raise ValueError("Frozen E7a explanations require class_biases=[0,0,0]")
    config = load_config(args.config)
    device = select_device(args.device)
    checkpoint_paths = [resolve_path(value) for value in args.checkpoints]
    member_names = validate_members(checkpoint_paths, args.member_names)
    data_dir = resolve_path(args.data_dir or config["data"]["attachment4_aligned"])
    dataset = Attachment4Dataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions_path = resolve_path(args.predictions)
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    required = {"sample_id", "pred_class_index", "pred_class_name"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file lacks columns: {sorted(missing)}")
    prediction_ids = predictions["sample_id"].astype(str).tolist()
    dataset_ids = [str(dataset[index]["sample_id"]) for index in range(len(dataset))]
    if prediction_ids != dataset_ids:
        raise ValueError("Prediction rows do not match attachment 4 sample ordering")
    target_classes = predictions["pred_class_index"].to_numpy(dtype=np.int64)

    sample_count = len(dataset)
    member_count = len(member_names)
    steps = int(config["data"]["sequence_length"])
    complete = np.zeros(sample_count, dtype=bool)
    applicable_all = np.zeros((sample_count, steps, 3), dtype=bool)
    shapley = np.full((member_count, sample_count, 3), np.nan, dtype=np.float64)
    coalition = np.full((member_count, sample_count, len(COALITIONS)), np.nan, dtype=np.float64)
    ig_signed = np.full((member_count, sample_count, steps, 3), np.nan, dtype=np.float64)
    ig_absolute = np.full_like(ig_signed, np.nan)
    ig_error = np.full((member_count, sample_count), np.nan, dtype=np.float64)
    member_metadata = []
    explain_cfg = config["explain"]

    for member_index, (checkpoint_path, member_name) in enumerate(zip(checkpoint_paths, member_names)):
        print(f"class-logit explanation pass: {member_name}", flush=True)
        model, metadata = load_inference_model(config, checkpoint_path, device)
        member_metadata.append({"name": member_name, **metadata})
        for sample_index, batch in enumerate(loader):
            tensors = batch_tensors(batch, device)
            applicable = tensors["valid_mask_by_modality"]
            observed = tensors["observed_mask"]
            is_complete = bool(torch.all(observed == applicable))
            if member_index == 0:
                complete[sample_index] = is_complete
                applicable_all[sample_index] = applicable[0].cpu().numpy()
            elif complete[sample_index] != is_complete:
                raise RuntimeError("Completeness changed between ensemble members")
            if not is_complete:
                continue
            with torch.no_grad():
                projected = model.encode_projected(
                    tensors["text_bert"], tensors["audio"], tensors["vision"]
                ).detach()
            target_class = int(target_classes[sample_index])
            class_bias = float(args.class_biases[target_class])
            shapley_result = exact_modality_shapley_class_logit(
                model, projected, tensors["valid_mask"], applicable,
                target_class, class_bias,
            )
            for modality_index, modality in enumerate(MODALITIES):
                shapley[member_index, sample_index, modality_index] = shapley_result["shapley"][modality]
            for coalition_index, key in enumerate(COALITIONS):
                coalition[member_index, sample_index, coalition_index] = shapley_result["coalition_values"][key]
            ig_result = integrated_gradients_projected_class_logit(
                model, projected, tensors["valid_mask"], applicable,
                target_class, class_bias, steps=int(explain_cfg["ig_steps"]),
            )
            ig_signed[member_index, sample_index] = ig_result["signed_by_position"][0].detach().cpu().numpy()
            ig_absolute[member_index, sample_index] = ig_result["absolute_by_position"][0].detach().cpu().numpy()
            ig_error[member_index, sample_index] = float(ig_result["completeness_error"][0])
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_shapley = np.full((sample_count, 3), np.nan, dtype=np.float64)
    std_shapley = np.full_like(mean_shapley, np.nan)
    mean_coalition = np.full((sample_count, len(COALITIONS)), np.nan, dtype=np.float64)
    mean_ig_signed = np.full((sample_count, steps, 3), np.nan, dtype=np.float64)
    mean_ig_absolute = np.full_like(mean_ig_signed, np.nan)
    std_ig_absolute = np.full_like(mean_ig_signed, np.nan)
    mean_shapley[complete] = shapley[:, complete].mean(axis=0)
    std_shapley[complete] = shapley[:, complete].std(axis=0)
    mean_coalition[complete] = coalition[:, complete].mean(axis=0)
    mean_ig_signed[complete] = ig_signed[:, complete].mean(axis=0)
    mean_ig_absolute[complete] = ig_absolute[:, complete].mean(axis=0)
    std_ig_absolute[complete] = ig_absolute[:, complete].std(axis=0)

    output = resolve_path(args.output_dir)
    figures = output / "class_logit_figures"
    figures.mkdir(parents=True, exist_ok=True)
    shapley_rows = []
    ig_rows = []
    selected_sets: dict[int, list[tuple[str, int, np.ndarray]]] = {}
    for sample_index, sample_id in enumerate(dataset_ids):
        if not complete[sample_index]:
            continue
        target_class = int(target_classes[sample_index])
        row = {
            "sample_id": sample_id,
            "target_class_index": target_class,
            "target_class_name": str(predictions.iloc[sample_index]["pred_class_name"]),
            "class_bias": float(args.class_biases[target_class]),
            **{
                f"raw_v_{key}": float(mean_coalition[sample_index, coalition_index])
                for coalition_index, key in enumerate(COALITIONS)
            },
        }
        for modality_index, modality in enumerate(MODALITIES):
            row[f"phi_{modality}"] = float(mean_shapley[sample_index, modality_index])
            row[f"phi_{modality}_member_std"] = float(std_shapley[sample_index, modality_index])
            for member_index, member_name in enumerate(member_names):
                row[f"phi_{modality}_{member_name}"] = float(shapley[member_index, sample_index, modality_index])
        row["additivity_error"] = float(
            mean_shapley[sample_index].sum()
            - (mean_coalition[sample_index, COALITIONS.index("TAV")] - mean_coalition[sample_index, COALITIONS.index("empty")])
        )
        primary_index = int(np.argmax(np.abs(mean_shapley[sample_index])))
        row["primary_modality_by_abs_phi"] = MODALITIES[primary_index]
        shapley_rows.append(row)

        legal = applicable_all[sample_index]
        sample_ig_rows = []
        for position, modality_index in np.argwhere(legal):
            ig_row = {
                "sample_id": sample_id,
                "target_class_index": target_class,
                "target_class_name": str(predictions.iloc[sample_index]["pred_class_name"]),
                "modality": MODALITIES[int(modality_index)],
                "aligned_position": int(position),
                "ensemble_signed_attribution": float(mean_ig_signed[sample_index, position, modality_index]),
                "mean_member_absolute_importance": float(mean_ig_absolute[sample_index, position, modality_index]),
                "member_std_absolute_importance": float(std_ig_absolute[sample_index, position, modality_index]),
                "mean_ig_completeness_error": float(np.nanmean(ig_error[:, sample_index])),
                "mapping_status": "aligned_position_only",
            }
            ig_rows.append(ig_row)
            sample_ig_rows.append(ig_row)
        heatmap(
            mean_ig_absolute[sample_index],
            f"{sample_id} predicted-class raw logits IG",
            figures / f"{sample_id}_class_logit_ig.png",
        )
        candidates = np.argwhere(legal)
        scores = mean_ig_absolute[sample_index][legal]
        k = max(1, int(round(len(candidates) * float(explain_cfg["topk_fraction"]))))
        top = candidates[np.argsort(scores)[-min(k, len(candidates)):]]
        selections: list[tuple[str, int, np.ndarray]] = [("top", 0, top)]
        rng = np.random.default_rng(int(explain_cfg["random_seed"]) + sample_index)
        for repeat in range(int(explain_cfg["random_repeats"])):
            picked = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
            selections.append(("random", repeat, candidates[picked]))
        selected_sets[sample_index] = selections

    selection_count = 1 + int(explain_cfg["random_repeats"])
    full_values = np.full((member_count, sample_count), np.nan, dtype=np.float64)
    removed_values = np.full((member_count, sample_count, selection_count), np.nan, dtype=np.float64)
    for member_index, (checkpoint_path, member_name) in enumerate(zip(checkpoint_paths, member_names)):
        print(f"class-logit faithfulness pass: {member_name}", flush=True)
        model, _ = load_inference_model(config, checkpoint_path, device)
        for sample_index, batch in enumerate(loader):
            if not complete[sample_index]:
                continue
            tensors = batch_tensors(batch, device)
            applicable = tensors["valid_mask_by_modality"]
            with torch.no_grad():
                projected = model.encode_projected(
                    tensors["text_bert"], tensors["audio"], tensors["vision"]
                ).detach()
            target_class = int(target_classes[sample_index])
            class_bias = float(args.class_biases[target_class])
            full_values[member_index, sample_index] = removed_class_logit(
                model, projected, tensors["valid_mask"], applicable,
                target_class, class_bias, np.empty((0, 2), dtype=np.int64),
            )
            for selection_index, (_, _, indices) in enumerate(selected_sets[sample_index]):
                removed_values[member_index, sample_index, selection_index] = removed_class_logit(
                    model, projected, tensors["valid_mask"], applicable,
                    target_class, class_bias, indices,
                )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    faith_rows = []
    for sample_index, selections in selected_sets.items():
        ensemble_full = float(np.nanmean(full_values[:, sample_index]))
        for selection_index, (selection, repeat, indices) in enumerate(selections):
            ensemble_removed = float(np.nanmean(removed_values[:, sample_index, selection_index]))
            faith_rows.append(
                {
                    "sample_id": dataset_ids[sample_index],
                    "target_class_index": int(target_classes[sample_index]),
                    "selection": selection,
                    "repeat": repeat,
                    "k": int(len(indices)),
                    "ensemble_full_raw_logit": ensemble_full,
                    "ensemble_removed_raw_logit": ensemble_removed,
                    "absolute_change": abs(ensemble_full - ensemble_removed),
                    "signed_change": ensemble_removed - ensemble_full,
                }
            )

    pd.DataFrame(shapley_rows).to_csv(output / "q3_class_logit_shapley.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ig_rows).to_csv(output / "q3_class_logit_position_ig.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(faith_rows).to_csv(output / "q3_class_logit_faithfulness.csv", index=False, encoding="utf-8-sig")
    save_json(
        {
            "method": "mean_of_member_raw_predicted_class_logit_explanations",
            "members": member_metadata,
            "class_biases": args.class_biases,
            "predictions": str(predictions_path),
            "sample_count": sample_count,
            "explained_complete_samples": int(complete.sum()),
            "blocked_incomplete_sample_ids": [dataset_ids[i] for i in range(sample_count) if not complete[i]],
            "explanation_target": "final_ensemble_predicted_class_raw_logit",
            "bias_note": "constant class bias is in coalition values and cancels from Shapley/IG contributions",
            "mapping_status": "aligned_position_only",
            "environment": environment_manifest(),
        },
        output / "q3_class_logit_explanation_manifest.json",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
