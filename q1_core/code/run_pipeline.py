from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.align_modalities import align_one_sample
from src.align_text_mfa import (
    collect_mfa_timestamps,
    download_mfa_models,
    prepare_mfa_corpus,
    run_mfa_alignment,
)
from src.build_dataset import build_dataset
from src.config import read_config
from src.extract_audio import extract_one_audio
from src.extract_audio_features import extract_one_audio_features
from src.extract_text_bert import BertWordExtractor, extract_one_text
from src.extract_visual_openface import extract_one_visual
from src.prepare_manifest import load_manifest, prepare_manifest
from src.utils import ensure_directories, get_logger, set_seed, setup_logging
from src.validate_features import validate_dataset
from src.visualize_alignment import plot_alignment_example


STAGE_ORDER = ["audit", "audio", "mfa", "text", "acoustic", "visual", "align", "build", "validate", "plot"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question 1 multimodal feature extraction pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "feature_config.yaml"))
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages: audit,audio,mfa,text,acoustic,visual,align,build,validate,plot or all",
    )
    parser.add_argument("--sample-id", action="append", default=[], help="Process only the named sample; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="Process the first N selected samples")
    parser.add_argument("--force", action="store_true", help="Ignore caches and re-run stages")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--download-mfa-models", action="store_true")
    parser.add_argument("--non-strict-validation", action="store_true")
    return parser.parse_args()


def selected_stages(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return STAGE_ORDER.copy()
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    return [stage for stage in STAGE_ORDER if stage in requested]


def filter_manifest(frame: pd.DataFrame, sample_ids: list[str], limit: int | None) -> pd.DataFrame:
    selected = frame
    if sample_ids:
        missing = sorted(set(sample_ids) - set(frame["sample_id"].astype(str)))
        if missing:
            raise ValueError(f"Unknown sample IDs: {missing}")
        selected = frame[frame["sample_id"].isin(sample_ids)]
    if limit is not None:
        selected = selected.head(limit)
    return selected.reset_index(drop=True)


def run_per_sample(
    name: str,
    frame: pd.DataFrame,
    function: Callable[[pd.Series], object],
    state: dict[str, dict[str, str]],
) -> None:
    logger = get_logger()
    for _, row in frame.iterrows():
        sid = str(row["sample_id"])
        try:
            function(row)
            state.setdefault(sid, {})[name] = "success"
        except Exception as exc:
            state.setdefault(sid, {})[name] = f"failed: {exc}"
            logger.error("Stage %s failed for %s: %s", name, sid, exc)
            logger.debug(traceback.format_exc())


def save_state(config: dict, state: dict[str, dict[str, str]]) -> None:
    path = Path(config["paths"]["work_dir"]) / "stage_status.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    ensure_directories(config)
    logger = setup_logging(config, verbose=args.verbose)
    set_seed(int(config["project"].get("seed", 2026)))
    stages = selected_stages(args.stages)
    logger.info("Starting stages: %s", ", ".join(stages))
    state: dict[str, dict[str, str]] = {}

    if "audit" in stages or not (Path(config["paths"]["work_dir"]) / config["output"]["manifest_file"]).exists():
        manifest = prepare_manifest(config)
    else:
        manifest = load_manifest(config)
    selected = filter_manifest(manifest, args.sample_id, args.limit)
    logger.info("Selected %d/%d samples", len(selected), len(manifest))

    if "audio" in stages:
        run_per_sample("audio", selected, lambda row: extract_one_audio(config, row, args.force), state)

    if "mfa" in stages:
        try:
            prepare_mfa_corpus(config, selected, force=args.force)
            if args.download_mfa_models:
                download_mfa_models(config)
            clean_mfa = args.force or bool(config["alignment"].get("mfa_clean_before_align", True))
            run_mfa_alignment(config, clean=clean_mfa)
            mfa_status = collect_mfa_timestamps(config, selected)
            for sid, value in mfa_status.items():
                state.setdefault(sid, {})["mfa"] = value
        except Exception as exc:
            logger.error("MFA batch stage failed: %s", exc)
            logger.debug(traceback.format_exc())
            for sid in selected["sample_id"].astype(str):
                state.setdefault(sid, {})["mfa"] = f"failed: {exc}"

    if "text" in stages:
        try:
            bert = BertWordExtractor(config)
            run_per_sample("text", selected, lambda row: extract_one_text(config, row, bert, args.force), state)
        except Exception as exc:
            logger.error("BERT initialization failed: %s", exc)
            logger.debug(traceback.format_exc())
            for sid in selected["sample_id"].astype(str):
                state.setdefault(sid, {})["text"] = f"failed: {exc}"

    if "acoustic" in stages:
        run_per_sample(
            "acoustic", selected, lambda row: extract_one_audio_features(config, row, args.force), state
        )

    if "visual" in stages:
        run_per_sample("visual", selected, lambda row: extract_one_visual(config, row, args.force), state)

    if "align" in stages:
        run_per_sample("align", selected, lambda row: align_one_sample(config, row, args.force), state)

    if "build" in stages:
        target_manifest = manifest if not args.sample_id and args.limit is None else selected
        try:
            build_dataset(config, target_manifest)
        except Exception as exc:
            logger.error("Dataset build failed: %s", exc)
            logger.debug(traceback.format_exc())

    if "validate" in stages:
        target_manifest = manifest if not args.sample_id and args.limit is None else selected
        try:
            validate_dataset(config, target_manifest, strict=not args.non_strict_validation)
        except Exception as exc:
            logger.error("Validation failed: %s", exc)

    if "plot" in stages:
        candidates = selected
        plotted = False
        for _, row in candidates.iterrows():
            try:
                plot_alignment_example(config, row)
                plotted = True
                break
            except Exception as exc:
                logger.warning("Cannot plot %s: %s", row["sample_id"], exc)
        if not plotted:
            logger.error("No alignment example could be generated")

    save_state(config, state)
    failed = sum(any(value.startswith("failed") for value in stages_.values()) for stages_ in state.values())
    logger.info("Pipeline finished; %d selected samples have at least one recorded failure", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
