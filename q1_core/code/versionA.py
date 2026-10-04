"""Build version A: official Excel text + MFA + BERT/COVAREP/OpenFace.

Data inputs are limited to the 100 source videos, label-100.xlsx, and the
precomputed COVAREP MAT files. Version-B WhisperX and an independently cached
Pyannote VAD audit expose official omissions and voiced speech that has no ASR
word coverage. ASR text embeddings and labels are never copied.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.align_modalities import align_one_sample
from src.align_text_mfa import (
    collect_mfa_timestamps,
    download_mfa_models,
    prepare_mfa_corpus,
    run_mfa_alignment,
)
from src.config import read_config
from src.extract_audio import extract_one_audio
from src.extract_audio_features import audio_feature_path
from src.extract_text_bert import BertWordExtractor, extract_many_texts
from src.extract_visual_openface import extract_one_visual
from src.prepare_manifest import prepare_manifest
from src.problem1_dataset import (
    build_problem1_dataset,
    covarep_frame_times,
    generate_problem1_figures,
    load_covarep_mat,
    pool_points_by_word_windows,
    save_problem1_dataset,
    wav_statistics,
)
from src.utils import ensure_directories, get_logger, safe_sample_name, set_seed, setup_logging
from src.vision_35 import extract_visual35_from_csv
from validate_problem1 import validate


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--dataset-root", default=None, help="Root containing video_id/clip_id.mp4")
    parser.add_argument("--label-file", default=None, help="label-100.xlsx")
    parser.add_argument(
        "--covarep-dir",
        default="work/audio",
        help="Directory containing <safe_sample_name>.mat COVAREP files",
    )
    parser.add_argument(
        "--work-dir",
        default="work/versionA",
        help="Isolated directory for generated WAV/MFA/BERT/OpenFace/aligned caches",
    )
    parser.add_argument("--output", default="outputs/versionA.pkl")
    parser.add_argument("--summary", default="outputs/versionA_feature_summary.csv")
    parser.add_argument("--validation-report", default="outputs/versionA_validation_report.json")
    parser.add_argument(
        "--asr-reference-pkl",
        default="outputs/versionB.pkl",
        help=(
            "WhisperX Version-B PKL used only to expose confidently detected "
            "official-transcript omissions on a real word timeline"
        ),
    )
    parser.add_argument(
        "--disable-official-text-gap-fill",
        action="store_true",
        help="Keep the legacy MFA-only timeline even when ASR proves official words are omitted",
    )
    parser.add_argument(
        "--gap-report",
        default="outputs/versionA_official_text_gap_report.json",
        help="Audit report for official-text positions filled from the ASR timeline",
    )
    parser.add_argument(
        "--vad-unrecognized-audit",
        default="outputs/versionA1_vad_unrecognized_audit.json",
        help="Accepted Pyannote-VAD/WhisperX coverage-gap candidates",
    )
    parser.add_argument(
        "--vad-gap-report",
        default="outputs/versionA_vad_text_gap_report.json",
        help="Audit report for [TEXT_MISSING] positions inserted from VAD evidence",
    )
    parser.add_argument(
        "--disable-vad-text-gap-fill",
        action="store_true",
        help="Do not insert VAD-confirmed speech regions missing from WhisperX text",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=16,
        help="Number of short utterances per BERT GPU forward pass",
    )
    parser.add_argument(
        "--openface-cpu-threads",
        type=int,
        default=8,
        help="CPU threads available to OpenFace while BERT uses the GPU",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate cached intermediate artifacts")
    parser.add_argument("--download-mfa-models", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def configure(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(project_path(args.config))
    work = project_path(args.work_dir)
    output = project_path("outputs")
    config["paths"]["work_dir"] = str(work)
    config["paths"]["output_dir"] = str(output)
    config["paths"]["mfa_temp_dir"] = str(work / "mfa" / "temp")
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(project_path(args.dataset_root))
    if args.label_file:
        config["paths"]["label_file"] = str(project_path(args.label_file))
    if args.bert_batch_size < 1:
        raise ValueError("--bert-batch-size must be at least 1")
    if args.openface_cpu_threads < 1:
        raise ValueError("--openface-cpu-threads must be at least 1")
    config["text"]["batch_size"] = int(args.bert_batch_size)
    # Keep the shared MFA/Hugging Face model caches from the base YAML. They
    # contain tool/model assets only, not sample features, so A/B artifacts
    # remain isolated without downloading the same models twice.
    config["output"]["log_file"] = "versionA_processing.log"
    ensure_directories(config)
    return config


def copy_covarep_inputs(
    config: dict[str, Any], manifest: Any, source_dir: Path, force: bool
) -> None:
    logger = get_logger()
    target_dir = Path(config["paths"]["work_dir"]) / "audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for sample_id in manifest["sample_id"].astype(str):
        stem = safe_sample_name(sample_id)
        source = source_dir / f"{stem}.mat"
        target = target_dir / f"{stem}.mat"
        if source.resolve() == target.resolve():
            if source.is_file():
                copied += 1
            else:
                missing += 1
            continue
        if source.is_file():
            if force or not target.is_file() or source.stat().st_size != target.stat().st_size:
                shutil.copy2(source, target)
            copied += 1
        else:
            # The isolated work tree is derived data.  Never let a stale MAT
            # make a currently missing source look available.
            if target.is_file():
                target.unlink()
            missing += 1
    logger.info("COVAREP inputs mirrored: %d present, %d missing", copied, missing)


def extract_all_audio(config: dict[str, Any], manifest: Any, force: bool) -> None:
    logger = get_logger()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        extract_one_audio(config, row, force=force)
        logger.info("Audio [%d/%d] %s", index, len(manifest), row["sample_id"])


def prepare_covarep_frame_cache(config: dict[str, Any], manifest: Any, force: bool) -> None:
    """Convert provided MATs to the frame-NPZ schema used by align_one_sample."""
    logger = get_logger()
    settings = config["problem1"]
    work = Path(config["paths"]["work_dir"])
    expected_dim = int(settings["covarep_expected_dim"])
    reference_names: list[str] | None = None

    # Establish and validate the canonical feature order before writing caches.
    for sample_id in manifest["sample_id"].astype(str):
        mat_path = work / "audio" / f"{safe_sample_name(sample_id)}.mat"
        if not mat_path.is_file():
            continue
        _, names, _ = load_covarep_mat(
            mat_path,
            str(settings["covarep_feature_key"]),
            str(settings["covarep_names_key"]),
            expected_dim,
        )
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError(f"Inconsistent COVAREP feature order: {mat_path}")
    if reference_names is None:
        raise RuntimeError("No COVAREP MAT files were found")

    threshold = float(settings.get("silent_max_abs_threshold", 1.0e-12))
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        sample_id = str(row["sample_id"])
        stem = safe_sample_name(sample_id)
        destination = audio_feature_path(config, sample_id)
        if destination.is_file() and not force:
            logger.info("COVAREP frame cache [%d/%d] hit: %s", index, len(manifest), sample_id)
            continue
        wav_path = work / "audio" / f"{stem}.wav"
        mat_path = work / "audio" / f"{stem}.mat"
        if mat_path.is_file():
            features, names, _ = load_covarep_mat(
                mat_path,
                str(settings["covarep_feature_key"]),
                str(settings["covarep_names_key"]),
                expected_dim,
            )
            if names != reference_names:
                raise ValueError(f"Inconsistent COVAREP feature order: {mat_path}")
            times = covarep_frame_times(
                wav_path,
                len(features),
                float(settings["covarep_hop_seconds"]),
                float(settings.get("covarep_first_sample_fraction", 0.5)),
            )
            valid = np.ones(len(features), dtype=np.uint8)
            extractor = "COVAREP 74-dimensional acoustic features"
            source_mat = str(mat_path)
        else:
            statistics = wav_statistics(wav_path)
            if float(statistics["max_abs"]) > threshold:
                raise FileNotFoundError(
                    f"Non-silent sample has no COVAREP MAT: {sample_id} ({mat_path})"
                )
            features = np.zeros((0, expected_dim), dtype=np.float32)
            times = np.zeros(0, dtype=np.float64)
            valid = np.zeros(0, dtype=np.uint8)
            extractor = "COVAREP unavailable because source WAV is digital silence"
            source_mat = ""
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            times=np.asarray(times, dtype=np.float64),
            features=np.asarray(features, dtype=np.float32),
            valid=valid,
            feature_names=np.asarray(reference_names, dtype=np.str_),
            extractor=np.asarray(extractor, dtype=np.str_),
            source_mat=np.asarray(source_mat, dtype=np.str_),
        )
        logger.info("COVAREP frame cache [%d/%d] %s", index, len(manifest), sample_id)


def run_mfa(config: dict[str, Any], manifest: Any, force: bool, download: bool) -> None:
    logger = get_logger()
    prepare_mfa_corpus(config, manifest, force=force)
    if download:
        download_mfa_models(config)
    aligned_root = Path(config["paths"]["work_dir"]) / "mfa" / "aligned"
    existing_grids = list(aligned_root.rglob("*.TextGrid")) + list(aligned_root.rglob("*.textgrid"))
    if force or not existing_grids:
        clean = force or bool(config["alignment"].get("mfa_clean_before_align", True))
        run_mfa_alignment(config, clean=clean)
    else:
        logger.info(
            "Resuming MFA from %d existing TextGrid files; batch alignment is not repeated",
            len(existing_grids),
        )

    failed: dict[str, str] = {}
    for attempt in range(1, 4):
        status = collect_mfa_timestamps(config, manifest)
        failed = {sample_id: value for sample_id, value in status.items() if value != "success"}
        if not failed:
            break
        logger.warning(
            "MFA collection attempt %d/3 left %d sample(s): %s",
            attempt,
            len(failed),
            sorted(failed),
        )
    if failed:
        raise RuntimeError(f"MFA did not produce valid word timestamps for all samples: {failed}")


def extract_all_text(config: dict[str, Any], manifest: Any, force: bool) -> None:
    logger = get_logger()
    extractor = BertWordExtractor(config)
    outputs = extract_many_texts(config, manifest, extractor, force=force)
    logger.info("BERT extraction complete: %d/%d samples", len(outputs), len(manifest))


def extract_all_visual(config: dict[str, Any], manifest: Any, force: bool) -> None:
    logger = get_logger()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        extract_one_visual(config, row, force=force)
        logger.info("OpenFace [%d/%d] %s", index, len(manifest), row["sample_id"])


def align_all(config: dict[str, Any], manifest: Any, force: bool) -> None:
    logger = get_logger()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        align_one_sample(config, row, force=force)
        logger.info("Legacy word cache [%d/%d] %s", index, len(manifest), row["sample_id"])


def _normalize_alignment_word(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower().replace("’", "'"))


def _subsequence_positions(
    official_words: list[str], asr_words: list[str]
) -> list[int] | None:
    """Return ASR indices when every official word occurs in order.

    The rule is intentionally strict: substitutions and official-only words are
    rejected. This prevents ordinary ASR recognition errors from being labelled
    as omissions in the official transcript.
    """
    official = [_normalize_alignment_word(word) for word in official_words]
    asr = [_normalize_alignment_word(word) for word in asr_words]
    if not official or any(not word for word in official) or any(not word for word in asr):
        return None
    positions: list[int] = []
    cursor = 0
    for word in official:
        match = next((index for index in range(cursor, len(asr)) if asr[index] == word), None)
        if match is None:
            return None
        positions.append(match)
        cursor = match + 1
    return positions


def apply_official_text_gap_fill(
    dataset: dict[str, Any], summary: Any, asr_reference_path: Path
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Expose official-transcript omissions as zero text vectors on an ASR timeline."""
    if not asr_reference_path.is_file():
        raise FileNotFoundError(
            "Official-text gap filling requires a Version-B WhisperX PKL: "
            f"{asr_reference_path}"
        )
    with asr_reference_path.open("rb") as handle:
        asr_dataset = pickle.load(handle)
    if str(asr_dataset.get("meta", {}).get("dataset_variant", "")) != "versionB":
        raise ValueError(f"ASR reference is not a Version-B dataset: {asr_reference_path}")
    asr_by_id = {
        str(sample["sample_id"]): sample for sample in asr_dataset.get("samples", [])
    }
    max_len = int(dataset["meta"]["max_len"])
    text_dim = int(dataset["meta"]["text_dim"])
    report: list[dict[str, Any]] = []

    for sample in dataset["samples"]:
        sample_id = str(sample["sample_id"])
        official_valid_length = int(sample["valid_length"])
        sample["timeline_source"] = "mfa_official_text"
        sample["text_status"] = "available"
        sample["official_text_gap_filled"] = False
        sample["official_text_missing_word_count"] = 0
        sample["official_word_count"] = int(sample["original_word_count"])
        if bool(sample.get("truncated", False)):
            continue
        asr = asr_by_id.get(sample_id)
        if asr is None or bool(asr.get("truncated", False)):
            continue
        asr_valid_length = int(asr.get("valid_length", 0))
        if not 0 < official_valid_length < asr_valid_length <= max_len:
            continue
        official_words = [str(word) for word in sample["words"][:official_valid_length]]
        asr_words = [str(word) for word in asr["words"][:asr_valid_length]]
        positions = _subsequence_positions(official_words, asr_words)
        if positions is None or len(positions) != official_valid_length:
            continue

        official_text = np.asarray(sample["text"], dtype=np.float32).copy()
        official_timestamps = np.asarray(sample["timestamps"], dtype=np.float32).copy()
        official_padded_words = list(sample["words"])
        remapped_text = np.zeros((max_len, text_dim), dtype=np.float32)
        remapped_text_mask = np.zeros(max_len, dtype=np.uint8)
        for official_index, asr_index in enumerate(positions):
            remapped_text[asr_index] = official_text[official_index]
            remapped_text_mask[asr_index] = 1

        sample["official_words"] = official_padded_words
        sample["official_timestamps"] = official_timestamps
        sample["official_valid_length"] = official_valid_length
        sample["official_text_position_indices"] = np.asarray(positions, dtype=np.int32)
        sample["words"] = list(asr["words"])
        sample["timestamps"] = np.asarray(asr["timestamps"], dtype=np.float32).copy()
        sample["text"] = remapped_text
        sample["text_mask"] = remapped_text_mask
        for field in (
            "audio",
            "vision",
            "vision_22",
            "sequence_mask",
            "audio_mask",
            "vision_mask",
            "audio_frame_counts",
            "vision_frame_counts",
        ):
            sample[field] = np.asarray(asr[field]).copy()
        for field in ("audio_status", "audio_available", "vision_status"):
            sample[field] = asr[field]
        sample["valid_length"] = asr_valid_length
        sample["original_word_count"] = int(asr["original_word_count"])
        sample["truncated"] = bool(asr["truncated"])
        sample["timeline_source"] = "whisperx_official_text_gap_fill"
        sample["alignment_provider"] = "WhisperX ASR word timestamps"
        sample["alignment_status"] = "success_with_official_text_gap_fill"
        sample["text_status"] = "partial"
        sample["official_text_gap_filled"] = True
        sample["official_text_missing_word_count"] = asr_valid_length - official_valid_length
        sample["vision_timestamp_match_before_correction"] = False
        sample["vision_alignment_source"] = "versionB_whisperx_word_windows"
        sample["asr_reference_model"] = asr.get("whisperx_model")
        sample["asr_reference_alignment_model"] = asr.get("whisperx_alignment_model")

        text_valid = int(remapped_text_mask[:asr_valid_length].sum())
        audio_valid = int(np.asarray(sample["audio_mask"][:asr_valid_length]).sum())
        vision_valid = int(np.asarray(sample["vision_mask"][:asr_valid_length]).sum())
        summary_index = summary.index[summary["sample_id"].astype(str) == sample_id]
        if len(summary_index) != 1:
            raise ValueError(f"Summary row not unique for {sample_id}")
        index = summary_index[0]
        updates = {
            "original_word_count": int(sample["original_word_count"]),
            "valid_length": asr_valid_length,
            "truncated": bool(sample["truncated"]),
            "text_valid_windows": text_valid,
            "audio_valid_windows": audio_valid,
            "vision_valid_windows": vision_valid,
            "text_valid_ratio": text_valid / asr_valid_length,
            "audio_valid_ratio": audio_valid / asr_valid_length,
            "vision_valid_ratio": vision_valid / asr_valid_length,
            "audio_status": sample["audio_status"],
            "vision_status": sample["vision_status"],
            "alignment_status": sample["alignment_status"],
            "timeline_source": sample["timeline_source"],
            "official_word_count": official_valid_length,
            "official_text_missing_word_count": asr_valid_length - official_valid_length,
        }
        for key, value in updates.items():
            summary.loc[index, key] = value

        missing_positions = sorted(set(range(asr_valid_length)) - set(positions))
        report.append(
            {
                "sample_id": sample_id,
                "official_word_count": official_valid_length,
                "asr_word_count": asr_valid_length,
                "missing_text_word_count": len(missing_positions),
                "missing_text_positions_1based": [index + 1 for index in missing_positions],
                "missing_asr_words": [asr_words[index] for index in missing_positions],
                "official_to_asr_positions_1based": [index + 1 for index in positions],
            }
        )

    dataset["meta"].update(
        {
            "alignment_reference": (
                "MFA official-text word timestamps; WhisperX timeline only for "
                "strictly proven official-text omissions"
            ),
            "text_mask_semantics": (
                "1=official text observed and encoded; 0=official text missing or sequence padding"
            ),
            "official_text_gap_fill_rule": (
                "all normalized official words must be an exact ordered subsequence of a longer ASR sequence"
            ),
            "official_text_gap_fill_source": str(asr_reference_path),
            "official_text_gap_fill_sample_count": len(report),
            "official_text_gap_fill_word_count": sum(
                int(item["missing_text_word_count"]) for item in report
            ),
        }
    )
    return dataset, summary, report


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _pool_candidate_modalities(
    config: dict[str, Any], sample_id: str, windows: np.ndarray
) -> dict[str, np.ndarray]:
    """Pool cached frame-level modalities over VAD-only text-gap windows."""
    work = Path(config["paths"]["work_dir"])
    stem = safe_sample_name(sample_id)
    settings = config["problem1"]

    mat_path = work / "audio" / f"{stem}.mat"
    wav_path = work / "audio" / f"{stem}.wav"
    if mat_path.is_file():
        audio_frames, _, _ = load_covarep_mat(
            mat_path,
            str(settings["covarep_feature_key"]),
            str(settings["covarep_names_key"]),
            int(settings["covarep_expected_dim"]),
        )
        audio_times = covarep_frame_times(
            wav_path,
            len(audio_frames),
            float(settings["covarep_hop_seconds"]),
            float(settings.get("covarep_first_sample_fraction", 0.5)),
        )
        audio, audio_mask, audio_counts = pool_points_by_word_windows(
            audio_times, audio_frames, windows
        )
    else:
        audio = np.zeros((len(windows), int(settings["covarep_expected_dim"])), dtype=np.float32)
        audio_mask = np.zeros(len(windows), dtype=np.uint8)
        audio_counts = np.zeros(len(windows), dtype=np.int32)

    vision35 = extract_visual35_from_csv(config, sample_id, config["_vision35_audit"])
    vision, vision_mask, vision_counts = pool_points_by_word_windows(
        vision35["times"],
        vision35["features"],
        windows,
        vision35["valid"],
    )
    vision22_frames = _load_npz_arrays(work / "features" / "vision" / f"{stem}.npz")
    vision22, _, _ = pool_points_by_word_windows(
        vision22_frames["times"],
        vision22_frames["features"],
        windows,
        vision22_frames.get("valid"),
    )
    return {
        "audio": audio,
        "audio_mask": audio_mask,
        "audio_frame_counts": audio_counts,
        "vision": vision,
        "vision_mask": vision_mask,
        "vision_frame_counts": vision_counts,
        "vision_22": vision22,
    }


def apply_vad_unrecognized_speech_fill(
    dataset: dict[str, Any],
    summary: Any,
    audit_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Insert one zero-text pseudo-position per accepted uncovered speech span."""
    if not audit_path.is_file():
        raise FileNotFoundError(
            "VAD text-gap filling requires an audit report. Run "
            f"audit_unrecognized_speech.py first: {audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    accepted = audit.get("accepted_candidates", [])
    if not isinstance(accepted, list):
        raise TypeError(f"accepted_candidates must be a list: {audit_path}")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for item in accepted:
        if not bool(item.get("accepted", True)):
            continue
        by_sample.setdefault(str(item["sample_id"]), []).append(dict(item))

    max_len = int(dataset["meta"]["max_len"])
    insertion_report: list[dict[str, Any]] = []
    skipped_report: list[dict[str, Any]] = []
    for sample in dataset["samples"]:
        sample_id = str(sample["sample_id"])
        candidates = sorted(
            by_sample.get(sample_id, []), key=lambda item: (float(item["start"]), float(item["end"]))
        )
        current_length = int(sample["valid_length"])
        sample["vad_text_gap_filled"] = False
        sample["vad_text_missing_position_count"] = 0
        sample["vad_text_missing_intervals"] = []
        base_reasons = [
            "observed" if int(sample["text_mask"][index]) else "official_transcript_omission"
            for index in range(current_length)
        ]
        sample["text_missing_reason"] = base_reasons + ["padding"] * (max_len - current_length)
        sample["text_missing_confidence"] = np.asarray(
            [1.0 if reason == "official_transcript_omission" else 0.0 for reason in base_reasons]
            + [0.0] * (max_len - current_length),
            dtype=np.float32,
        )
        if not candidates:
            continue
        if bool(sample.get("truncated", False)) or current_length + len(candidates) > max_len:
            skipped_report.append(
                {
                    "sample_id": sample_id,
                    "reason": "max_len_would_drop_existing_positions",
                    "current_length": current_length,
                    "candidate_count": len(candidates),
                }
            )
            continue

        windows = np.asarray(
            [[float(item["start"]), float(item["end"])] for item in candidates],
            dtype=np.float64,
        )
        pooled = _pool_candidate_modalities(config, sample_id, windows)
        official_positions = {
            int(position): official_index
            for official_index, position in enumerate(
                np.asarray(
                    sample.get("official_text_position_indices", np.arange(current_length)),
                    dtype=np.int64,
                )
            )
        }
        records: list[dict[str, Any]] = []
        for index in range(current_length):
            timestamp = np.asarray(sample["timestamps"][index], dtype=np.float64)
            records.append(
                {
                    "kind": "existing",
                    "sort_time": float(timestamp[0]),
                    "source_index": index,
                    "official_index": official_positions.get(index),
                    "reason": base_reasons[index],
                    "confidence": float(sample["text_missing_confidence"][index]),
                }
            )
        for candidate_index, candidate in enumerate(candidates):
            records.append(
                {
                    "kind": "vad_missing",
                    "sort_time": float(candidate["start"]),
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "reason": "asr_unrecognized_speech",
                    "confidence": float(candidate.get("confidence", 0.0)),
                }
            )
        records.sort(key=lambda item: (item["sort_time"], item["kind"] == "vad_missing"))
        new_length = len(records)

        words = [""] * max_len
        timestamps = np.zeros((max_len, 2), dtype=np.float32)
        text = np.zeros_like(sample["text"])
        audio = np.zeros_like(sample["audio"])
        vision = np.zeros_like(sample["vision"])
        vision22 = np.zeros_like(sample["vision_22"])
        text_mask = np.zeros(max_len, dtype=np.uint8)
        audio_mask = np.zeros(max_len, dtype=np.uint8)
        vision_mask = np.zeros(max_len, dtype=np.uint8)
        audio_counts = np.zeros(max_len, dtype=np.int32)
        vision_counts = np.zeros(max_len, dtype=np.int32)
        reasons = ["padding"] * max_len
        confidences = np.zeros(max_len, dtype=np.float32)
        remapped_official = np.full(len(official_positions), -1, dtype=np.int32)
        inserted_intervals: list[dict[str, Any]] = []

        for output_index, record in enumerate(records):
            reasons[output_index] = str(record["reason"])
            confidences[output_index] = float(record["confidence"])
            if record["kind"] == "existing":
                source = int(record["source_index"])
                words[output_index] = str(sample["words"][source])
                timestamps[output_index] = sample["timestamps"][source]
                text[output_index] = sample["text"][source]
                audio[output_index] = sample["audio"][source]
                vision[output_index] = sample["vision"][source]
                vision22[output_index] = sample["vision_22"][source]
                text_mask[output_index] = sample["text_mask"][source]
                audio_mask[output_index] = sample["audio_mask"][source]
                vision_mask[output_index] = sample["vision_mask"][source]
                audio_counts[output_index] = sample["audio_frame_counts"][source]
                vision_counts[output_index] = sample["vision_frame_counts"][source]
                official_index = record.get("official_index")
                if official_index is not None and official_index < len(remapped_official):
                    remapped_official[int(official_index)] = output_index
            else:
                candidate_index = int(record["candidate_index"])
                candidate = dict(record["candidate"])
                words[output_index] = "[TEXT_MISSING]"
                timestamps[output_index] = windows[candidate_index]
                audio[output_index] = pooled["audio"][candidate_index]
                vision[output_index] = pooled["vision"][candidate_index]
                vision22[output_index] = pooled["vision_22"][candidate_index]
                audio_mask[output_index] = pooled["audio_mask"][candidate_index]
                vision_mask[output_index] = pooled["vision_mask"][candidate_index]
                audio_counts[output_index] = pooled["audio_frame_counts"][candidate_index]
                vision_counts[output_index] = pooled["vision_frame_counts"][candidate_index]
                candidate["position_1based"] = output_index + 1
                inserted_intervals.append(candidate)
                insertion_report.append({"sample_id": sample_id, **candidate})

        if len(remapped_official) and np.any(remapped_official < 0):
            raise ValueError(f"Lost an official text position while inserting VAD gap: {sample_id}")
        sample.update(
            {
                "words": words,
                "timestamps": timestamps,
                "text": text,
                "audio": audio,
                "vision": vision,
                "vision_22": vision22,
                "sequence_mask": np.r_[
                    np.ones(new_length, dtype=np.uint8),
                    np.zeros(max_len - new_length, dtype=np.uint8),
                ],
                "text_mask": text_mask,
                "audio_mask": audio_mask,
                "vision_mask": vision_mask,
                "audio_frame_counts": audio_counts,
                "vision_frame_counts": vision_counts,
                "valid_length": new_length,
                "original_word_count": new_length,
                "text_missing_reason": reasons,
                "text_missing_confidence": confidences,
                "vad_text_gap_filled": True,
                "vad_text_missing_position_count": len(inserted_intervals),
                "vad_text_missing_intervals": inserted_intervals,
                "text_status": "partial",
                "timeline_source": str(sample.get("timeline_source", "mfa_official_text"))
                + "+vad_unrecognized_speech",
                "alignment_status": "success_with_text_gap_fill",
            }
        )
        sample["official_text_position_indices"] = remapped_official
        sample["official_word_count"] = len(remapped_official)

        summary_index = summary.index[summary["sample_id"].astype(str) == sample_id]
        if len(summary_index) != 1:
            raise ValueError(f"Summary row not unique for {sample_id}")
        index = summary_index[0]
        text_valid = int(text_mask[:new_length].sum())
        summary.loc[index, "original_word_count"] = new_length
        summary.loc[index, "valid_length"] = new_length
        summary.loc[index, "text_valid_windows"] = text_valid
        summary.loc[index, "audio_valid_windows"] = int(audio_mask[:new_length].sum())
        summary.loc[index, "vision_valid_windows"] = int(vision_mask[:new_length].sum())
        summary.loc[index, "text_valid_ratio"] = text_valid / new_length
        summary.loc[index, "audio_valid_ratio"] = int(audio_mask[:new_length].sum()) / new_length
        summary.loc[index, "vision_valid_ratio"] = int(vision_mask[:new_length].sum()) / new_length
        summary.loc[index, "alignment_status"] = sample["alignment_status"]
        summary.loc[index, "timeline_source"] = sample["timeline_source"]
        summary.loc[index, "vad_text_missing_position_count"] = len(inserted_intervals)

    dataset["meta"].update(
        {
            "vad_text_gap_fill_rule": audit.get("method"),
            "vad_text_gap_fill_audit": str(audit_path),
            "vad_text_gap_fill_settings": audit.get("missing_speech_settings", {}),
            "vad_text_gap_fill_sample_count": len(
                {item["sample_id"] for item in insertion_report}
            ),
            "vad_text_gap_fill_position_count": len(insertion_report),
            "vad_text_gap_fill_skipped_sample_count": len(skipped_report),
            "text_missing_reason_semantics": {
                "observed": "official text observed and BERT encoded",
                "official_transcript_omission": "ASR word absent from official text",
                "asr_unrecognized_speech": "voiced VAD span absent from ASR and official text",
                "padding": "outside the real aligned sequence",
            },
        }
    )
    return dataset, summary, insertion_report + [
        {"status": "skipped", **item} for item in skipped_report
    ]


def main() -> None:
    args = parse_args()
    config = configure(args)
    logger = setup_logging(config, verbose=args.verbose)
    set_seed(int(config["project"].get("seed", 2026)))
    source_covarep = project_path(args.covarep_dir)
    output_path = project_path(args.output)
    summary_path = project_path(args.summary)
    report_path = project_path(args.validation_report)
    gap_report_path = project_path(args.gap_report)
    asr_reference_path = project_path(args.asr_reference_pkl)
    vad_audit_path = project_path(args.vad_unrecognized_audit)
    vad_gap_report_path = project_path(args.vad_gap_report)

    logger.info("Version A isolated work directory: %s", config["paths"]["work_dir"])
    logger.info("Version A output: %s", output_path)
    manifest = prepare_manifest(config)
    if len(manifest) != 100 or manifest["sample_id"].nunique() != 100:
        raise ValueError(
            f"Version A requires exactly 100 unique samples; got rows={len(manifest)}, "
            f"unique={manifest['sample_id'].nunique()}"
        )

    copy_covarep_inputs(config, manifest, source_covarep, args.force)
    extract_all_audio(config, manifest, args.force)
    prepare_covarep_frame_cache(config, manifest, args.force)
    run_mfa(config, manifest, args.force, args.download_mfa_models)
    os.environ["OMP_NUM_THREADS"] = str(args.openface_cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.openface_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.openface_cpu_threads)
    logger.info(
        "Parallel extraction: BERT=GPU batches of %d, OpenFace=CPU(%d threads)",
        args.bert_batch_size,
        args.openface_cpu_threads,
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="versionA") as executor:
        text_future = executor.submit(extract_all_text, config, manifest, args.force)
        vision_future = executor.submit(extract_all_visual, config, manifest, args.force)
        text_future.result()
        vision_future.result()
    align_all(config, manifest, args.force)

    dataset, summary = build_problem1_dataset(config)
    gap_report: list[dict[str, Any]] = []
    if not args.disable_official_text_gap_fill:
        dataset, summary, gap_report = apply_official_text_gap_fill(
            dataset, summary, asr_reference_path
        )
        gap_report_path.parent.mkdir(parents=True, exist_ok=True)
        gap_report_path.write_text(
            json.dumps(gap_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Official-text gap fill: samples=%d missing_positions=%d report=%s",
            len(gap_report),
            sum(int(item["missing_text_word_count"]) for item in gap_report),
            gap_report_path,
        )
    vad_gap_report: list[dict[str, Any]] = []
    if not args.disable_vad_text_gap_fill:
        dataset, summary, vad_gap_report = apply_vad_unrecognized_speech_fill(
            dataset, summary, vad_audit_path, config
        )
        vad_gap_report_path.parent.mkdir(parents=True, exist_ok=True)
        vad_gap_report_path.write_text(
            json.dumps(vad_gap_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        inserted_vad_gaps = [
            item for item in vad_gap_report if item.get("status") != "skipped"
        ]
        logger.info(
            "VAD-unrecognized text gap fill: samples=%d positions=%d report=%s",
            len({item["sample_id"] for item in inserted_vad_gaps}),
            len(inserted_vad_gaps),
            vad_gap_report_path,
        )
    dataset["meta"].update(
        {
            "schema_version": "versionA-multisource-text-gap-aware-1.2",
            "dataset_variant": "versionA",
            "data_inputs": (
                "100 MP4 + label-100.xlsx + COVAREP MAT + Version-B WhisperX "
                "timeline audit for official omissions + Pyannote/COVAREP audit "
                "for ASR-unrecognized voiced speech"
            ),
            "isolated_work_dir": str(config["paths"]["work_dir"]),
            "bert_batch_size": int(args.bert_batch_size),
            "openface_cpu_threads": int(args.openface_cpu_threads),
        }
    )
    save_problem1_dataset(
        config,
        dataset,
        summary,
        dataset_path=output_path,
        summary_path=summary_path,
    )
    logger.info("Saved Version A dataset: %s", output_path)

    if not args.skip_figures:
        generate_problem1_figures(config, dataset)
    if not args.skip_validation:
        report = validate(config, output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Version A validation: passed=%d/%d, errors=%d",
            report["passed_sample_count"],
            report["actual_sample_count"],
            report["error_count"],
        )
        if not report["ok"]:
            preview = "\n".join(
                f"{item['sample_id']} | {item['error_type']} | {item['details']}"
                for item in report["errors"][:30]
            )
            raise RuntimeError(f"Version A validation failed:\n{preview}")

    print(f"VERSION_A_OK: {output_path}")
    print(f"SUMMARY: {summary_path}")
    if not args.disable_official_text_gap_fill:
        print(f"TEXT_GAP_REPORT: {gap_report_path}")
    if not args.disable_vad_text_gap_fill:
        print(f"VAD_TEXT_GAP_REPORT: {vad_gap_report_path}")
    if not args.skip_validation:
        print(f"VALIDATION: {report_path}")


if __name__ == "__main__":
    main()
