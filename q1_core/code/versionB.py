"""Build Version B from videos and precomputed COVAREP MAT files only.

Version B discovers the 100 videos directly (no Excel input), extracts WAV and
OpenFace features, obtains ASR words and word timestamps with WhisperX, creates
BERT features from those ASR words, and pools COVAREP/OpenFace frames over the
WhisperX word windows.  All generated artifacts are isolated below
``work/versionB`` and the final dataset is explicitly marked as unlabeled.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Set the allocator policy before anything can import torch.  It reduces CUDA
# fragmentation when the WhisperX models are released and BERT is loaded.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

from src.config import read_config
from src.extract_audio import extract_one_audio
from src.extract_visual_openface import extract_one_visual
from src.prepare_manifest import probe_video
from src.utils import (
    ensure_directories,
    get_logger,
    resolve_executable,
    safe_sample_name,
    sample_id,
    set_seed,
    setup_logging,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--dataset-root", default=None, help="Root containing video_id/clip_id.mp4")
    parser.add_argument(
        "--covarep-dir",
        default="work/audio",
        help="Directory containing <safe_sample_name>.mat COVAREP files",
    )
    parser.add_argument("--work-dir", default="work/versionB")
    parser.add_argument("--output", default="outputs/versionB.pkl")
    parser.add_argument("--summary", default="outputs/versionB_feature_summary.csv")
    parser.add_argument("--validation-report", default="outputs/versionB_validation_report.json")
    parser.add_argument("--manifest-output", default="outputs/versionB_manifest.csv")
    parser.add_argument("--asr-summary", default="outputs/versionB_whisperx_summary.csv")
    parser.add_argument("--whisperx-model", default="large-v3")
    parser.add_argument("--whisperx-fallback-model", default="medium.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--compute-type",
        choices=("float16", "float32", "int8", "auto"),
        default="float16",
    )
    parser.add_argument(
        "--whisperx-batch-size",
        type=int,
        default=8,
        help="ASR batch size. 8 is the safe default for large-v3 on the remote RTX 3090",
    )
    parser.add_argument("--whisperx-threads", type=int, default=8)
    parser.add_argument(
        "--openface-cpu-threads",
        type=int,
        default=8,
        help="CPU threads exposed to the CPU-only OpenFace/OpenBLAS process",
    )
    parser.add_argument("--vad-method", choices=("pyannote", "silero"), default="pyannote")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate every derived artifact")
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
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(project_path(args.dataset_root))
    # Shared directories contain model weights only.  Sample-derived A/B data
    # remain isolated because work_dir is replaced above.
    config["output"]["log_file"] = "versionB_processing.log"
    ensure_directories(config)
    return config


def build_unlabeled_manifest(
    config: dict[str, Any], manifest_output: Path
) -> pd.DataFrame:
    """Discover all videos without reading a label Excel file."""
    root = Path(config["paths"]["dataset_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {root}")
    video_paths = sorted(root.rglob("*.mp4"), key=lambda path: path.as_posix())
    if not video_paths:
        raise ValueError(f"No MP4 files found in {root}")
    ffprobe = resolve_executable(config["paths"]["ffprobe"])
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to build the Version B manifest")
    separator = str(config["data"].get("sample_id_separator", "$_$"))
    records: list[dict[str, Any]] = []
    for feature_index, video_path in enumerate(video_paths):
        video_id = video_path.parent.name
        clip_id = video_path.stem
        sid = sample_id(video_id, clip_id, separator)
        records.append(
            {
                "feature_index": feature_index,
                "sample_id": sid,
                "video_id": video_id,
                "clip_id": clip_id,
                "video_path": str(video_path.resolve()),
                "video_relpath": str(video_path.relative_to(root)),
                # These placeholders are only for compatibility with the
                # shared builder. They are replaced by None in the final PKL.
                "raw_text": "",
                "regression_label": 0.0,
                "classification_label": -1,
                "annotation": "Unlabeled",
                "labels_available": False,
                "source_size_bytes": video_path.stat().st_size,
                "source_sha256": "",
                **probe_video(video_path, ffprobe),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs discovered from video paths: {duplicates}")
    work_manifest = Path(config["paths"]["work_dir"]) / config["output"]["manifest_file"]
    for destination in (work_manifest, manifest_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
    get_logger().info(
        "Discovered and audited %d unlabeled videos: %s",
        len(frame),
        work_manifest,
    )
    return frame


def extract_all_audio(config: dict[str, Any], manifest: pd.DataFrame, force: bool) -> None:
    logger = get_logger()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        extract_one_audio(config, row, force=force)
        logger.info("Audio [%d/%d] %s", index, len(manifest), row["sample_id"])


def extract_all_visual(config: dict[str, Any], manifest: pd.DataFrame, force: bool) -> None:
    logger = get_logger()
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        extract_one_visual(config, row, force=force)
        logger.info("OpenFace [%d/%d] %s", index, len(manifest), row["sample_id"])


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sanitize_whisperx_words(path: Path) -> dict[str, Any]:
    """Keep only words with real intervals; preserve dropped items for audit."""
    from tools.extract_whisperx_words import atomic_write_json

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_words = payload.get("words", []) if isinstance(payload.get("words"), list) else []
    valid: list[dict[str, Any]] = []
    dropped = list(payload.get("unaligned_words", []))
    for item in raw_words:
        if not isinstance(item, dict):
            dropped.append({"value": repr(item), "reason": "not_an_object"})
            continue
        word = str(item.get("word", "")).strip()
        start = finite_number(item.get("start"))
        end = finite_number(item.get("end"))
        if not word or start is None or end is None or start < 0 or end <= start:
            if item not in dropped:
                dropped.append(dict(item))
            continue
        valid.append({"word": word, "start": start, "end": end, "score": item.get("score")})
    payload["words"] = valid
    payload["word_count"] = len(valid)
    payload["aligned_word_count"] = len(valid)
    payload["unaligned_words"] = dropped
    payload["unaligned_word_count"] = len(dropped)
    payload["dropped_unaligned_word_count"] = len(dropped)
    if payload.get("status") == "success" and not valid:
        payload["status"] = "no_aligned_words"
    atomic_write_json(path, payload)
    return payload


def cached_whisperx_result(path: Path, requested_model: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    status = str(payload.get("status", ""))
    if status in {"silent", "no_speech", "no_aligned_words"}:
        return payload
    if status == "success" and str(payload.get("requested_model", requested_model)) == requested_model:
        return payload
    return None


def configure_gpu(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.whisperx_batch_size < 1:
        raise ValueError("--whisperx-batch-size must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Version B requested CUDA, but torch.cuda.is_available() is false")
    selected = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda":
        count = torch.cuda.device_count()
        if args.device_index < 0 or args.device_index >= count:
            raise ValueError(f"CUDA device index {args.device_index} is invalid; device_count={count}")
        props = torch.cuda.get_device_properties(args.device_index)
        torch.cuda.set_device(args.device_index)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        info = {
            "device": "cuda",
            "device_index": args.device_index,
            "gpu_name": props.name,
            "gpu_memory_mib": round(props.total_memory / 1024**2),
            "compute_type": args.compute_type,
            "batch_size": args.whisperx_batch_size,
        }
    else:
        if args.compute_type == "float16":
            raise ValueError("float16 cannot be used with CPU; choose --compute-type int8")
        info = {
            "device": "cpu",
            "device_index": None,
            "gpu_name": None,
            "gpu_memory_mib": 0,
            "compute_type": args.compute_type,
            "batch_size": args.whisperx_batch_size,
        }
    get_logger().info("Version B compute runtime: %s", info)
    return info


def release_gpu_models(runner: Any) -> None:
    try:
        runner._release_asr()  # WhisperX runner's explicit ASR cleanup.
    except Exception:
        runner.asr_model = None
    runner.align_model = None
    runner.align_metadata = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_whisperx(
    config: dict[str, Any],
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from tools.extract_whisperx_words import (
        WhisperXRunner,
        ensure_nltk_alignment_data,
        process_one,
    )

    logger = get_logger()
    work = Path(config["paths"]["work_dir"])
    audio_dir = work / "audio"
    output_dir = work / "asr" / "whisperx"
    text_dir = work / "asr" / "whisperx_txt"
    cache_dir = Path(config["paths"]["huggingface_home"]) / "whisperx"
    for directory in (output_dir, text_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["NLTK_DATA"] = str(cache_dir / "nltk_data")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    ensure_nltk_alignment_data(cache_dir)

    runtime_args = argparse.Namespace(
        model=args.whisperx_model,
        fallback_model=args.whisperx_fallback_model,
        align_model=None,
        language=args.language,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        batch_size=args.whisperx_batch_size,
        threads=args.whisperx_threads,
        vad_method=args.vad_method,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        print_progress=False,
        silent_threshold=float(config["problem1"].get("silent_max_abs_threshold", 1.0e-12)),
        overwrite=args.force,
    )
    runner = WhisperXRunner(runtime_args, cache_dir)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for index, (_, row) in enumerate(manifest.iterrows(), start=1):
            sid = str(row["sample_id"])
            wav_path = audio_dir / f"{safe_sample_name(sid)}.wav"
            json_path = output_dir / f"{safe_sample_name(sid)}.json"
            payload = None if args.force else cached_whisperx_result(json_path, args.whisperx_model)
            if payload is None:
                logger.info("WhisperX [%d/%d] processing %s", index, len(manifest), sid)
                payload = process_one(
                    wav_path, sid, output_dir, text_dir, runner, runtime_args
                )
            else:
                logger.info("WhisperX [%d/%d] cache hit %s", index, len(manifest), sid)
            payload = sanitize_whisperx_words(json_path)
            status = str(payload.get("status", "unknown"))
            if status == "error":
                failures.append(f"{sid}: {payload.get('error', 'unknown error')}")
            records.append(
                {
                    "sample_id": sid,
                    "status": status,
                    "text": str(payload.get("text", "")),
                    "aligned_word_count": int(payload.get("aligned_word_count", 0)),
                    "dropped_unaligned_word_count": int(
                        payload.get("dropped_unaligned_word_count", 0)
                    ),
                    "model": payload.get("model"),
                    "alignment_model": payload.get("alignment_model"),
                    "device": payload.get("device"),
                    "compute_type": payload.get("compute_type"),
                    "fallback_reason": payload.get("fallback_reason"),
                    "duration": payload.get("duration"),
                    "rms": payload.get("rms"),
                    "max_abs": payload.get("max_abs"),
                }
            )
    finally:
        release_gpu_models(runner)
    if failures:
        raise RuntimeError("WhisperX failed for one or more samples:\n" + "\n".join(failures))
    return pd.DataFrame.from_records(records)


def modality_status_counts(dataset: dict[str, Any], modality: str) -> dict[str, int]:
    counts = {"no_timeline": 0, "none": 0, "partial": 0, "all": 0}
    for sample in dataset["samples"]:
        sequence = np.asarray(sample["sequence_mask"]).astype(bool)
        mask = np.asarray(sample[f"{modality}_mask"]).astype(bool)
        total = int(sequence.sum())
        present = int((sequence & mask).sum())
        if total == 0:
            counts["no_timeline"] += 1
        elif present == 0:
            counts["none"] += 1
        elif present < total:
            counts["partial"] += 1
        else:
            counts["all"] += 1
    return counts


def main() -> None:
    args = parse_args()
    config = configure(args)
    logger = setup_logging(config, verbose=args.verbose)
    set_seed(int(config["project"].get("seed", 2026)))
    source_covarep = project_path(args.covarep_dir)
    if not source_covarep.is_dir():
        raise FileNotFoundError(f"COVAREP directory does not exist: {source_covarep}")

    output_path = project_path(args.output)
    summary_path = project_path(args.summary)
    report_path = project_path(args.validation_report)
    manifest_output = project_path(args.manifest_output)
    asr_summary_path = project_path(args.asr_summary)
    logger.info("Version B isolated work directory: %s", config["paths"]["work_dir"])
    logger.info("Version B output: %s", output_path)

    manifest = build_unlabeled_manifest(config, manifest_output)
    if args.openface_cpu_threads < 1:
        raise ValueError("--openface-cpu-threads must be at least 1")

    # Import these helpers only after PYTORCH_CUDA_ALLOC_CONF has been set.
    from versionA import copy_covarep_inputs, prepare_covarep_frame_cache

    copy_covarep_inputs(config, manifest, source_covarep, args.force)
    extract_all_audio(config, manifest, args.force)
    prepare_covarep_frame_cache(config, manifest, args.force)

    gpu_info = configure_gpu(args)
    # OpenFace 2.2 FeatureExtraction has no CUDA backend in this build.  Run
    # its genuine AU/pose/gaze extraction on CPU while WhisperX occupies the
    # RTX GPU.  This shortens wall time without changing the visual definition.
    os.environ["OMP_NUM_THREADS"] = str(args.openface_cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.openface_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.openface_cpu_threads)
    logger.info(
        "Parallel extraction: OpenFace=CPU(%d threads), WhisperX=%s",
        args.openface_cpu_threads,
        gpu_info,
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="versionB") as executor:
        vision_future = executor.submit(extract_all_visual, config, manifest, args.force)
        whisperx_future = executor.submit(run_whisperx, config, manifest, args)
        asr_summary = whisperx_future.result()
        vision_future.result()
    asr_summary_path.parent.mkdir(parents=True, exist_ok=True)
    asr_summary.to_csv(asr_summary_path, index=False, encoding="utf-8-sig")

    from tools.build_whisperx_aligned_pkl import (
        atomic_pickle,
        atomic_text,
        build_dataset,
        validate_dataset,
    )

    work = Path(config["paths"]["work_dir"])
    whisperx_dir = work / "asr" / "whisperx"
    text_cache_dir = work / "features" / "text_whisperx"
    dataset, summary = build_dataset(
        config, whisperx_dir, text_cache_dir, sample_ids=None, force_text=args.force
    )

    # No label file is an intentional Version B input constraint.  Never let
    # compatibility placeholders escape as invented ground-truth labels.
    dropped_total = 0
    for sample in dataset["samples"]:
        sid = str(sample["sample_id"])
        result = json.loads(
            (whisperx_dir / f"{safe_sample_name(sid)}.json").read_text(encoding="utf-8")
        )
        dropped = int(result.get("dropped_unaligned_word_count", 0))
        dropped_total += dropped
        sample["raw_text"] = ""
        sample["regression_label"] = None
        sample["classification_label"] = None
        sample["annotation"] = None
        sample["labels_available"] = False
        sample["dropped_unaligned_word_count"] = dropped
    for column in ("regression_label", "classification_label", "annotation"):
        if column in summary:
            summary[column] = None
    summary["labels_available"] = False

    dataset["meta"].update(
        {
            "schema_version": "versionB-whisperx-unlabeled-1.0",
            "dataset_variant": "versionB",
            "data_inputs": (f"{len(manifest)} MP4 + COVAREP MAT; "
                "no Excel or human transcript"),
            "labels_available": False,
            "label_source": "not provided",
            "official_text_role": "not provided and not used",
            "isolated_work_dir": str(work),
            "gpu_runtime": gpu_info,
            "dropped_unaligned_whisperx_words": dropped_total,
        }
    )
    dataset["meta"].pop("classification_mapping", None)

    expected_ids = manifest.sort_values("feature_index")["sample_id"].astype(str).tolist()
    report = validate_dataset(dataset, expected_ids)
    report.update(
        {
            "labels_available": False,
            "gpu_runtime": gpu_info,
            "text_status_counts": {
                str(key): int(value)
                for key, value in asr_summary["status"].value_counts().to_dict().items()
            },
            "audio_sample_status_counts": modality_status_counts(dataset, "audio"),
            "vision_sample_status_counts": modality_status_counts(dataset, "vision"),
            "dropped_unaligned_whisperx_words": dropped_total,
        }
    )
    if not report["overall_pass"]:
        preview = "\n".join(report["errors"][:30])
        raise RuntimeError(f"Version B validation failed:\n{preview}")

    atomic_pickle(output_path, dataset)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    logger.info(
        "Version B validation: passed=%d/%d, errors=%d",
        report["passed_samples"],
        report["validated_samples"],
        len(report["errors"]),
    )
    print(f"VERSION_B_OK: {output_path}")
    print(f"SUMMARY: {summary_path}")
    print(f"ASR_SUMMARY: {asr_summary_path}")
    print(f"VALIDATION: {report_path}")


if __name__ == "__main__":
    main()
