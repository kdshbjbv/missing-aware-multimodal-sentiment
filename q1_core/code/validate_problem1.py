from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import read_config
from src.prepare_manifest import load_manifest, normalize_clip_id
from src.problem1_dataset import load_mfa_reference, wav_statistics
from src.utils import safe_sample_name
from src.vision_35 import audit_openface_au_fields, vision35_feature_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the final problem-1 aligned feature dataset")
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def _zero(array: np.ndarray, atol: float = 0.0) -> bool:
    return bool(np.all(np.abs(np.asarray(array)) <= atol))


def _norm_word(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower().replace("’", "'"))


def _label_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frame = pd.read_excel(
        config["paths"]["label_file"],
        sheet_name=config["data"].get("label_sheet", "label"),
        dtype={"video_id": str},
    )
    result: dict[str, dict[str, Any]] = {}
    separator = str(config["data"].get("sample_id_separator", "$_$"))
    mapping = config["data"]["classification_mapping"]
    for row in frame.itertuples(index=False):
        video_id = str(row.video_id).strip()
        clip_id = normalize_clip_id(row.clip_id)
        sample_id = f"{video_id}{separator}{clip_id}"
        result[sample_id] = {
            "video_id": video_id,
            "clip_id": clip_id,
            "raw_text": str(row.text),
            "regression_label": float(row.label),
            "annotation": str(row.annotation),
            "classification_label": int(mapping[str(row.annotation)]),
        }
    return result


def validate(config: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    samples = dataset.get("samples", [])
    meta = dataset.get("meta", {})
    manifest = load_manifest(config)
    labels = _label_map(config)
    vision_audit = audit_openface_au_fields(config, labels.keys())
    manifest_by_id = {str(row.sample_id): row for row in manifest.itertuples(index=False)}
    work = Path(config["paths"]["work_dir"])
    max_len = int(config["project"]["max_len"])
    expected_shapes = {
        "text": (max_len, int(config["problem1"]["text_expected_dim"])),
        "audio": (max_len, int(config["problem1"]["covarep_expected_dim"])),
        "vision": (max_len, int(config["problem1"]["vision_expected_dim"])),
        "vision_22": (max_len, int(config["problem1"]["vision_legacy_dim"])),
        "timestamps": (max_len, 2),
    }
    errors: list[dict[str, str]] = []
    sample_error_counts: dict[str, int] = defaultdict(int)

    def add(sample_id: str, error_type: str, details: str) -> None:
        errors.append({"sample_id": sample_id, "error_type": error_type, "details": details})
        sample_error_counts[sample_id] += 1

    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    if len(samples) != len(labels):
        add("__dataset__", "SAMPLE_COUNT", f"PKL={len(samples)}, Excel={len(labels)}")
    duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicates:
        add("__dataset__", "DUPLICATE_SAMPLE_ID", repr(duplicates))
    missing_ids = sorted(set(labels) - set(sample_ids))
    extra_ids = sorted(set(sample_ids) - set(labels))
    if missing_ids or extra_ids:
        add("__dataset__", "SAMPLE_ID_SET", f"missing={missing_ids}, extra={extra_ids}")
    expected_vision_names = list(vision_audit["feature_names"])
    if list(meta.get("vision_feature_names", [])) != expected_vision_names:
        add(
            "__dataset__",
            "VISION_FEATURE_NAMES",
            f"meta={meta.get('vision_feature_names')}, audited={expected_vision_names}",
        )
    if int(meta.get("vision_dim", -1)) != 35:
        add("__dataset__", "VISION_DIM", f"meta vision_dim={meta.get('vision_dim')}")
    if meta.get("vision_definition") != "17 AU intensity + 18 AU presence":
        add("__dataset__", "VISION_DEFINITION", repr(meta.get("vision_definition")))
    if meta.get("vision_extractor") != "OpenFace 2.2" or bool(meta.get("vision_is_facet", True)):
        add("__dataset__", "VISION_PROVENANCE", repr(meta.get("vision_extractor")))

    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if sample_id not in labels or sample_id not in manifest_by_id:
            add(sample_id, "UNKNOWN_SAMPLE_ID", "Not found in label/manifest mapping")
            continue
        label = labels[sample_id]
        row = manifest_by_id[sample_id]
        stem = safe_sample_name(sample_id)
        required_files = {
            "video": Path(str(row.video_path)),
            "MFA JSON": work / "mfa" / "word_timestamps" / f"{stem}.json",
            "BERT NPZ": work / "features" / "text" / f"{stem}.npz",
            "OpenFace NPZ": work / "features" / "vision" / f"{stem}.npz",
            "OpenFace 35-D NPZ": vision35_feature_path(config, sample_id),
            "word-level visual NPZ": work / "aligned" / f"{stem}.npz",
            "WAV": work / "audio" / f"{stem}.wav",
        }
        for name, path in required_files.items():
            if not path.is_file():
                add(sample_id, "MISSING_FILE", f"{name}: {path}")

        for field, shape in expected_shapes.items():
            value = np.asarray(sample.get(field))
            if value.shape != shape:
                add(sample_id, "SHAPE", f"{field}={value.shape}, expected={shape}")
            elif not np.isfinite(value).all():
                add(sample_id, "NONFINITE", field)
        for field in ("sequence_mask", "text_mask", "audio_mask", "vision_mask"):
            value = np.asarray(sample.get(field))
            if value.shape != (max_len,):
                add(sample_id, "SHAPE", f"{field}={value.shape}, expected={(max_len,)}")
        for field in ("audio_frame_counts", "vision_frame_counts"):
            value = np.asarray(sample.get(field))
            if value.shape != (max_len,):
                add(sample_id, "SHAPE", f"{field}={value.shape}, expected={(max_len,)}")

        valid_length = int(sample.get("valid_length", -1))
        original_count = int(sample.get("original_word_count", -1))
        if not 0 <= valid_length <= max_len:
            add(sample_id, "VALID_LENGTH", str(valid_length))
            continue
        if valid_length != min(original_count, max_len):
            add(sample_id, "VALID_LENGTH", f"valid={valid_length}, original={original_count}")
        if bool(sample.get("truncated")) != (original_count > max_len):
            add(sample_id, "TRUNCATION_FLAG", f"original={original_count}")

        sequence = np.asarray(sample["sequence_mask"])
        expected_sequence = np.r_[
            np.ones(valid_length, dtype=np.uint8), np.zeros(max_len - valid_length, dtype=np.uint8)
        ]
        if not np.array_equal(sequence, expected_sequence):
            add(sample_id, "SEQUENCE_MASK", "Real/padding positions are inconsistent")
        text_mask = np.asarray(sample["text_mask"])
        gap_filled = bool(sample.get("official_text_gap_filled", False))
        vad_gap_filled = bool(sample.get("vad_text_gap_filled", False))
        any_text_gap = gap_filled or vad_gap_filled
        if np.any((text_mask > 0) & (sequence == 0)):
            add(sample_id, "TEXT_MASK", "text_mask extends into sequence padding")
        if not any_text_gap and not np.array_equal(text_mask, sequence):
            add(sample_id, "TEXT_MASK", "MFA-only sample requires text_mask == sequence_mask")
        if not _zero(np.asarray(sample["text"])[text_mask == 0]):
            add(sample_id, "TEXT_MISSING_NONZERO", "text_mask=0 but text vector is nonzero")
        reasons = list(sample.get("text_missing_reason", []))
        confidences = np.asarray(sample.get("text_missing_confidence", []), dtype=np.float32)
        if any_text_gap:
            if len(reasons) != max_len:
                add(sample_id, "TEXT_MISSING_REASON", f"length={len(reasons)}, expected={max_len}")
            else:
                allowed_reasons = {
                    "observed",
                    "official_transcript_omission",
                    "asr_unrecognized_speech",
                    "padding",
                }
                if any(reason not in allowed_reasons for reason in reasons):
                    add(sample_id, "TEXT_MISSING_REASON", repr(reasons))
                if any(reason != "padding" for reason in reasons[valid_length:]):
                    add(sample_id, "TEXT_MISSING_REASON", "non-padding reason after valid_length")
                for index in range(valid_length):
                    expected_valid = reasons[index] == "observed"
                    if bool(text_mask[index]) != expected_valid:
                        add(
                            sample_id,
                            "TEXT_MISSING_REASON_MASK",
                            f"index={index}, reason={reasons[index]}, mask={text_mask[index]}",
                        )
            if confidences.shape != (max_len,) or not np.isfinite(confidences).all():
                add(sample_id, "TEXT_MISSING_CONFIDENCE", f"shape={confidences.shape}")

        try:
            mfa_words, mfa_times = load_mfa_reference(config, sample_id)
        except Exception as exc:
            add(sample_id, "MFA_READ", repr(exc))
            continue
        if any_text_gap:
            official_word_count = int(sample.get("official_word_count", -1))
            positions = np.asarray(
                sample.get("official_text_position_indices", []), dtype=np.int64
            )
            if official_word_count != len(mfa_words):
                add(
                    sample_id,
                    "MFA_WORD_COUNT",
                    f"MFA={len(mfa_words)}, official_word_count={official_word_count}",
                )
            if len(positions) != len(mfa_words):
                add(
                    sample_id,
                    "OFFICIAL_POSITION_COUNT",
                    f"positions={len(positions)}, MFA={len(mfa_words)}",
                )
            elif (
                np.any(positions < 0)
                or np.any(positions >= valid_length)
                or (len(positions) > 1 and np.any(np.diff(positions) <= 0))
            ):
                add(sample_id, "OFFICIAL_POSITIONS", repr(positions.tolist()))
            else:
                stored_official = [str(sample["words"][index]) for index in positions]
                if [_norm_word(word) for word in stored_official] != [
                    _norm_word(word) for word in mfa_words
                ]:
                    add(
                        sample_id,
                        "OFFICIAL_ASR_WORD_MISMATCH",
                        f"stored={stored_official}, MFA={mfa_words}",
                    )
                expected_mask = np.zeros(max_len, dtype=np.uint8)
                expected_mask[positions] = 1
                if not np.array_equal(text_mask, expected_mask):
                    add(sample_id, "TEXT_MASK", "text_mask does not encode official positions")
                expected_missing = (
                    sum(reason == "official_transcript_omission" for reason in reasons[:valid_length])
                    if len(reasons) == max_len
                    else valid_length - len(positions)
                )
                if int(sample.get("official_text_missing_word_count", -1)) != expected_missing:
                    add(
                        sample_id,
                        "OFFICIAL_MISSING_COUNT",
                        f"stored={sample.get('official_text_missing_word_count')}, expected={expected_missing}",
                    )
        else:
            if len(mfa_words) != original_count:
                add(sample_id, "MFA_WORD_COUNT", f"MFA={len(mfa_words)}, PKL={original_count}")
            if list(sample["words"][:valid_length]) != mfa_words[:valid_length]:
                add(sample_id, "PKL_MFA_WORD_MISMATCH", "Stored words differ from MFA reference")
        if any(word != "" for word in sample["words"][valid_length:]):
            add(sample_id, "PADDING_WORD", "Padding words must be empty strings")
        if not any_text_gap and not np.allclose(
            sample["timestamps"][:valid_length],
            mfa_times[:valid_length],
            atol=1e-6,
            rtol=0,
        ):
            add(sample_id, "PKL_MFA_TIMESTAMP_MISMATCH", "Stored timestamps differ from MFA JSON")

        bert_path = work / "features" / "text" / f"{stem}.npz"
        if bert_path.is_file():
            with np.load(bert_path, allow_pickle=False) as bert:
                bert_words = [str(value) for value in bert["words"].tolist()]
                bert_features = np.asarray(bert["features"], dtype=np.float32)
            if bert_words != mfa_words:
                mismatch = next(
                    (i for i, pair in enumerate(zip(mfa_words, bert_words)) if pair[0] != pair[1]),
                    min(len(mfa_words), len(bert_words)),
                )
                add(
                    sample_id,
                    "MFA_BERT_WORD_MISMATCH",
                    f"index={mismatch}, MFA={mfa_words[mismatch:mismatch+2]}, BERT={bert_words[mismatch:mismatch+2]}",
                )
            if any_text_gap:
                positions = np.asarray(
                    sample.get("official_text_position_indices", []), dtype=np.int64
                )
                if len(positions) == len(bert_features) and not np.allclose(
                    np.asarray(sample["text"])[positions], bert_features, atol=1e-6, rtol=0
                ):
                    add(
                        sample_id,
                        "OFFICIAL_BERT_REMAP",
                        "Mapped text vectors differ from the official-text BERT cache",
                    )
        if not bool(sample.get("mfa_bert_match")):
            add(sample_id, "MFA_BERT_MATCH_FLAG", "False")

        real_times = np.asarray(sample["timestamps"][:valid_length])
        if valid_length:
            if np.any(real_times[:, 1] < real_times[:, 0]):
                add(sample_id, "TIMESTAMP_ORDER", "One or more end times precede start times")
            starts_nonmonotonic = np.any(np.diff(real_times[:, 0]) < -1e-6)
            ends_nonmonotonic = np.any(np.diff(real_times[:, 1]) < -1e-6)
            if starts_nonmonotonic or (ends_nonmonotonic and not vad_gap_filled):
                add(sample_id, "TIMESTAMP_MONOTONICITY", "Word timestamps are not monotonic")
            if float(real_times.max()) > float(sample["duration"]) + 0.10:
                add(
                    sample_id,
                    "TIMESTAMP_DURATION",
                    f"max={float(real_times.max()):.6f}, duration={float(sample['duration']):.6f}",
                )

        if not _zero(sample["timestamps"][valid_length:]):
            add(sample_id, "PADDING_TIMESTAMP", "Padding timestamps are not [0,0]")
        for field in ("text", "audio", "vision", "vision_22"):
            if not _zero(np.asarray(sample[field])[valid_length:]):
                add(sample_id, "PADDING_FEATURE", f"{field} padding is not zero")
        for field in ("text_mask", "audio_mask", "vision_mask", "audio_frame_counts", "vision_frame_counts"):
            if not _zero(np.asarray(sample[field])[valid_length:]):
                add(sample_id, "PADDING_AUXILIARY", f"{field} padding is not zero")

        audio_mask = np.asarray(sample["audio_mask"])
        vision_mask = np.asarray(sample["vision_mask"])
        audio_counts = np.asarray(sample["audio_frame_counts"])
        vision_counts = np.asarray(sample["vision_frame_counts"])
        if not np.array_equal(audio_mask, (audio_counts > 0).astype(np.uint8)):
            add(sample_id, "AUDIO_MASK_COUNT", "audio_mask != (audio_frame_counts > 0)")
        if not np.array_equal(vision_mask, (vision_counts > 0).astype(np.uint8)):
            add(sample_id, "VISION_MASK_COUNT", "vision_mask != (vision_frame_counts > 0)")
        if not _zero(np.asarray(sample["audio"])[audio_mask == 0]):
            add(sample_id, "AUDIO_MISSING_NONZERO", "audio_mask=0 but audio vector is nonzero")
        if not _zero(np.asarray(sample["vision"])[vision_mask == 0]):
            add(sample_id, "VISION_MISSING_NONZERO", "vision_mask=0 but vision vector is nonzero")
        presence = np.asarray(sample["vision"], dtype=np.float32)[:valid_length, 17:]
        if np.any(presence < -1e-6) or np.any(presence > 1.0 + 1e-6):
            add(sample_id, "VISION_PRESENCE_RANGE", "Mean AU presence is outside [0,1]")
        vision35_path = vision35_feature_path(config, sample_id)
        if vision35_path.is_file():
            with np.load(vision35_path, allow_pickle=False) as vision35:
                cached_names = [str(value) for value in vision35["feature_names"].tolist()]
                if cached_names != expected_vision_names:
                    add(sample_id, "VISION_FEATURE_NAMES", repr(cached_names))
                if vision35["features"].ndim != 2 or vision35["features"].shape[1] != 35:
                    add(sample_id, "VISION_FRAME_SHAPE", str(vision35["features"].shape))
                elif not np.isfinite(vision35["features"]).all():
                    add(sample_id, "NONFINITE", "OpenFace 35-D frame cache")

        mat_path = work / "audio" / f"{stem}.mat"
        if sample["audio_status"] == "covarep":
            if not mat_path.is_file():
                add(sample_id, "MISSING_COVAREP_MAT", str(mat_path))
        elif sample["audio_status"] == "silent":
            if mat_path.exists():
                add(sample_id, "SILENT_HAS_MAT", str(mat_path))
            wav_path = work / "audio" / f"{stem}.wav"
            if wav_path.is_file() and float(wav_statistics(wav_path)["max_abs"]) > float(
                config["problem1"].get("silent_max_abs_threshold", 1.0e-12)
            ):
                add(sample_id, "SILENCE_CHECK", "WAV is not digital silence")
        elif sample["audio_status"] == "covarep_missing":
            add(sample_id, "COVAREP_MISSING", "Non-silent WAV has no MAT")
        else:
            add(sample_id, "AUDIO_STATUS", str(sample["audio_status"]))

        for field in ("video_id", "clip_id", "raw_text", "annotation", "classification_label"):
            if sample[field] != label[field]:
                add(sample_id, "LABEL_MAPPING", f"{field}: PKL={sample[field]!r}, Excel={label[field]!r}")
        if not np.isclose(float(sample["regression_label"]), float(label["regression_label"])):
            add(
                sample_id,
                "LABEL_MAPPING",
                f"regression_label: PKL={sample['regression_label']}, Excel={label['regression_label']}",
            )

    per_sample_errors = {sample_id: count for sample_id, count in sample_error_counts.items() if sample_id != "__dataset__"}
    passed_ids = [sample_id for sample_id in sample_ids if sample_id not in per_sample_errors]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "expected_sample_count": len(labels),
        "actual_sample_count": len(samples),
        "passed_sample_count": len(passed_ids),
        "failed_sample_count": len(set(sample_ids) - set(passed_ids)),
        "dataset_level_error_count": sample_error_counts.get("__dataset__", 0),
        "error_count": len(errors),
        "ok": len(errors) == 0,
        "meta": meta,
        "errors": errors,
    }
    return report


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    output = Path(config["paths"]["output_dir"])
    dataset_path = Path(args.dataset) if args.dataset else output / str(config["problem1"]["dataset_file"])
    report_path = Path(args.report) if args.report else output / str(config["problem1"]["validation_report"])
    report = validate(config, dataset_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "expected_sample_count", "actual_sample_count", "passed_sample_count",
        "failed_sample_count", "error_count", "ok"
    )}, ensure_ascii=False, indent=2))
    print(f"Validation report: {report_path}")
    if not report["ok"]:
        for error in report["errors"][:30]:
            print(f"{error['sample_id']} | {error['error_type']} | {error['details']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
