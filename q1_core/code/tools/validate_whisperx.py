#!/usr/bin/env python3
"""Validate WhisperX ASR/alignment JSON outputs against their source WAVs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import safe_sample_name  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def wav_stats(path: Path) -> tuple[float, float, float]:
    info = sf.info(path)
    values, _ = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float(info.duration), 0.0, 0.0
    return (
        float(info.duration),
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.max(np.abs(values))),
    )


def validate_sample(
    sample_id: str,
    wav_path: Path,
    json_path: Path,
    silent_threshold: float,
    duration_tolerance: float,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    duration, rms, max_abs = wav_stats(wav_path)
    expected_silent = rms <= silent_threshold and max_abs <= silent_threshold
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "wav": str(wav_path),
        "json": str(json_path),
        "duration": duration,
        "rms": rms,
        "max_abs": max_abs,
        "expected_silent": expected_silent,
        "errors": errors,
        "warnings": warnings,
    }
    if not json_path.is_file():
        errors.append("result JSON is missing")
        record.update({"status": "missing", "word_count": None, "valid": False})
        return record
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as error:
        errors.append(f"cannot parse JSON: {type(error).__name__}: {error}")
        record.update({"status": "invalid_json", "word_count": None, "valid": False})
        return record

    if payload.get("sample_id") != sample_id:
        errors.append(f"sample_id mismatch: JSON={payload.get('sample_id')!r}")
    status = payload.get("status")
    text = payload.get("text")
    words = payload.get("words")
    if not isinstance(text, str):
        errors.append("text is not a string")
    if not isinstance(words, list):
        errors.append("words is not a list")
        words = []
    if expected_silent:
        if status != "silent":
            errors.append(f"digital-silence WAV must have status='silent', got {status!r}")
        if words:
            errors.append("digital-silence WAV must have words == []")
        if text not in ("", None):
            errors.append("digital-silence WAV must have an empty transcript")
    elif status == "silent":
        errors.append("non-silent WAV was marked silent")
    elif status == "no_speech":
        if words:
            errors.append("status='no_speech' requires words == []")
        if text not in ("", None):
            errors.append("status='no_speech' requires an empty transcript")
    elif status != "success":
        errors.append(f"non-silent sample has unsupported status: {status!r}")

    previous_start = -math.inf
    aligned_count = 0
    unaligned_count = 0
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            errors.append(f"words[{index}] is not an object")
            continue
        start_raw, end_raw = word.get("start"), word.get("end")
        if start_raw is None and end_raw is None:
            unaligned_count += 1
            continue
        start, end = finite_float(start_raw), finite_float(end_raw)
        if start is None or end is None:
            errors.append(f"words[{index}] has only one valid timestamp")
            continue
        aligned_count += 1
        if start < 0:
            errors.append(f"words[{index}].start is negative: {start}")
        if end < start:
            errors.append(f"words[{index}].end < start: {start}, {end}")
        if start + 1.0e-8 < previous_start:
            errors.append(f"words[{index}] is out of chronological order")
        previous_start = start
        if end > duration + duration_tolerance:
            errors.append(
                f"words[{index}].end={end:.6f} exceeds duration={duration:.6f} "
                f"by more than tolerance={duration_tolerance:.6f}"
            )
    declared_unaligned = payload.get("unaligned_word_count")
    if declared_unaligned is not None and int(declared_unaligned) != unaligned_count:
        errors.append(
            f"unaligned_word_count mismatch: JSON={declared_unaligned}, observed={unaligned_count}"
        )
    record.update(
        {
            "status": status,
            "word_count": len(words),
            "aligned_word_count": aligned_count,
            "unaligned_word_count": unaligned_count,
            "valid": not errors,
        }
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="work/audio")
    parser.add_argument("--json-dir", default="work/asr/whisperx")
    parser.add_argument("--manifest", default="work/manifest.csv")
    parser.add_argument("--output", default="outputs/whisperx_validation_report.json")
    parser.add_argument("--silent-threshold", type=float, default=1.0e-12)
    parser.add_argument("--duration-tolerance", type=float, default=0.10)
    parser.add_argument("--allow-partial", action="store_true", help="Validate only JSONs that currently exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = resolve_path(args.audio_dir)
    json_dir = resolve_path(args.json_dir)
    manifest_path = resolve_path(args.manifest)
    output_path = resolve_path(args.output)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    if "sample_id" not in manifest.columns:
        raise ValueError("Manifest is missing sample_id")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values")

    records: list[dict[str, Any]] = []
    seen_result_ids: dict[str, str] = {}
    duplicate_result_ids: list[dict[str, str]] = []
    for result_path in sorted(json_dir.glob("*.json")):
        try:
            result_id = str(json.loads(result_path.read_text(encoding="utf-8")).get("sample_id"))
        except Exception:
            continue
        if result_id in seen_result_ids:
            duplicate_result_ids.append(
                {"sample_id": result_id, "first": seen_result_ids[result_id], "second": str(result_path)}
            )
        else:
            seen_result_ids[result_id] = str(result_path)

    for sample_id in manifest["sample_id"].astype(str):
        safe_name = safe_sample_name(sample_id)
        wav_path = audio_dir / f"{safe_name}.wav"
        json_path = json_dir / f"{safe_name}.json"
        if args.allow_partial and not json_path.is_file():
            continue
        if not wav_path.is_file():
            records.append(
                {
                    "sample_id": sample_id,
                    "wav": str(wav_path),
                    "json": str(json_path),
                    "status": "missing_wav",
                    "errors": ["source WAV is missing"],
                    "warnings": [],
                    "valid": False,
                }
            )
            continue
        records.append(
            validate_sample(
                sample_id,
                wav_path,
                json_path,
                args.silent_threshold,
                args.duration_tolerance,
            )
        )

    error_count = sum(len(record.get("errors", [])) for record in records) + len(duplicate_result_ids)
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_dir": str(audio_dir),
        "json_dir": str(json_dir),
        "manifest": str(manifest_path),
        "expected_sample_count": int(len(manifest)),
        "validated_sample_count": len(records),
        "valid_sample_count": sum(bool(record.get("valid")) for record in records),
        "invalid_sample_count": sum(not bool(record.get("valid")) for record in records),
        "status_counts": status_counts,
        "duplicate_result_ids": duplicate_result_ids,
        "error_count": error_count,
        "overall_pass": error_count == 0 and (args.allow_partial or len(records) == len(manifest)),
        "settings": {
            "silent_threshold": args.silent_threshold,
            "duration_tolerance": args.duration_tolerance,
            "allow_partial": args.allow_partial,
        },
        "samples": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"WhisperX validation: pass={payload['overall_pass']} "
        f"valid={payload['valid_sample_count']}/{payload['validated_sample_count']} "
        f"errors={payload['error_count']}"
    )
    print(f"Report: {output_path}")
    raise SystemExit(0 if payload["overall_pass"] else 1)


if __name__ == "__main__":
    main()
