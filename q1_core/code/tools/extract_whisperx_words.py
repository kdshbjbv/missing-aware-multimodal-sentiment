#!/usr/bin/env python3
"""WhisperX ASR and word-level alignment as an isolated validation branch.

This script intentionally does not write to the existing MFA, BERT, aligned
feature, or final PKL outputs.  Its only project outputs are:

* work/asr/whisperx/*.json
* work/asr/whisperx_txt/*.txt
* outputs/whisperx_asr_summary.csv
* outputs/whisperx_vs_reference.csv
* outputs/whisperx_vs_mfa.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, safe_sample_name  # noqa: E402


LOGGER = logging.getLogger("whisperx_words")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "feature_config.yaml"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": str, "video_id": str, "clip_id": str})
    required = {"sample_id", "raw_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values in manifest: {duplicates}")
    return frame


def manifest_name_map(manifest: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sample_id in manifest["sample_id"].astype(str):
        safe_name = safe_sample_name(sample_id)
        existing = mapping.get(safe_name)
        if existing is not None and existing != sample_id:
            raise ValueError(
                f"Ambiguous safe filename {safe_name!r}: {existing!r} and {sample_id!r}"
            )
        mapping[safe_name] = sample_id
    return mapping


def sample_id_for_wav(wav_path: Path, mapping: dict[str, str]) -> str:
    if wav_path.stem in mapping:
        return mapping[wav_path.stem]
    # This is only a guarded fallback for a single externally supplied WAV.
    # Project WAVs must resolve through the manifest mapping above.
    if "__" in wav_path.stem:
        video_id, clip_id = wav_path.stem.rsplit("__", 1)
        candidate = f"{video_id}$_${clip_id}"
        if safe_sample_name(candidate) == wav_path.stem:
            LOGGER.warning(
                "WAV %s is not listed in the manifest; recovered sample_id=%s using "
                "the project's safe-name rule",
                wav_path,
                candidate,
            )
            return candidate
    raise KeyError(f"Cannot map WAV filename to a project sample_id: {wav_path.name}")


def audio_statistics(path: Path) -> tuple[float, float, float]:
    info = sf.info(path)
    samples, _ = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return float(info.duration), 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(values))))
    max_abs = float(np.max(np.abs(values)))
    return float(info.duration), rms, max_abs


def normalize_tokens(text_or_words: str | Iterable[str]) -> list[str]:
    if isinstance(text_or_words, str):
        raw_tokens = text_or_words.split()
    else:
        raw_tokens = [str(item) for item in text_or_words]
    normalized: list[str] = []
    for token in raw_tokens:
        token = unicodedata.normalize("NFKC", token).lower()
        token = re.sub(r"[^a-z0-9]+", "", token)
        if token:
            normalized.append(token)
    return normalized


def normalized_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def fallback_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    previous = list(range(len(hyp) + 1))
    for ref_index, ref_word in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            substitution = previous[hyp_index - 1] + (ref_word != hyp_word)
            deletion = previous[hyp_index] + 1
            insertion = current[hyp_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return float(previous[-1] / len(ref))


def calculate_wer(reference: str, hypothesis: str) -> float:
    reference_normalized = normalized_text(reference)
    hypothesis_normalized = normalized_text(hypothesis)
    try:
        from jiwer import wer

        if not reference_normalized:
            return 0.0 if not hypothesis_normalized else float(len(hypothesis_normalized.split()))
        return float(wer(reference_normalized, hypothesis_normalized))
    except ImportError:
        return fallback_wer(reference, hypothesis)


def ensure_nltk_alignment_data(cache_dir: Path) -> None:
    """Keep WhisperX sentence-tokenizer data inside the project cache."""
    nltk_dir = cache_dir / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nltk_dir)
    import nltk

    nltk_path = str(nltk_dir)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        LOGGER.info("Downloading NLTK punkt_tab into %s", nltk_dir)
        downloaded = nltk.download("punkt_tab", download_dir=str(nltk_dir), quiet=False)
        if not downloaded:
            raise RuntimeError("NLTK punkt_tab download failed")
        nltk.data.find("tokenizers/punkt_tab/english/")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_word_segments(raw_words: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    words: list[dict[str, Any]] = []
    unaligned: list[dict[str, Any]] = []
    previous_start = -math.inf
    for raw in raw_words:
        word = str(raw.get("word", ""))
        start = finite_float(raw.get("start"))
        end = finite_float(raw.get("end"))
        score = finite_float(raw.get("score"))
        valid = start is not None and end is not None and start >= 0 and end >= start
        if valid and start + 1.0e-8 < previous_start:
            valid = False
        if valid:
            previous_start = start
        else:
            start = None
            end = None
        item = {"word": word, "start": start, "end": end, "score": score}
        words.append(item)
        if not valid:
            unaligned.append(dict(item))
    return words, unaligned


def transcript_from_segments(segments: Sequence[dict[str, Any]]) -> str:
    pieces = [str(segment.get("text", "")).strip() for segment in segments]
    return " ".join(piece for piece in pieces if piece).strip()


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


@dataclass
class RuntimeChoice:
    requested_model: str
    model_name: str
    device: str
    compute_type: str
    fallback_reason: str | None = None


class WhisperXRunner:
    def __init__(self, args: argparse.Namespace, cache_dir: Path):
        self.args = args
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whisperx: Any = None
        self.asr_model: Any = None
        self.align_model: Any = None
        self.align_metadata: dict[str, Any] | None = None
        self.alignment_model_name: str | None = args.align_model
        self.choice = self._initial_choice()

    def _initial_choice(self) -> RuntimeChoice:
        import torch

        if self.args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.args.device
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU int8")
            device = "cpu"
        compute_type = self.args.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        if device == "cpu" and compute_type == "float16":
            LOGGER.warning("float16 is not suitable for CPU inference; using int8")
            compute_type = "int8"
        return RuntimeChoice(
            requested_model=self.args.model,
            model_name=self.args.model,
            device=device,
            compute_type=compute_type,
        )

    def _import_whisperx(self) -> Any:
        if self.whisperx is None:
            import whisperx

            self.whisperx = whisperx
        return self.whisperx

    def _load_asr_once(self, model_name: str, compute_type: str) -> Any:
        whisperx = self._import_whisperx()
        LOGGER.info(
            "Loading WhisperX ASR model=%s device=%s compute_type=%s vad=%s",
            model_name,
            self.choice.device,
            compute_type,
            self.args.vad_method,
        )
        return whisperx.load_model(
            model_name,
            self.choice.device,
            device_index=self.args.device_index,
            compute_type=compute_type,
            language=self.args.language,
            vad_method=self.args.vad_method,
            download_root=str(self.cache_dir / "asr"),
            local_files_only=self.args.local_files_only,
            use_auth_token=self.args.hf_token,
            threads=self.args.threads,
        )

    def ensure_asr_model(self) -> None:
        if self.asr_model is not None:
            return
        try:
            self.asr_model = self._load_asr_once(self.choice.model_name, self.choice.compute_type)
            return
        except Exception as error:
            if self.choice.device == "cuda" and self.choice.compute_type == "float16":
                LOGGER.warning("float16 model load failed (%s); retrying with int8", error)
                self.choice.compute_type = "int8"
                self.choice.fallback_reason = f"float16 load failed: {error}"
                try:
                    self.asr_model = self._load_asr_once(self.choice.model_name, "int8")
                    return
                except Exception as int8_error:
                    if is_cuda_oom(int8_error) and self.choice.model_name != self.args.fallback_model:
                        self._switch_to_fallback(f"int8 ASR model load OOM: {int8_error}")
                        return
                    raise
            if is_cuda_oom(error) and self.choice.model_name != self.args.fallback_model:
                self._switch_to_fallback(f"ASR model load OOM: {error}")
                return
            raise

    def _release_asr(self) -> None:
        self.asr_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _switch_to_fallback(self, reason: str) -> None:
        if self.choice.model_name == self.args.fallback_model:
            raise RuntimeError(f"Fallback model {self.args.fallback_model} also failed: {reason}")
        LOGGER.warning(
            "Switching ASR model from %s to %s because of OOM: %s",
            self.choice.model_name,
            self.args.fallback_model,
            reason,
        )
        self._release_asr()
        self.choice.model_name = self.args.fallback_model
        self.choice.fallback_reason = reason
        self.asr_model = self._load_asr_once(self.choice.model_name, self.choice.compute_type)

    def ensure_align_model(self, language: str) -> None:
        if self.align_model is not None:
            if self.align_metadata and self.align_metadata.get("language") != language:
                raise ValueError(
                    f"Loaded alignment language {self.align_metadata.get('language')} != {language}"
                )
            return
        whisperx = self._import_whisperx()
        if self.alignment_model_name is None:
            import whisperx.alignment as alignment

            self.alignment_model_name = (
                alignment.DEFAULT_ALIGN_MODELS_TORCH.get(language)
                or alignment.DEFAULT_ALIGN_MODELS_HF.get(language)
            )
        LOGGER.info(
            "Loading alignment model=%s language=%s device=%s",
            self.alignment_model_name,
            language,
            self.choice.device,
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.choice.device,
            model_name=self.args.align_model,
            model_dir=str(self.cache_dir / "alignment"),
            model_cache_only=self.args.local_files_only,
        )

    def transcribe_and_align(self, wav_path: Path) -> dict[str, Any]:
        self.ensure_asr_model()
        whisperx = self._import_whisperx()
        audio = whisperx.load_audio(str(wav_path))
        try:
            transcription = self.asr_model.transcribe(
                audio,
                batch_size=self.args.batch_size,
                language=self.args.language,
                print_progress=self.args.print_progress,
            )
        except Exception as error:
            if is_cuda_oom(error) and self.choice.model_name != self.args.fallback_model:
                self._switch_to_fallback(f"transcription OOM for {wav_path.name}: {error}")
                transcription = self.asr_model.transcribe(
                    audio,
                    batch_size=max(1, min(self.args.batch_size, 4)),
                    language=self.args.language,
                    print_progress=self.args.print_progress,
                )
            else:
                raise
        language = str(transcription.get("language") or self.args.language)
        self.ensure_align_model(language)
        aligned = whisperx.align(
            transcription.get("segments", []),
            self.align_model,
            self.align_metadata,
            audio,
            self.choice.device,
            interpolate_method="nearest",
            return_char_alignments=False,
            print_progress=self.args.print_progress,
        )
        return dict(aligned)


def read_existing_complete(path: Path, requested_model: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("status") == "silent":
        return payload
    if payload.get("status") == "success" and payload.get("requested_model", requested_model) == requested_model:
        return payload
    return None


def process_one(
    wav_path: Path,
    sample_id: str,
    output_dir: Path,
    text_dir: Path,
    runner: WhisperXRunner,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = output_dir / f"{safe_sample_name(sample_id)}.json"
    text_path = text_dir / f"{safe_sample_name(sample_id)}.txt"
    if not args.overwrite:
        existing = read_existing_complete(output_path, args.model)
        if existing is not None:
            LOGGER.info("Resume: %s (%s)", sample_id, existing.get("status"))
            if not text_path.exists():
                atomic_write_text(text_path, str(existing.get("text", "")) + "\n")
            return existing

    duration, rms, max_abs = audio_statistics(wav_path)
    common: dict[str, Any] = {
        "sample_id": sample_id,
        "source_wav": relative_or_absolute(wav_path),
        "requested_model": args.model,
        "language": args.language,
        "duration": duration,
        "rms": rms,
        "max_abs": max_abs,
    }
    if rms <= args.silent_threshold and max_abs <= args.silent_threshold:
        payload = {
            **common,
            "model": None,
            "alignment_model": None,
            "device": None,
            "compute_type": None,
            "text": "",
            "segments": [],
            "words": [],
            "unaligned_words": [],
            "word_count": 0,
            "aligned_word_count": 0,
            "unaligned_word_count": 0,
            "status": "silent",
        }
        atomic_write_json(output_path, payload)
        atomic_write_text(text_path, "")
        LOGGER.info("Silent: %s rms=%.3g max_abs=%.3g", sample_id, rms, max_abs)
        return payload

    try:
        result = runner.transcribe_and_align(wav_path)
        segments = json_safe(result.get("segments", []))
        raw_words = result.get("word_segments", [])
        if not raw_words:
            raw_words = [
                word
                for segment in result.get("segments", [])
                for word in segment.get("words", [])
            ]
        words, unaligned = clean_word_segments(raw_words)
        text = transcript_from_segments(result.get("segments", []))
        payload = {
            **common,
            "model": runner.choice.model_name,
            "alignment_model": runner.alignment_model_name,
            "device": runner.choice.device,
            "compute_type": runner.choice.compute_type,
            "fallback_reason": runner.choice.fallback_reason,
            "language": str(result.get("language") or args.language),
            "text": text,
            "segments": segments,
            "words": words,
            "unaligned_words": unaligned,
            "word_count": len(words),
            "aligned_word_count": len(words) - len(unaligned),
            "unaligned_word_count": len(unaligned),
            "status": "success" if (text or words) else "no_speech",
        }
    except Exception as error:
        LOGGER.exception("Failed: %s", sample_id)
        payload = {
            **common,
            "model": runner.choice.model_name,
            "alignment_model": runner.alignment_model_name,
            "device": runner.choice.device,
            "compute_type": runner.choice.compute_type,
            "fallback_reason": runner.choice.fallback_reason,
            "text": "",
            "segments": [],
            "words": [],
            "unaligned_words": [],
            "word_count": 0,
            "aligned_word_count": 0,
            "unaligned_word_count": 0,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_write_json(output_path, payload)
    atomic_write_text(text_path, str(payload.get("text", "")) + "\n")
    return payload


def load_results(output_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sample_id = str(payload["sample_id"])
        except Exception as error:
            LOGGER.warning("Ignoring unreadable result %s: %s", path, error)
            continue
        if sample_id in results:
            raise ValueError(f"Duplicate WhisperX result sample_id: {sample_id}")
        results[sample_id] = payload
    return results


def create_summary_tables(
    output_dir: Path,
    manifest: pd.DataFrame,
    mfa_dir: Path,
    summary_path: Path,
    reference_path: Path,
    mfa_path: Path,
) -> None:
    results = load_results(output_dir)
    reference_map = manifest.set_index("sample_id")["raw_text"].fillna("").astype(str).to_dict()
    ordered_ids = [sample_id for sample_id in manifest["sample_id"].astype(str) if sample_id in results]
    extra_ids = sorted(set(results).difference(ordered_ids))

    summary_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    mfa_rows: list[dict[str, Any]] = []

    for sample_id in ordered_ids + extra_ids:
        result = results[sample_id]
        words = result.get("words", []) if isinstance(result.get("words"), list) else []
        aligned_count = sum(
            finite_float(word.get("start")) is not None and finite_float(word.get("end")) is not None
            for word in words
            if isinstance(word, dict)
        )
        transcript = str(result.get("text", ""))
        reference = reference_map.get(sample_id, "")
        summary_rows.append(
            {
                "sample_id": sample_id,
                "duration": finite_float(result.get("duration")),
                "status": result.get("status"),
                "language": result.get("language"),
                "model": result.get("model"),
                "alignment_model": result.get("alignment_model"),
                "device": result.get("device"),
                "compute_type": result.get("compute_type"),
                "word_count": len(words),
                "aligned_word_count": aligned_count,
                "unaligned_word_count": len(words) - aligned_count,
                "transcript": transcript,
                "reference_text": reference,
                "rms": finite_float(result.get("rms")),
                "max_abs": finite_float(result.get("max_abs")),
            }
        )
        reference_rows.append(
            {
                "sample_id": sample_id,
                "reference_text": reference,
                "asr_text": transcript,
                "WER": calculate_wer(reference, transcript),
                "reference_word_count": len(normalize_tokens(reference)),
                "asr_word_count": len(normalize_tokens(transcript)),
                "status": result.get("status"),
            }
        )

        mfa_json = mfa_dir / f"{safe_sample_name(sample_id)}.json"
        mfa_words: list[dict[str, Any]] = []
        if mfa_json.is_file():
            try:
                mfa_words = json.loads(mfa_json.read_text(encoding="utf-8")).get("words", [])
            except Exception as error:
                LOGGER.warning("Cannot read MFA JSON %s: %s", mfa_json, error)
        whisper_tokens = normalize_tokens(word.get("word", "") for word in words if isinstance(word, dict))
        mfa_tokens = normalize_tokens(word.get("word", "") for word in mfa_words if isinstance(word, dict))
        sequence_match = bool(mfa_words) and whisper_tokens == mfa_tokens
        start_differences: list[float] = []
        end_differences: list[float] = []
        if sequence_match:
            for whisper_word, mfa_word in zip(words, mfa_words):
                whisper_start = finite_float(whisper_word.get("start"))
                whisper_end = finite_float(whisper_word.get("end"))
                mfa_start = finite_float(mfa_word.get("start"))
                mfa_end = finite_float(mfa_word.get("end"))
                if None not in (whisper_start, whisper_end, mfa_start, mfa_end):
                    start_differences.append(abs(whisper_start - mfa_start))
                    end_differences.append(abs(whisper_end - mfa_end))
        mfa_rows.append(
            {
                "sample_id": sample_id,
                "status": result.get("status"),
                "whisperx_word_count": len(words),
                "mfa_word_count": len(mfa_words),
                "word_sequence_match": sequence_match,
                "timestamp_comparison_available": sequence_match and len(start_differences) == len(words),
                "compared_word_count": len(start_differences),
                "mean_start_abs_diff": float(np.mean(start_differences)) if start_differences else None,
                "mean_end_abs_diff": float(np.mean(end_differences)) if end_differences else None,
                "max_start_abs_diff": float(np.max(start_differences)) if start_differences else None,
                "max_end_abs_diff": float(np.max(end_differences)) if end_differences else None,
            }
        )

    for path in (summary_path, reference_path, mfa_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_rows).to_csv(reference_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(mfa_rows).to_csv(mfa_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote summary rows=%d to %s", len(summary_rows), summary_path)


def discover_wavs(args: argparse.Namespace, audio_dir: Path) -> list[Path]:
    if args.audio:
        wavs = [resolve_path(args.audio)]
    else:
        wavs = sorted(audio_dir.glob("*.wav"))
    missing = [str(path) for path in wavs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio files not found: {missing}")
    return wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", help="Directory containing project WAV files")
    source.add_argument("--audio", help="Single WAV file")
    parser.add_argument("--output-dir", default="work/asr/whisperx")
    parser.add_argument("--txt-output-dir", default="work/asr/whisperx_txt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", default="work/manifest.csv")
    parser.add_argument("--mfa-dir", default="work/mfa/word_timestamps")
    parser.add_argument("--summary-path", default="outputs/whisperx_asr_summary.csv")
    parser.add_argument("--reference-comparison", default="outputs/whisperx_vs_reference.csv")
    parser.add_argument("--mfa-comparison", default="outputs/whisperx_vs_mfa.csv")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--fallback-model", default="medium.en")
    parser.add_argument("--align-model", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", choices=("auto", "float16", "float32", "int8"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vad-method", choices=("pyannote", "silero"), default="pyannote")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--silent-threshold", type=float, default=None)
    parser.add_argument("--sample-id", action="append", default=[], help="Process only this sample_id; repeatable")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(resolve_path(args.config))
    audio_dir = resolve_path(args.audio_dir or "work/audio")
    output_dir = resolve_path(args.output_dir)
    text_dir = resolve_path(args.txt_output_dir)
    manifest_path = resolve_path(args.manifest)
    mfa_dir = resolve_path(args.mfa_dir)
    summary_path = resolve_path(args.summary_path)
    reference_path = resolve_path(args.reference_comparison)
    mfa_path = resolve_path(args.mfa_comparison)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    if args.silent_threshold is None:
        args.silent_threshold = float(
            config.get("problem1", {}).get("silent_max_abs_threshold", 1.0e-12)
        )
    if args.cache_dir:
        cache_dir = resolve_path(args.cache_dir)
    else:
        cache_dir = Path(config["paths"]["huggingface_home"]) / "whisperx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["NLTK_DATA"] = str(cache_dir / "nltk_data")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    # WhisperX alignment invokes NLTK sentence splitting after acoustic
    # alignment.  Download it before loading multi-GB models so a network
    # failure is reported early and all cache writes stay in the project.
    ensure_nltk_alignment_data(cache_dir)

    manifest = load_manifest(manifest_path)
    name_map = manifest_name_map(manifest)
    wavs = discover_wavs(args, audio_dir)
    items = [(wav, sample_id_for_wav(wav, name_map)) for wav in wavs]
    if args.sample_id:
        requested = set(args.sample_id)
        items = [(wav, sample_id) for wav, sample_id in items if sample_id in requested]
        missing_ids = sorted(requested.difference(sample_id for _, sample_id in items))
        if missing_ids:
            raise KeyError(f"Requested sample_id values have no selected WAV: {missing_ids}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No WAV files selected")

    runner = WhisperXRunner(args, cache_dir)
    status_counts: dict[str, int] = {}
    for index, (wav_path, sample_id) in enumerate(items, start=1):
        LOGGER.info("[%d/%d] %s <- %s", index, len(items), sample_id, wav_path.name)
        payload = process_one(wav_path, sample_id, output_dir, text_dir, runner, args)
        status = str(payload.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    create_summary_tables(
        output_dir,
        manifest,
        mfa_dir,
        summary_path,
        reference_path,
        mfa_path,
    )
    LOGGER.info("Selected run complete: %s", status_counts)


if __name__ == "__main__":
    main()
