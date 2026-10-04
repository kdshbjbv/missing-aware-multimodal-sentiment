from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VADSettings:
    onset: float = 0.50
    offset: float = 0.363
    min_speech_seconds: float = 0.10
    min_silence_seconds: float = 0.10


@dataclass(frozen=True)
class MissingSpeechSettings:
    word_padding_seconds: float = 0.10
    min_gap_seconds: float = 0.35
    min_mean_vad_score: float = 0.70
    min_voiced_fraction: float = 0.50


def normalize_word(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower().replace("��", "'"))


def lcs_word_matches(
    official_words: Sequence[str], asr_words: Sequence[str]
) -> list[tuple[int, int]]:
    """Return deterministic exact-token LCS matches as (official, ASR) indices."""
    official = [normalize_word(word) for word in official_words]
    asr = [normalize_word(word) for word in asr_words]
    rows = len(official) + 1
    columns = len(asr) + 1
    lengths = np.zeros((rows, columns), dtype=np.int16)
    for i in range(len(official) - 1, -1, -1):
        for j in range(len(asr) - 1, -1, -1):
            if official[i] and official[i] == asr[j]:
                lengths[i, j] = lengths[i + 1, j + 1] + 1
            else:
                lengths[i, j] = max(lengths[i + 1, j], lengths[i, j + 1])
    matches: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(official) and j < len(asr):
        if official[i] and official[i] == asr[j]:
            matches.append((i, j))
            i += 1
            j += 1
        elif lengths[i + 1, j] >= lengths[i, j + 1]:
            i += 1
        else:
            j += 1
    return matches


def merge_intervals(
    intervals: Iterable[Sequence[float]], *, collar: float = 0.0
) -> list[tuple[float, float]]:
    cleaned = sorted(
        (float(item[0]), float(item[1]))
        for item in intervals
        if len(item) >= 2
        and math.isfinite(float(item[0]))
        and math.isfinite(float(item[1]))
        and float(item[1]) > float(item[0])
    )
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + collar:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def subtract_intervals(
    sources: Iterable[Sequence[float]], covers: Iterable[Sequence[float]]
) -> list[tuple[float, float]]:
    """Subtract the union of covers from the union of source intervals."""
    source_union = merge_intervals(sources)
    cover_union = merge_intervals(covers)
    output: list[tuple[float, float]] = []
    for source_start, source_end in source_union:
        cursor = source_start
        for cover_start, cover_end in cover_union:
            if cover_end <= cursor:
                continue
            if cover_start >= source_end:
                break
            if cover_start > cursor:
                output.append((cursor, min(cover_start, source_end)))
            cursor = max(cursor, cover_end)
            if cursor >= source_end:
                break
        if cursor < source_end:
            output.append((cursor, source_end))
    return [(start, end) for start, end in output if end > start]


def interval_overlap(interval: Sequence[float], other: Sequence[float]) -> float:
    return max(0.0, min(float(interval[1]), float(other[1])) - max(float(interval[0]), float(other[0])))


def _score_slice(
    score_times: np.ndarray, scores: np.ndarray, start: float, end: float
) -> np.ndarray:
    selected = (score_times >= start) & (score_times <= end)
    return scores[selected]


def find_unrecognized_speech_candidates(
    *,
    vad_payload: dict[str, Any],
    asr_words: Sequence[dict[str, Any]],
    official_words: Sequence[str],
    official_timestamps: np.ndarray,
    covarep_times: np.ndarray,
    covarep_vuv: np.ndarray,
    settings: MissingSpeechSettings,
) -> list[dict[str, Any]]:
    """Find voiced VAD regions not covered by ASR or official-only words.

    ASR word windows are padded before subtraction because aligned word
    timestamps generally cover the lexical core rather than every acoustic
    frame. Official words absent from the ASR LCS are treated as known text and
    therefore suppress a candidate that substantially overlaps their MFA span.
    """
    speech_intervals = [
        (float(item["start"]), float(item["end"]))
        for item in vad_payload.get("speech_segments", [])
    ]
    asr_intervals: list[tuple[float, float]] = []
    valid_asr_words: list[str] = []
    duration = float(vad_payload.get("duration", 0.0))
    for item in asr_words:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        asr_intervals.append(
            (
                max(0.0, start - settings.word_padding_seconds),
                min(duration, end + settings.word_padding_seconds),
            )
        )
        valid_asr_words.append(str(item.get("word", "")))

    matches = lcs_word_matches(official_words, valid_asr_words)
    matched_official = {official_index for official_index, _ in matches}
    official_only_intervals = [
        tuple(map(float, official_timestamps[index]))
        for index in range(min(len(official_words), len(official_timestamps)))
        if index not in matched_official and official_timestamps[index, 1] > official_timestamps[index, 0]
    ]

    score_times = np.asarray(vad_payload.get("score_times", []), dtype=np.float64)
    scores = np.asarray(vad_payload.get("speech_scores", []), dtype=np.float64)
    # Known official words that Whisper missed are not text-missing positions.
    # Remove their MFA spans from the uncovered VAD regions before thresholding.
    gaps = subtract_intervals(
        subtract_intervals(speech_intervals, asr_intervals),
        official_only_intervals,
    )
    candidates: list[dict[str, Any]] = []
    for start, end in gaps:
        gap_duration = end - start
        if gap_duration + 1.0e-9 < settings.min_gap_seconds:
            continue
        gap_scores = _score_slice(score_times, scores, start, end)
        mean_vad_score = float(gap_scores.mean()) if len(gap_scores) else 0.0
        vuv_selected = (covarep_times >= start) & (covarep_times < end)
        voiced_fraction = (
            float(np.mean(covarep_vuv[vuv_selected] >= 0.5)) if np.any(vuv_selected) else 0.0
        )
        accepted = (
            mean_vad_score >= settings.min_mean_vad_score
            and voiced_fraction >= settings.min_voiced_fraction
        )
        duration_component = np.clip(
            (gap_duration - settings.min_gap_seconds) / max(0.35, settings.min_gap_seconds),
            0.0,
            1.0,
        )
        vad_component = np.clip(
            (mean_vad_score - settings.min_mean_vad_score)
            / max(1.0e-6, 1.0 - settings.min_mean_vad_score),
            0.0,
            1.0,
        )
        voiced_component = np.clip(
            (voiced_fraction - settings.min_voiced_fraction)
            / max(1.0e-6, 1.0 - settings.min_voiced_fraction),
            0.0,
            1.0,
        )
        confidence = float(0.25 * duration_component + 0.45 * vad_component + 0.30 * voiced_component)
        candidates.append(
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(gap_duration, 6),
                "mean_vad_score": round(mean_vad_score, 6),
                "voiced_fraction": round(voiced_fraction, 6),
                "official_only_overlap_fraction": 0.0,
                "confidence": round(confidence, 6),
                "accepted": bool(accepted),
            }
        )
    return candidates


class PyannoteVADExtractor:
    """Cache raw Pyannote VAD scores and binarized speech intervals."""

    def __init__(self, device: str, settings: VADSettings) -> None:
        import torch
        from whisperx.vads.pyannote import Binarize, Pyannote

        self.device = device
        self.settings = settings
        self._binarize_type = Binarize
        self.model = Pyannote(
            torch.device(device),
            vad_onset=settings.onset,
            vad_offset=settings.offset,
            chunk_size=30,
        )

    def extract(self, wav_path: Path) -> dict[str, Any]:
        import soundfile as sf
        import whisperx

        audio = whisperx.load_audio(str(wav_path))
        waveform = self.model.preprocess_audio(audio)
        raw_scores = self.model({"waveform": waveform, "sample_rate": 16000})
        binarizer = self._binarize_type(
            onset=self.settings.onset,
            offset=self.settings.offset,
            min_duration_on=self.settings.min_speech_seconds,
            min_duration_off=self.settings.min_silence_seconds,
        )
        active = binarizer(raw_scores)
        segments = [
            {"start": round(float(segment.start), 6), "end": round(float(segment.end), 6)}
            for segment in active.get_timeline()
            if segment.end > segment.start
        ]
        data = np.asarray(raw_scores.data, dtype=np.float64)
        speech_scores = data.max(axis=1) if data.ndim == 2 else data.reshape(-1)
        score_times = np.asarray(
            [raw_scores.sliding_window[index].middle for index in range(len(speech_scores))],
            dtype=np.float64,
        )
        return {
            "source_wav": str(wav_path),
            "duration": float(sf.info(wav_path).duration),
            "device": self.device,
            "vad_settings": asdict(self.settings),
            "speech_segments": segments,
            "score_times": np.round(score_times, 6).tolist(),
            "speech_scores": np.round(speech_scores, 7).tolist(),
        }


def load_or_extract_vad(
    extractor: PyannoteVADExtractor,
    wav_path: Path,
    cache_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    expected_settings = asdict(extractor.settings)
    if cache_path.is_file() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("vad_settings") == expected_settings:
                return cached
        except Exception:
            pass
    payload = extractor.extract(wav_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return payload
