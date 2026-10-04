from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .extract_audio import audio_output_path
from .utils import get_logger, resolve_executable, run_command, safe_sample_name


SILENCE_MARKS = {"", "sil", "sp", "spn", "<eps>", "<unk>"}


def normalize_transcript(text: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", str(text))
    return " ".join(words).lower()


def timestamp_json_path(config: dict[str, Any], sample_id: str) -> Path:
    return Path(config["paths"]["work_dir"]) / "mfa" / "word_timestamps" / f"{safe_sample_name(sample_id)}.json"


def prepare_mfa_sample(config: dict[str, Any], row: pd.Series, force: bool = False) -> tuple[Path, Path]:
    corpus_root = Path(config["paths"]["work_dir"]) / "mfa" / "corpus"
    speaker_dir = corpus_root / safe_sample_name(str(row["video_id"]))
    speaker_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_sample_name(str(row["sample_id"]))
    source_wav = audio_output_path(config, str(row["sample_id"]))
    if not source_wav.exists():
        raise FileNotFoundError(f"Audio has not been extracted: {source_wav}")
    destination_wav = speaker_dir / f"{stem}.wav"
    transcript_path = speaker_dir / f"{stem}.lab"
    if force or not destination_wav.exists():
        shutil.copy2(source_wav, destination_wav)
    normalized = normalize_transcript(str(row["raw_text"]))
    if not normalized:
        raise ValueError(f"Transcript is empty after normalization: {row['sample_id']}")
    transcript_path.write_text(normalized + "\n", encoding="utf-8")
    return destination_wav, transcript_path


def prepare_mfa_corpus(config: dict[str, Any], manifest: pd.DataFrame, force: bool = False) -> None:
    logger = get_logger()
    for _, row in manifest.iterrows():
        prepare_mfa_sample(config, row, force=force)
    logger.info("Prepared MFA corpus with %d utterances", len(manifest))


def download_mfa_models(config: dict[str, Any]) -> None:
    mfa = resolve_executable(config["paths"]["mfa"])
    if mfa is None:
        raise RuntimeError("MFA executable was not found")
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(config["paths"]["mfa_root_dir"])
    for model_type, name in (
        ("dictionary", config["alignment"]["mfa_dictionary"]),
        ("acoustic", config["alignment"]["mfa_acoustic_model"]),
    ):
        result = run_command(
            [mfa, "model", "download", model_type, str(name)], check=False, env=environment
        )
        if result.returncode != 0 and "already exists" not in (result.stderr or "").lower():
            raise RuntimeError(f"Failed to download MFA {model_type} model {name}: {result.stderr}")


def run_mfa_alignment(config: dict[str, Any], clean: bool = False) -> Path:
    logger = get_logger()
    mfa = resolve_executable(config["paths"]["mfa"])
    if mfa is None:
        raise RuntimeError(
            "MFA executable was not found. Install Montreal Forced Aligner in temenv, "
            "then download the configured acoustic and dictionary models."
        )
    work = Path(config["paths"]["work_dir"])
    corpus = work / "mfa" / "corpus"
    output = work / "mfa" / "aligned"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        mfa,
        "align",
        str(corpus),
        str(config["alignment"]["mfa_dictionary"]),
        str(config["alignment"]["mfa_acoustic_model"]),
        str(output),
        "--num_jobs",
        str(config["alignment"].get("mfa_num_jobs", 4)),
        "--temporary_directory",
        str(config["paths"]["mfa_temp_dir"]),
        "--output_format",
        "long_textgrid",
    ]
    if clean:
        command.append("--clean")
    if config["alignment"].get("mfa_use_threading", False):
        # The project lives on a distributed filesystem; MFA multiprocessing
        # can race while cleaning shared model files there. Threading keeps the
        # same parallel job count without cross-process model teardown.
        command.append("--use_threading")
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(config["paths"]["mfa_root_dir"])
    result = run_command(command, check=False, capture=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"MFA alignment failed with exit code {result.returncode}")
    logger.info("MFA alignment completed: %s", output)
    return output


def parse_textgrid(path: Path) -> list[dict[str, Any]]:
    try:
        import textgrid
    except ImportError as exc:
        raise RuntimeError("The textgrid package is required to parse MFA output") from exc
    grid = textgrid.TextGrid.fromFile(str(path))
    tier = None
    for candidate in grid.tiers:
        name = str(getattr(candidate, "name", "")).lower()
        if name in {"words", "word"} or "word" in name:
            tier = candidate
            break
    if tier is None:
        raise ValueError(f"No word tier found in {path}")
    words: list[dict[str, Any]] = []
    for interval in tier:
        mark = str(getattr(interval, "mark", "")).strip()
        if mark.lower() in SILENCE_MARKS:
            continue
        start = float(interval.minTime)
        end = float(interval.maxTime)
        if end <= start:
            continue
        words.append({"word": mark, "start": start, "end": end})
    if not words:
        raise ValueError(f"No aligned words found in {path}")
    return words


def retry_mfa_sample_isolated(config: dict[str, Any], row: pd.Series) -> Path | None:
    """Retry one missing utterance in its own MFA corpus.

    MFA can occasionally omit an utterance from an otherwise successful batch
    alignment.  Isolating the utterance also isolates speaker adaptation and is
    considerably cheaper than rerunning the entire 100-sample corpus.
    """
    logger = get_logger()
    mfa = resolve_executable(config["paths"]["mfa"])
    if mfa is None:
        raise RuntimeError("MFA executable was not found")
    sid = str(row["sample_id"])
    stem = safe_sample_name(sid)
    work = Path(config["paths"]["work_dir"])
    retry_root = work / "mfa" / "retry" / stem
    if retry_root.exists():
        shutil.rmtree(retry_root)
    corpus = retry_root / "corpus" / "speaker"
    aligned = retry_root / "aligned"
    temporary = retry_root / "temp"
    corpus.mkdir(parents=True, exist_ok=True)
    aligned.mkdir(parents=True, exist_ok=True)
    source_wav = audio_output_path(config, sid)
    if not source_wav.exists():
        raise FileNotFoundError(f"Audio has not been extracted: {source_wav}")
    shutil.copy2(source_wav, corpus / f"{stem}.wav")
    normalized = normalize_transcript(str(row["raw_text"]))
    if not normalized:
        raise ValueError(f"Transcript is empty after normalization: {sid}")
    (corpus / f"{stem}.lab").write_text(normalized + "\n", encoding="utf-8")
    command = [
        mfa,
        "align",
        str(retry_root / "corpus"),
        str(config["alignment"]["mfa_dictionary"]),
        str(config["alignment"]["mfa_acoustic_model"]),
        str(aligned),
        "--num_jobs",
        "1",
        "--temporary_directory",
        str(temporary),
        "--output_format",
        "long_textgrid",
        "--beam",
        str(config["alignment"].get("mfa_retry_beam", 100)),
        "--retry_beam",
        str(config["alignment"].get("mfa_retry_retry_beam", 1000)),
        "--clean",
        "--use_threading",
    ]
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(config["paths"]["mfa_root_dir"])
    logger.warning("Retrying missing MFA alignment in isolation: %s", sid)
    result = run_command(command, check=False, capture=False, env=environment)
    if result.returncode != 0:
        logger.error("Isolated MFA retry failed for %s with exit code %d", sid, result.returncode)
        return None
    matches = list(aligned.rglob(f"{stem}.TextGrid")) + list(aligned.rglob(f"{stem}.textgrid"))
    if not matches:
        logger.error("Isolated MFA retry still produced no TextGrid for %s", sid)
        return None
    preserved_dir = work / "mfa" / "aligned" / "_isolated_retry"
    preserved_dir.mkdir(parents=True, exist_ok=True)
    preserved = preserved_dir / f"{stem}.TextGrid"
    shutil.copy2(matches[0], preserved)
    logger.info("Isolated MFA retry succeeded: %s", sid)
    return preserved


def collect_mfa_timestamps(config: dict[str, Any], manifest: pd.DataFrame) -> dict[str, str]:
    logger = get_logger()
    aligned_root = Path(config["paths"]["work_dir"]) / "mfa" / "aligned"
    output_dir = Path(config["paths"]["work_dir"]) / "mfa" / "word_timestamps"
    output_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        stem = safe_sample_name(sid)
        matches = list(aligned_root.rglob(f"{stem}.TextGrid")) + list(aligned_root.rglob(f"{stem}.textgrid"))
        if not matches and bool(config["alignment"].get("mfa_retry_missing_isolated", True)):
            try:
                retry_path = retry_mfa_sample_isolated(config, row)
                if retry_path is not None:
                    matches = [retry_path]
            except Exception:
                logger.exception("Isolated MFA retry raised an exception for %s", sid)
        if not matches:
            status[sid] = "missing_textgrid"
            logger.error("MFA output missing for %s", sid)
            continue
        try:
            words = parse_textgrid(matches[0])
            payload = {
                "sample_id": sid,
                "provider": "Montreal Forced Aligner",
                "source_textgrid": str(matches[0]),
                "words": words,
            }
            timestamp_json_path(config, sid).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            status[sid] = "success"
        except Exception as exc:
            status[sid] = f"failed: {exc}"
            logger.exception("Failed to parse MFA output for %s", sid)
    succeeded = sum(value == "success" for value in status.values())
    logger.info("Collected MFA timestamps for %d/%d samples", succeeded, len(manifest))
    return status


def load_word_timestamps(config: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    path = timestamp_json_path(config, sample_id)
    if not path.exists():
        raise FileNotFoundError(f"MFA timestamps are missing for {sample_id}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["words"]
