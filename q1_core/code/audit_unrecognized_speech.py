from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.config import read_config
from src.detect_unrecognized_speech import (
    MissingSpeechSettings,
    PyannoteVADExtractor,
    VADSettings,
    find_unrecognized_speech_candidates,
    load_or_extract_vad,
)
from src.problem1_dataset import covarep_frame_times, load_covarep_mat
from src.utils import safe_sample_name


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit speech regions that have neither ASR words nor official text."
    )
    parser.add_argument("--config", default="configs/feature_config.yaml")
    parser.add_argument("--work-dir", default="work/versionA1")
    parser.add_argument("--official-pkl", default="outputs/versionA1_before_text_gap_fix.pkl")
    parser.add_argument("--whisperx-dir", default="work/versionB/asr/whisperx")
    parser.add_argument("--cache-dir", default="work/versionA1/vad/pyannote")
    parser.add_argument("--output", default="outputs/versionA1_vad_unrecognized_audit.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force-vad", action="store_true")
    parser.add_argument("--word-padding-seconds", type=float, default=0.10)
    parser.add_argument("--min-gap-seconds", type=float, default=0.35)
    parser.add_argument("--min-vad-score", type=float, default=0.70)
    parser.add_argument("--min-voiced-fraction", type=float, default=0.50)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = read_config(project_path(args.config))
    work = project_path(args.work_dir)
    config["paths"]["work_dir"] = str(work)
    official_path = project_path(args.official_pkl)
    with official_path.open("rb") as handle:
        official_dataset = pickle.load(handle)

    vad_settings = VADSettings()
    missing_settings = MissingSpeechSettings(
        word_padding_seconds=args.word_padding_seconds,
        min_gap_seconds=args.min_gap_seconds,
        min_mean_vad_score=args.min_vad_score,
        min_voiced_fraction=args.min_voiced_fraction,
    )
    extractor = PyannoteVADExtractor(args.device, vad_settings)
    cache_dir = project_path(args.cache_dir)
    whisperx_dir = project_path(args.whisperx_dir)
    sample_audits: list[dict[str, Any]] = []
    sensitivity_settings = [
        (padding, gap)
        for padding in (0.07, 0.10, 0.15)
        for gap in (0.25, 0.32, 0.40, 0.50)
    ]
    sensitivity: dict[str, dict[str, int]] = {
        f"pad={padding:.2f},gap={gap:.2f}": {"samples": 0, "positions": 0}
        for padding, gap in sensitivity_settings
    }

    for sample_index, sample in enumerate(official_dataset["samples"], start=1):
        sample_id = str(sample["sample_id"])
        stem = safe_sample_name(sample_id)
        wav_path = work / "audio" / f"{stem}.wav"
        mat_path = work / "audio" / f"{stem}.mat"
        asr_path = whisperx_dir / f"{stem}.json"
        vad_payload = load_or_extract_vad(
            extractor,
            wav_path,
            cache_dir / f"{stem}.json",
            force=args.force_vad,
        )
        asr_payload = load_json(asr_path)
        if mat_path.is_file():
            features, names, _ = load_covarep_mat(
                mat_path,
                str(config["problem1"]["covarep_feature_key"]),
                str(config["problem1"]["covarep_names_key"]),
                int(config["problem1"]["covarep_expected_dim"]),
            )
            name_map = {name.strip().upper(): index for index, name in enumerate(names)}
            if "VUV" not in name_map:
                raise KeyError(f"{sample_id}: COVAREP feature names do not contain VUV")
            covarep_times = covarep_frame_times(
                wav_path,
                len(features),
                float(config["problem1"]["covarep_hop_seconds"]),
                float(config["problem1"].get("covarep_first_sample_fraction", 0.5)),
            )
            covarep_vuv = features[:, name_map["VUV"]]
        else:
            # Missing COVAREP is already represented as an unavailable audio
            # modality. Keep the VAD audit reproducible and conservative: no
            # candidate can pass the voiced-fraction guard without this check.
            covarep_times = np.zeros(0, dtype=np.float64)
            covarep_vuv = np.zeros(0, dtype=np.float32)
        official_length = int(sample["valid_length"])
        official_words = [str(word) for word in sample["words"][:official_length]]
        official_timestamps = np.asarray(
            sample["timestamps"][:official_length], dtype=np.float64
        )

        def candidates_for(settings: MissingSpeechSettings) -> list[dict[str, Any]]:
            return find_unrecognized_speech_candidates(
                vad_payload=vad_payload,
                asr_words=asr_payload.get("words", []),
                official_words=official_words,
                official_timestamps=official_timestamps,
                covarep_times=covarep_times,
                covarep_vuv=covarep_vuv,
                settings=settings,
            )

        candidates = candidates_for(missing_settings)
        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        if candidates:
            sample_audits.append(
                {
                    "sample_index_1based": sample_index,
                    "sample_id": sample_id,
                    "asr_word_count": len(asr_payload.get("words", [])),
                    "candidate_count": len(candidates),
                    "accepted_count": len(accepted),
                    "candidates": candidates,
                }
            )

        for padding, gap in sensitivity_settings:
            key = f"pad={padding:.2f},gap={gap:.2f}"
            settings = replace(
                missing_settings,
                word_padding_seconds=padding,
                min_gap_seconds=gap,
            )
            accepted_sensitivity = [
                candidate for candidate in candidates_for(settings) if candidate["accepted"]
            ]
            if accepted_sensitivity:
                sensitivity[key]["samples"] += 1
                sensitivity[key]["positions"] += len(accepted_sensitivity)
        print(
            f"VAD_AUDIT [{sample_index:03d}/100] {sample_id} "
            f"candidates={len(candidates)} accepted={len(accepted)}",
            flush=True,
        )

    accepted_records = [
        {
            "sample_index_1based": item["sample_index_1based"],
            "sample_id": item["sample_id"],
            **candidate,
        }
        for item in sample_audits
        for candidate in item["candidates"]
        if candidate["accepted"]
    ]
    report = {
        "method": "Pyannote VAD minus padded WhisperX word coverage, guarded by COVAREP VUV and official-only MFA spans",
        "vad_settings": asdict(vad_settings),
        "missing_speech_settings": asdict(missing_settings),
        "accepted_sample_count": len({item["sample_id"] for item in accepted_records}),
        "accepted_position_count": len(accepted_records),
        "accepted_candidates": accepted_records,
        "sample_audits": sample_audits,
        "sensitivity": sensitivity,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"VAD_AUDIT_OK samples={report['accepted_sample_count']} "
        f"positions={report['accepted_position_count']} output={output}"
    )


if __name__ == "__main__":
    main()
