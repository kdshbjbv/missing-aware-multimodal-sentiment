from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from src.detect_unrecognized_speech import (
    MissingSpeechSettings,
    find_unrecognized_speech_candidates,
    lcs_word_matches,
    subtract_intervals,
)
from versionA import (
    _subsequence_positions,
    apply_official_text_gap_fill,
    apply_vad_unrecognized_speech_fill,
)


def test_subsequence_positions_detects_only_strict_official_omissions() -> None:
    assert _subsequence_positions(
        ["they've", "been", "able"], ["that", "we", "do,", "they've", "been", "able"]
    ) == [3, 4, 5]
    assert _subsequence_positions(["a", "wrong", "word"], ["a", "right", "word"]) is None
    assert _subsequence_positions(["same", "words"], ["same", "words"]) == [0, 1]


def test_gap_fill_inserts_zero_text_and_reuses_asr_timeline(tmp_path) -> None:
    sample = {
        "sample_id": "video$_$13",
        "words": ["hello", "world", "", "", ""],
        "timestamps": np.asarray([[0.0, 0.2], [0.2, 0.4], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        "text": np.asarray([[1, 2], [3, 4], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        "text_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "audio": np.zeros((5, 1), dtype=np.float32),
        "vision": np.zeros((5, 1), dtype=np.float32),
        "vision_22": np.zeros((5, 1), dtype=np.float32),
        "sequence_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "audio_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "vision_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "audio_frame_counts": np.asarray([1, 1, 0, 0, 0], dtype=np.int32),
        "vision_frame_counts": np.asarray([1, 1, 0, 0, 0], dtype=np.int32),
        "valid_length": 2,
        "original_word_count": 2,
        "truncated": False,
        "audio_status": "covarep",
        "audio_available": True,
        "vision_status": "available",
        "alignment_status": "success",
        "vision_timestamp_match_before_correction": True,
        "vision_alignment_source": "existing",
    }
    asr_sample = {
        "sample_id": "video$_$13",
        "words": ["well", "hello", "world", "today", ""],
        "timestamps": np.asarray(
            [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0, 0]], dtype=np.float32
        ),
        "audio": np.ones((5, 1), dtype=np.float32),
        "vision": np.ones((5, 1), dtype=np.float32),
        "vision_22": np.ones((5, 1), dtype=np.float32),
        "sequence_mask": np.asarray([1, 1, 1, 1, 0], dtype=np.uint8),
        "audio_mask": np.asarray([1, 1, 1, 1, 0], dtype=np.uint8),
        "vision_mask": np.asarray([1, 1, 1, 1, 0], dtype=np.uint8),
        "audio_frame_counts": np.asarray([1, 1, 1, 1, 0], dtype=np.int32),
        "vision_frame_counts": np.asarray([1, 1, 1, 1, 0], dtype=np.int32),
        "valid_length": 4,
        "original_word_count": 4,
        "truncated": False,
        "audio_status": "covarep",
        "audio_available": True,
        "vision_status": "available",
        "whisperx_model": "test",
        "whisperx_alignment_model": "test",
    }
    asr_path = tmp_path / "versionB.pkl"
    with asr_path.open("wb") as handle:
        pickle.dump(
            {"samples": [asr_sample], "meta": {"dataset_variant": "versionB"}}, handle
        )
    summary = pd.DataFrame.from_records(
        [{"sample_id": "video$_$13", "text_valid_windows": 2}]
    )
    dataset = {"samples": [sample], "meta": {"max_len": 5, "text_dim": 2}}

    updated, updated_summary, report = apply_official_text_gap_fill(
        dataset, summary, asr_path
    )
    result = updated["samples"][0]
    np.testing.assert_array_equal(result["text_mask"], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(result["text"][1], [1, 2])
    np.testing.assert_array_equal(result["text"][2], [3, 4])
    np.testing.assert_array_equal(result["text"][[0, 3, 4]], 0)
    np.testing.assert_array_equal(result["audio"], asr_sample["audio"])
    assert result["timeline_source"] == "whisperx_official_text_gap_fill"
    assert result["official_text_missing_word_count"] == 2
    assert int(updated_summary.loc[0, "text_valid_windows"]) == 2
    assert report[0]["missing_text_positions_1based"] == [1, 4]
    assert report[0]["missing_asr_words"] == ["well", "today"]


def test_vad_candidate_excludes_asr_and_official_only_windows() -> None:
    vad = {
        "duration": 2.0,
        "speech_segments": [{"start": 0.0, "end": 2.0}],
        "score_times": np.arange(0.0, 2.0, 0.01).tolist(),
        "speech_scores": np.full(200, 0.95).tolist(),
    }
    candidates = find_unrecognized_speech_candidates(
        vad_payload=vad,
        asr_words=[{"word": "hello", "start": 0.0, "end": 0.4}],
        official_words=["hello", "known"],
        official_timestamps=np.asarray([[0.0, 0.4], [0.5, 0.9]], dtype=np.float64),
        covarep_times=np.arange(0.0, 2.0, 0.01),
        covarep_vuv=np.ones(200),
        settings=MissingSpeechSettings(
            word_padding_seconds=0.1,
            min_gap_seconds=0.35,
            min_mean_vad_score=0.7,
            min_voiced_fraction=0.5,
        ),
    )
    assert lcs_word_matches(["hello", "known"], ["hello"]) == [(0, 0)]
    assert subtract_intervals([(0, 2)], [(0, 0.5), (0.5, 0.9)]) == [(0.9, 2.0)]
    assert len(candidates) == 1
    assert candidates[0]["start"] == 0.9
    assert candidates[0]["accepted"] is True


def test_vad_fill_inserts_pseudo_position_and_pools_modalities(tmp_path, monkeypatch) -> None:
    sample = {
        "sample_id": "video$_$1",
        "words": ["hello", "world", "", "", ""],
        "timestamps": np.asarray([[0.0, 0.4], [1.0, 1.4], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        "text": np.asarray([[1, 2], [3, 4], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        "audio": np.zeros((5, 1), dtype=np.float32),
        "vision": np.zeros((5, 1), dtype=np.float32),
        "vision_22": np.zeros((5, 1), dtype=np.float32),
        "sequence_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "text_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "audio_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "vision_mask": np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
        "audio_frame_counts": np.asarray([2, 2, 0, 0, 0], dtype=np.int32),
        "vision_frame_counts": np.asarray([3, 3, 0, 0, 0], dtype=np.int32),
        "valid_length": 2,
        "original_word_count": 2,
        "truncated": False,
        "official_text_gap_filled": False,
        "official_text_missing_word_count": 0,
        "timeline_source": "mfa_official_text",
        "alignment_status": "success",
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "method": "test",
                "missing_speech_settings": {},
                "accepted_candidates": [
                    {
                        "sample_id": "video$_$1",
                        "start": 0.5,
                        "end": 0.9,
                        "confidence": 0.8,
                        "accepted": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_pool(config, sample_id, windows):
        return {
            "audio": np.asarray([[7]], dtype=np.float32),
            "audio_mask": np.asarray([1], dtype=np.uint8),
            "audio_frame_counts": np.asarray([4], dtype=np.int32),
            "vision": np.asarray([[8]], dtype=np.float32),
            "vision_mask": np.asarray([1], dtype=np.uint8),
            "vision_frame_counts": np.asarray([5], dtype=np.int32),
            "vision_22": np.asarray([[9]], dtype=np.float32),
        }

    monkeypatch.setattr("versionA._pool_candidate_modalities", fake_pool)
    summary = pd.DataFrame([{"sample_id": "video$_$1"}])
    dataset = {"samples": [sample], "meta": {"max_len": 5}}
    updated, updated_summary, report = apply_vad_unrecognized_speech_fill(
        dataset, summary, audit_path, {"paths": {"work_dir": str(tmp_path)}}
    )
    result = updated["samples"][0]
    assert result["words"][:3] == ["hello", "[TEXT_MISSING]", "world"]
    np.testing.assert_array_equal(result["text_mask"], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(result["text"][1], [0, 0])
    np.testing.assert_array_equal(result["audio"][1], [7])
    np.testing.assert_array_equal(result["vision"][1], [8])
    assert result["text_missing_reason"][1] == "asr_unrecognized_speech"
    assert result["text_missing_confidence"][1] == np.float32(0.8)
    np.testing.assert_array_equal(result["official_text_position_indices"], [0, 2])
    assert int(updated_summary.loc[0, "vad_text_missing_position_count"]) == 1
    assert report[0]["position_1based"] == 2
