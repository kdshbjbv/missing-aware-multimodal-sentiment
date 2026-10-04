from src.prepare_manifest import normalize_clip_id


def test_normalize_clip_id():
    assert normalize_clip_id(13.0) == "13"
    assert normalize_clip_id("13") == "13"
    assert normalize_clip_id("13.0") == "13"
