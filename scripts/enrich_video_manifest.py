"""Copy nonsensitive probe fields from an existing Q1 reference audit.

This does not probe media anew. The source size and relative path must match.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ("duration_s", "width", "height", "fps", "video_codec", "has_audio", "audio_codec", "audio_sample_rate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.reference.open(newline="", encoding="utf-8-sig") as stream:
        reference = {row["video_relpath"].replace("\\", "/"): row for row in csv.DictReader(stream)}
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        existing_fields = list(reader.fieldnames or ())
    matched = 0
    for row in rows:
        parts = Path(row["relative_path"]).parts
        if row["file_type"] != "mp4":
            continue
        key = "/".join(parts[1:])
        source = reference.get(key)
        if source is None or int(source["source_size_bytes"]) != int(row["file_size_bytes"]):
            raise ValueError(f"Reference path or size mismatch: {row['relative_path']}")
        for field in FIELDS:
            row[field] = source[field]
        row["metadata_source"] = "original_Q1_reference_manifest_not_reprobed"
        matched += 1
    if matched != len(reference):
        raise ValueError(f"Reference count {len(reference)} differs from matched videos {matched}")
    fields = existing_fields + [field for field in (*FIELDS, "metadata_source") if field not in existing_fields]
    with args.manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Attached prior Q1 probe fields for {matched} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
