"""Verify a file manifest without printing record contents."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_attachment2_schema(root: Path) -> list[str]:
    import numpy as np
    from e7a_core.src.data import _normalise_split, load_pickle_compat

    problems = []
    for filename, steps in (("aligned_50.pkl", 50), ("unaligned_50.pkl", 500)):
        data = load_pickle_compat(root / filename)
        if set(data) != {"train", "valid", "test"}:
            problems.append(f"Unexpected splits in {filename}")
            continue
        seen_ids = set()
        for split_name, section in data.items():
            count = len(section.get("id", []))
            ids = [str(value) for value in section["id"]]
            if len(set(ids)) != count or seen_ids.intersection(ids):
                problems.append(f"Duplicate sample IDs in {filename}/{split_name}")
            seen_ids.update(ids)
            text = np.asarray(section["text_bert"])
            audio = np.asarray(section["audio"])
            vision = np.asarray(section["vision"])
            if text.shape != (count, 3, 50):
                problems.append(f"text_bert shape mismatch in {filename}/{split_name}")
            if audio.shape != (count, steps, 74) or vision.shape != (count, steps, 35):
                problems.append(f"Audio/Vision shape mismatch in {filename}/{split_name}")
            for key in ("classification_labels", "regression_labels", "raw_text"):
                if len(section.get(key, [])) != count:
                    problems.append(f"{key} length mismatch in {filename}/{split_name}")
            if steps == 50:
                _normalise_split(section)
        del data
        gc.collect()
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, str | int]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "file_type": path.suffix.lower().lstrip(".") or "unknown",
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="Create manifest from current files")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Missing data directory: {args.data_dir}")
    actual = inventory(args.data_dir)
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("relative_path", "file_size_bytes", "sha256", "file_type"))
            writer.writeheader()
            writer.writerows(actual)
        print(f"Manifest written: {len(actual)} files, {sum(row['file_size_bytes'] for row in actual)} bytes")
        return 0
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        expected = list(csv.DictReader(stream))
    expected_by_path = {row["relative_path"]: row for row in expected}
    actual_by_path = {str(row["relative_path"]): row for row in actual}
    problems = []
    for path in sorted(set(expected_by_path) | set(actual_by_path)):
        if path not in expected_by_path:
            problems.append(f"Unexpected file: {path}")
        elif path not in actual_by_path:
            problems.append(f"Missing file: {path}")
        else:
            current = actual_by_path[path]
            reference = expected_by_path[path]
            if current["file_size_bytes"] != int(reference["file_size_bytes"]):
                problems.append(f"Size mismatch: {path}")
            if current["sha256"] != reference["sha256"]:
                problems.append(f"SHA-256 mismatch: {path}")
    if args.data_dir.name == "attachment1":
        video_count = sum(row["file_type"] in ("mp4", "avi", "mov", "mkv") for row in actual)
        if video_count != 100:
            problems.append(f"Expected 100 videos, found {video_count}")
    if args.data_dir.name == "attachment2":
        names = {Path(str(row["relative_path"])).name for row in actual}
        required = {"aligned_50.pkl", "unaligned_50.pkl", "label.xlsx"}
        if names != required or len(actual) != 3:
            problems.append(f"Expected attachment 2 files: {', '.join(sorted(required))}")
        elif not problems:
            problems.extend(validate_attachment2_schema(args.data_dir))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"Verified {len(actual)} files, {sum(row['file_size_bytes'] for row in actual)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
