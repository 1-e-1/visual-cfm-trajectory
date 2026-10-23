"""Verify the frozen Visual CFM checkpoint and dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    expected = [
        (
            (root / manifest["checkpoint"]["path"]).resolve(),
            manifest["checkpoint"]["sha256"],
        )
    ]
    dataset_root = (root / manifest["dataset"]["root"]).resolve()
    expected.extend(
        (dataset_root / item["path"], item["sha256"])
        for item in manifest["dataset"]["files"]
    )
    failures = []
    for path, expected_hash in expected:
        actual_hash = sha256(path) if path.is_file() else "missing"
        ok = actual_hash.lower() == str(expected_hash).lower()
        print(f"{'OK' if ok else 'FAIL'} {path} {actual_hash}")
        if not ok:
            failures.append(path)
    if failures:
        raise SystemExit(f"frozen asset verification failed for {len(failures)} file(s)")


if __name__ == "__main__":
    main()
