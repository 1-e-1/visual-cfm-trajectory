"""Combine compatible NPZ visual datasets with optional training oversampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base_dir", type=Path, required=True)
parser.add_argument("--extra_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--extra_train_repeats", type=int, default=1)
args = parser.parse_args()


def _load_split(root: Path, split: str) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    rows = []
    camera_names = None
    for path in sorted(root.glob(f"{split}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            camera_names = data["camera_view_names"]
            rows.append({key: data[key] for key in data.files if key != "camera_view_names"})
    if not rows or camera_names is None:
        raise FileNotFoundError(f"No {split} shards under {root}")
    return rows, camera_names


def main() -> None:
    if args.extra_train_repeats < 1:
        raise ValueError("extra_train_repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        base_rows, camera_names = _load_split(args.base_dir, split)
        extra_rows, extra_camera_names = _load_split(args.extra_dir, split)
        if not np.array_equal(camera_names, extra_camera_names):
            raise ValueError(f"camera view mismatch in {split}")
        common_keys = set(base_rows[0]).intersection(extra_rows[0])
        repeats = args.extra_train_repeats if split == "train" else 1
        payload = {
            key: np.concatenate(
                [row[key] for row in base_rows]
                + [row[key] for _ in range(repeats) for row in extra_rows],
                axis=0,
            )
            for key in sorted(common_keys)
        }
        payload["camera_view_names"] = camera_names
        output = args.output_dir / f"{split}_shard_00000.npz"
        np.savez_compressed(output, **payload)
        print(
            f"[OK] {split}: {len(payload['q'])} samples, "
            f"extra repeats={repeats} -> {output}"
        )


if __name__ == "__main__":
    main()
