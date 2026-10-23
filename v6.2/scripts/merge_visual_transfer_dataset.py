"""Merge left/right NPZ shards into the shared arm-base-frame dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        paths = sorted((args.input_dir / "left").glob(f"{split}_shard_*.npz"))
        paths += sorted((args.input_dir / "right").glob(f"{split}_shard_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no {split} shards under {args.input_dir}")
        rows = []
        camera_names = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                camera_names = data["camera_view_names"]
                rows.append({key: data[key] for key in data.files if key != "camera_view_names"})
        keys = rows[0].keys()
        payload = {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}
        payload["camera_view_names"] = camera_names
        output = args.output_dir / f"{split}_shard_00000.npz"
        np.savez_compressed(output, **payload)
        print(f"[OK] {split}: {len(payload['q'])} -> {output}")


if __name__ == "__main__":
    main()
