"""Audit frozen Pose Visual CFM dataset shapes, splits, labels and IK validity."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def counter(values: np.ndarray) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(np.asarray(values).reshape(-1).tolist()).items())
    }


def summarize_split(paths: list[Path]) -> dict[str, object]:
    arrays: dict[str, list[np.ndarray]] = {}
    first_shapes = {}
    camera_names = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in (
                "layout_index",
                "object_type",
                "target_component_label",
                "q_ik_candidate_mask_grasp",
                "q_ik_candidate_mask_place",
                "collision_free",
            ):
                arrays.setdefault(key, []).append(np.asarray(data[key]))
            if not first_shapes:
                first_shapes = {
                    key: list(data[key].shape[1:])
                    for key in (
                        "condition_grasp",
                        "q_grasp",
                        "q_grasp_experts",
                        "q_ik_candidates_grasp",
                        "obstacle_tokens",
                        "scene_points",
                        "rgb",
                        "depth",
                        "semantic",
                    )
                }
                camera_names = data["camera_view_names"].tolist()
    merged = {key: np.concatenate(values) for key, values in arrays.items()}
    target_index = np.argmax(merged["target_component_label"], axis=1)
    grasp_ik_count = merged["q_ik_candidate_mask_grasp"].sum(axis=1)
    place_ik_count = merged["q_ik_candidate_mask_place"].sum(axis=1)
    target_counts = Counter(target_index.tolist())
    majority = max(target_counts.values()) / len(target_index)
    return {
        "samples": int(len(target_index)),
        "shards": [str(path.relative_to(ROOT)) for path in paths],
        "sample_tensor_shapes": first_shapes,
        "camera_view_names": camera_names,
        "layout_distribution": counter(merged["layout_index"]),
        "object_type_distribution": counter(merged["object_type"]),
        "target_component_index_distribution": counter(target_index),
        "target_component_majority_accuracy": float(majority),
        "valid_grasp_ik_count_distribution": counter(grasp_ik_count),
        "valid_place_ik_count_distribution": counter(place_ik_count),
        "collision_free_rate": float(np.mean(merged["collision_free"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path, default=ROOT / "data/visual_v622_pose/left"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_v622/dataset_audit.json",
    )
    args = parser.parse_args()

    splits = {}
    for split in ("train", "validation", "test"):
        paths = sorted(args.data_dir.glob(f"{split}_shard_*.npz"))
        if not paths:
            raise FileNotFoundError(f"missing {split} shards under {args.data_dir}")
        splits[split] = summarize_split(paths)
    train_layouts = set(splits["train"]["layout_distribution"])
    validation_layouts = set(splits["validation"]["layout_distribution"])
    test_layouts = set(splits["test"]["layout_distribution"])
    payload = {
        "data_dir": str(args.data_dir.relative_to(ROOT)),
        "splits": splits,
        "layout_overlap": {
            "train_validation": sorted(train_layouts & validation_layouts),
            "train_test": sorted(train_layouts & test_layouts),
            "validation_test": sorted(validation_layouts & test_layouts),
        },
        "notes": [
            "IK arrays reserve eight slots; candidate masks determine actual valid solutions.",
            "A disjoint layout split is an environment-layout OOD split, not an IID random split.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
