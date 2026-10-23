"""Compute paired joint/TCP diversity and geometry-mode counts from candidate banks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[1]


def wrapped_delta(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def pairwise_path_distance(paths: np.ndarray, periodic: bool) -> np.ndarray:
    count = len(paths)
    matrix = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        delta = paths[left + 1 :] - paths[left]
        if periodic:
            delta = wrapped_delta(delta)
        distances = np.linalg.norm(delta, axis=-1).mean(axis=-1)
        matrix[left, left + 1 :] = distances
        matrix[left + 1 :, left] = distances
    return matrix


def mode_count(distance_matrix: np.ndarray, threshold: float) -> int:
    if len(distance_matrix) <= 1:
        return len(distance_matrix)
    condensed = squareform(distance_matrix, checks=False)
    tree = linkage(condensed, method="average")
    return int(len(np.unique(fcluster(tree, t=threshold, criterion="distance"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        type=Path,
        default=ROOT / "outputs/paper_v622/candidate_diversity",
    )
    parser.add_argument("--joint_cluster_threshold_rad", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_figures/data_cache/candidate_diversity_per_task.csv",
    )
    args = parser.parse_args()

    rows = []
    for bank_dir in sorted(args.input_root.glob("k*_banks"), key=lambda p: int(p.name[1:-6])):
        count = int(bank_dir.name[1:-6])
        for path in sorted(bank_dir.glob("task_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                for stage in ("grasp", "place"):
                    q = np.asarray(data[f"{stage}_q"], dtype=np.float64)
                    tcp = np.asarray(data[f"{stage}_tcp"], dtype=np.float64)
                    success = np.asarray(data[f"{stage}_success"], dtype=np.bool_)
                    q_distance = pairwise_path_distance(q, periodic=True)
                    tcp_distance = pairwise_path_distance(tcp, periodic=False)
                    upper = np.triu_indices(len(q), 1)
                    rows.append(
                        {
                            "task_index": int(data["task_index"]),
                            "sample_id": str(data["sample_id"]),
                            "stage": stage,
                            "candidates": count,
                            "mean_pairwise_joint_distance_rad": (
                                float(q_distance[upper].mean()) if count > 1 else float("nan")
                            ),
                            "mean_pairwise_tcp_distance_m": (
                                float(tcp_distance[upper].mean()) if count > 1 else float("nan")
                            ),
                            "joint_geometry_modes": mode_count(
                                q_distance, args.joint_cluster_threshold_rad
                            ),
                            "passing_candidates": int(success.sum()),
                            "task_success": int(success.any()),
                            "cluster_threshold_rad": args.joint_cluster_threshold_rad,
                            "source_file": str(path.relative_to(ROOT)),
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"no candidate bank files under {args.input_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} stage-task rows to {args.output}")


if __name__ == "__main__":
    main()
