"""Upgrade v6.2.1 shards to pose-aware IK labels without changing CFM inputs.

The 15-D condition remains ``q_start + q_pose_ik + p_goal``.  Desired TCP
orientation is represented by the selected pose-IK branch and is also stored as
an explicit quaternion for auditing.  Splits are made by layout, never by row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.npz_dataset import write_npz_shards
from src.h1_1111_grasp.trajectory_safety import (
    environment_clearance,
    link_segments,
    self_clearance,
)
from src.planning.ik_solver import (
    quaternion_distance_rad,
    solve_pose_ik_candidates,
)
from src.robot.torch_kinematics import DifferentiableZBotFK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "visual_v621_full" / "left",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "visual_v622_pose" / "left",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Verify and store only the source exact pose branch; keep unused slots masked.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument(
        "--all-as-test",
        action="store_true",
        help="Place every newly generated layout in a test-only supplementary split.",
    )
    return parser.parse_args()


def load_rows(source_dir: Path, max_rows: int) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    paths = sorted(source_dir.glob("*_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no source shards in {source_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            keys = [key for key in data.files if key != "camera_view_names"]
            count = len(data["sample_id"])
            for index in range(count):
                rows.append(
                    {key: np.array(data[key][index], copy=True) for key in keys}
                )
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def candidate_bank(
    fk: DifferentiableZBotFK,
    primary_q: np.ndarray,
    scene_cloud: np.ndarray,
    count: int,
    restarts: int,
    iterations: int,
    seed: int,
    device: str,
    primary_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    primary = np.asarray(primary_q, dtype=np.float32).reshape(6)
    with torch.no_grad():
        goal_position_t, goal_quaternion_t = fk(
            torch.as_tensor(primary[None], dtype=torch.float32, device=device)
        )
    goal_position = goal_position_t[0].cpu().numpy()
    goal_quaternion = goal_quaternion_t[0].cpu().numpy()

    def segments(q_batch: np.ndarray) -> np.ndarray:
        return link_segments(fk, q_batch, device=device)

    if primary_only:
        primary_segments = segments(primary[None])
        self_value = self_clearance(primary_segments, capsule_radius_m=0.045)
        environment_value = environment_clearance(
            primary_segments[0, 1:], scene_cloud, capsule_radius_m=0.05
        )
        if self_value < 0.01 or environment_value < 0.01:
            raise RuntimeError(
                "the source pose-IK anchor failed collision filtering: "
                f"self={self_value:.4f}, environment={environment_value:.4f}"
            )
        padded = np.repeat(primary[None], count, axis=0).astype(np.float32)
        mask = np.zeros(count, dtype=np.bool_)
        mask[0] = True
        return padded, mask, goal_quaternion.astype(np.float32), {
            "valid_candidates": 1,
            "max_position_error_m": 0.0,
            "max_orientation_error_rad": 0.0,
            "min_self_clearance_m": float(self_value),
            "min_environment_clearance_m": float(environment_value),
        }

    solutions, diagnostics = solve_pose_ik_candidates(
        fk,
        goal_position,
        goal_quaternion,
        n_candidates=count,
        restarts=restarts,
        iterations=iterations,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=np.deg2rad(5.0),
        self_clearance_fn=lambda q: self_clearance(segments(q), capsule_radius_m=0.045),
        environment_clearance_fn=lambda q: environment_clearance(
            segments(q)[0, 1:], scene_cloud, capsule_radius_m=0.05
        ),
        clearance_threshold_m=0.01,
        seed_configurations=primary[None],
        seed=seed,
        device=device,
        return_diagnostics=True,
    )
    if len(solutions) == 0:
        raise RuntimeError("the source pose-IK anchor failed collision filtering")
    primary_distances = np.linalg.norm(
        (solutions - primary[None] + np.pi) % (2.0 * np.pi) - np.pi, axis=1
    )
    primary_index = int(np.argmin(primary_distances))
    if primary_index != 0:
        solutions[[0, primary_index]] = solutions[[primary_index, 0]]
        diagnostics[0], diagnostics[primary_index] = (
            diagnostics[primary_index],
            diagnostics[0],
        )
    valid_count = len(solutions)
    padded = np.repeat(solutions[:1], count, axis=0)
    padded[:valid_count] = solutions
    mask = np.zeros(count, dtype=np.bool_)
    mask[:valid_count] = True
    metrics = {
        "valid_candidates": int(valid_count),
        "max_position_error_m": float(
            max(item.position_error_m for item in diagnostics)
        ),
        "max_orientation_error_rad": float(
            max(item.orientation_error_rad for item in diagnostics)
        ),
        "min_self_clearance_m": float(
            min(item.self_clearance_m for item in diagnostics if item.self_clearance_m is not None)
        ),
        "min_environment_clearance_m": float(
            min(
                item.environment_clearance_m
                for item in diagnostics
                if item.environment_clearance_m is not None
            )
        ),
    }
    return padded.astype(np.float32), mask, goal_quaternion.astype(np.float32), metrics


def main() -> None:
    args = parse_args()
    if args.candidates < 1:
        raise ValueError("--candidates must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = load_rows(args.source_dir, args.max_rows)
    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(
        args.device
    )
    cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
    branch_counts: list[int] = []
    all_metrics: list[dict[str, float]] = []
    started = perf_counter()
    for index, row in enumerate(rows):
        cloud = np.asarray(row["scene_points"], dtype=np.float64)[
            np.asarray(row["scene_point_mask"], dtype=np.bool_)
        ]
        for stage, condition_key in (("grasp", "condition_grasp"), ("place", "condition_place")):
            primary = np.asarray(row[condition_key][6:12], dtype=np.float32)
            cache_key = tuple(np.round(primary, 5))
            result = cache.get(cache_key)
            if result is None:
                result = candidate_bank(
                    fk,
                    primary,
                    cloud,
                    args.candidates,
                    args.restarts,
                    args.iterations,
                    args.seed + index * 17 + (0 if stage == "grasp" else 1),
                    args.device,
                    args.primary_only,
                )
                cache[cache_key] = result
            candidates, candidate_mask, goal_quaternion, metrics = result
            row[f"q_ik_candidates_{stage}"] = candidates
            row[f"q_ik_candidate_mask_{stage}"] = candidate_mask
            row[f"goal_quaternion_{stage}"] = goal_quaternion
            branch_counts.append(int(candidate_mask.sum()))
            all_metrics.append(metrics)
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(
                f"[POSE DATA] {index + 1}/{len(rows)} cache={len(cache)} "
                f"mean_branches={np.mean(branch_counts):.2f}",
                flush=True,
            )

    layout_ids = np.unique(
        [int(np.asarray(row["layout_index"]).reshape(-1)[0]) for row in rows]
    )
    if len(layout_ids) < 3 and not args.all_as_test:
        raise RuntimeError("at least three layouts are required for leakage-free splits")
    if args.all_as_test:
        train_layouts: set[int] = set()
        validation_layouts: set[int] = set()
        test_layouts = set(layout_ids)
    else:
        rng = np.random.default_rng(args.seed)
        layout_ids = rng.permutation(layout_ids)
        validation_count = max(1, int(round(0.1 * len(layout_ids))))
        test_count = max(1, int(round(0.1 * len(layout_ids))))
        train_layouts = set(layout_ids[: len(layout_ids) - validation_count - test_count])
        validation_layouts = set(layout_ids[len(train_layouts) : len(train_layouts) + validation_count])
        test_layouts = set(layout_ids[-test_count:])
    split_layouts = {
        "train": train_layouts,
        "validation": validation_layouts,
        "test": test_layouts,
    }
    split_indices = {
        split: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if int(np.asarray(row["layout_index"]).reshape(-1)[0]) in layouts
            ],
            dtype=np.int64,
        )
        for split, layouts in split_layouts.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[str]] = {}
    for split, indices in split_indices.items():
        paths = write_npz_shards(
            rows, indices, args.output_dir, split, shard_size=args.shard_size
        )
        shards[split] = [str(path) for path in paths]
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["camera_view_names"] = np.asarray(
                ["wrist", "cube", "chest", "abdomen", "head"]
            )
            np.savez_compressed(path, **payload)

    report = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "rows": len(rows),
        "condition_layout": "q_start[6]+q_pose_ik[6]+p_goal[3]",
        "orientation_label": "goal_quaternion_<stage>[4], wxyz",
        "split_by_layout": True,
        "supplementary_test_only": bool(args.all_as_test),
        "split_layouts": {
            key: sorted(int(value) for value in values)
            for key, values in split_layouts.items()
        },
        "split_rows": {key: int(len(value)) for key, value in split_indices.items()},
        "shards": shards,
        "ik_candidates_requested": args.candidates,
        "ik_branch_histogram": {
            str(key): int(value) for key, value in sorted(Counter(branch_counts).items())
        },
        "ik_mean_valid_branches": float(np.mean(branch_counts)),
        "ik_max_position_error_m": float(
            max(item["max_position_error_m"] for item in all_metrics)
        ),
        "ik_max_orientation_error_rad": float(
            max(item["max_orientation_error_rad"] for item in all_metrics)
        ),
        "ik_min_self_clearance_m": float(
            min(item["min_self_clearance_m"] for item in all_metrics)
        ),
        "ik_min_environment_clearance_m": float(
            min(item["min_environment_clearance_m"] for item in all_metrics)
        ),
        "elapsed_s": perf_counter() - started,
    }
    report_path = args.output_dir / "generation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
