"""Build task-level difficulty bins and enumerated failure-mode statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robot.torch_kinematics import DifferentiableZBotFK
from src.h1_1111_grasp.trajectory_safety import environment_clearance, link_segments


FAILURE_REASONS = (
    "ENV_COLLISION",
    "SELF_COLLISION",
    "TERMINAL_POSITION",
    "TERMINAL_ORIENTATION",
    "DYNAMIC_LIMIT",
    "NO_IK_OR_UNREACHABLE",
    "NO_VALID_CANDIDATE",
    "TARGET_ASSOCIATION",
    "GRASP_CONTACT",
    "OBJECT_SLIP",
    "PLACE_ERROR",
    "EMERGENCY_STOP",
    "UNKNOWN",
)


def wilson(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z / denominator * np.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    )
    return [float(center - radius), float(center + radius)]


def wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def load_dataset_rows(data_dir: Path) -> list[dict[str, np.ndarray]]:
    rows = []
    for path in sorted(data_dir.glob("test_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            keys = [key for key in data.files if key != "camera_view_names"]
            rows.extend(
                {key: np.array(data[key][index], copy=True) for key in keys}
                for index in range(len(data["sample_id"]))
            )
    if not rows:
        raise FileNotFoundError(f"no test shards under {data_dir}")
    return rows


def primary_failure(record: dict) -> str:
    if record["two_stage_success"]:
        return "NONE"
    stages = (record["grasp"], record["place"])
    if any(not stage["environment_ok"] for stage in stages):
        return "ENV_COLLISION"
    if any(not stage["self_clearance_ok"] for stage in stages):
        return "SELF_COLLISION"
    if any(not stage["terminal_ok"] for stage in stages):
        return "TERMINAL_POSITION"
    if any(not stage["orientation_ok"] for stage in stages):
        return "TERMINAL_ORIENTATION"
    if any(not (stage["dynamics_ok"] and stage["duration_ok"]) for stage in stages):
        return "DYNAMIC_LIMIT"
    if not record["target_component_ok"]:
        return "TARGET_ASSOCIATION"
    if any(
        not record[f"{stage}_has_passing_candidate"]["success"]
        for stage in ("grasp", "place")
    ):
        return "NO_VALID_CANDIDATE"
    return "UNKNOWN"


def summarize_bins(rows: list[dict], field: str, edges: list[float], labels: list[str]) -> list[dict]:
    output = []
    values = np.asarray([float(row[field]) for row in rows])
    for lower, upper, label in zip(edges[:-1], edges[1:], labels):
        mask = (values >= lower) & (values < upper)
        selected = [row for row, keep in zip(rows, mask) if keep]
        successes = sum(int(row["success_joint"]) for row in selected)
        output.append(
            {
                "bin": label,
                "lower": float(lower) if np.isfinite(lower) else None,
                "upper": float(upper) if np.isfinite(upper) else None,
                "n": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected) if selected else None,
                "wilson_95": wilson(successes, len(selected)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data/supplementary_unseen_v622_pose/left",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_c/full_k16_seed20260815.json",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=ROOT / "paper_figures/data_cache/difficulty_failure_per_task.csv",
    )
    parser.add_argument(
        "--output_summary",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_c/difficulty_failure_summary.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset = load_dataset_rows(args.data_dir)
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    records = evaluation["records"]
    by_id = {str(row["sample_id"]): row for row in dataset}
    if set(by_id) != {str(record["sample_id"]) for record in records}:
        raise RuntimeError("dataset and evaluation sample_id sets are not paired")

    q_batch = []
    for record in records:
        row = by_id[str(record["sample_id"])]
        q_batch.extend((row["q_start"], row["q_hold"]))
    fk = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json").to(args.device)
    with torch.no_grad():
        tcp, _ = fk(torch.as_tensor(np.asarray(q_batch), dtype=torch.float32, device=args.device))
    tcp = tcp.cpu().numpy().reshape(len(records), 2, 3)

    phase = np.linspace(0.0, 1.0, 64)
    blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
    direct_paths = []
    for record in records:
        row = by_id[str(record["sample_id"])]
        for condition_key, start_key in (
            ("condition_grasp", "q_start"),
            ("condition_place", "q_hold"),
        ):
            start = np.asarray(row[start_key], dtype=np.float64)
            goal = start + wrap(np.asarray(row[condition_key])[6:12] - start)
            direct_paths.append(start[None] + blend[:, None] * (goal - start)[None])
    direct_segments = link_segments(
        fk, np.asarray(direct_paths).reshape(-1, 6), device=args.device
    ).reshape(len(records), 2, 64, -1, 2, 3)

    task_rows = []
    for index, record in enumerate(records):
        row = by_id[str(record["sample_id"])]
        grasp_joint_distance = np.linalg.norm(
            wrap(np.asarray(row["condition_grasp"])[6:12] - row["q_start"])
        )
        place_joint_distance = np.linalg.norm(
            wrap(np.asarray(row["condition_place"])[6:12] - row["q_hold"])
        )
        grasp_tcp_distance = np.linalg.norm(
            np.asarray(row["condition_grasp"])[-3:] - tcp[index, 0]
        )
        place_tcp_distance = np.linalg.norm(
            np.asarray(row["condition_place"])[-3:] - tcp[index, 1]
        )
        token_mask = np.asarray(row["obstacle_mask"], dtype=bool)
        target_mask = np.asarray(row["target_component_label"], dtype=bool)
        component_centers = np.asarray(row["obstacle_tokens"])[token_mask, 4:7]
        component_is_target = target_mask[token_mask]
        goal = np.asarray(row["condition_grasp"])[-3:]
        local_components = int(
            np.sum(
                (np.linalg.norm(component_centers - goal[None], axis=1) <= 0.20)
                & ~component_is_target
            )
        )
        selected_env_clearance = min(
            float(record["grasp"]["environment_clearance_m"]),
            float(record["place"]["environment_clearance_m"]),
        )
        cloud = np.asarray(row["scene_points"])[
            np.asarray(row["scene_point_mask"], dtype=bool)
        ]
        target_center = np.asarray(row["object_center_foot0"], dtype=np.float32)
        target_size = float(np.max(np.asarray(row["object_size"], dtype=np.float32)))
        target_padding = 0.5 * target_size + 0.015
        place_cloud = cloud[
            np.any(np.abs(cloud - target_center[None]) > target_padding, axis=1)
        ]
        direct_clearance = min(
            environment_clearance(
                direct_segments[index, 0, :, 1:], cloud, capsule_radius_m=0.05
            ),
            environment_clearance(
                direct_segments[index, 1, :, 1:], place_cloud, capsule_radius_m=0.05
            ),
        )
        failure = primary_failure(record)
        task_rows.append(
            {
                "task_id": str(record["sample_id"]),
                "task_index": int(record["task_index"]),
                "layout_index": int(np.asarray(row["layout_index"]).reshape(-1)[0]),
                "obstacle_count": max(0, int(row["n_objects"]) - 1),
                "minimum_passage_clearance_proxy_m": direct_clearance,
                "selected_trajectory_min_env_clearance_m": selected_env_clearance,
                "start_goal_joint_distance_rad": 0.5
                * (grasp_joint_distance + place_joint_distance),
                "start_goal_tcp_distance_m": 0.5
                * (grasp_tcp_distance + place_tcp_distance),
                "goal_local_component_count_0p20m": local_components,
                "target_component_ok": int(record["target_component_ok"]),
                "success_grasp": int(record["grasp"]["success"]),
                "success_place": int(record["place"]["success"]),
                "success_joint": int(record["two_stage_success"]),
                "failure_reason": failure,
                "env_failure": int(
                    not (
                        record["grasp"]["environment_ok"]
                        and record["place"]["environment_ok"]
                    )
                ),
                "self_failure": int(
                    not (
                        record["grasp"]["self_clearance_ok"]
                        and record["place"]["self_clearance_ok"]
                    )
                ),
                "source_evaluation": str(args.evaluation.relative_to(ROOT)),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    clearance_bins = summarize_bins(
        task_rows,
        "minimum_passage_clearance_proxy_m",
        [-np.inf, 0.02, 0.04, 0.06, np.inf],
        ["<2 cm", "2-4 cm", "4-6 cm", ">=6 cm"],
    )
    obstacle_bins = []
    for label, selector in (
        ("1", lambda value: value == 1),
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4+", lambda value: value >= 4),
    ):
        selected = [row for row in task_rows if selector(int(row["obstacle_count"]))]
        successes = sum(int(row["success_joint"]) for row in selected)
        obstacle_bins.append(
            {
                "bin": label,
                "n": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected) if selected else None,
                "wilson_95": wilson(successes, len(selected)),
            }
        )
    joint_values = np.asarray([row["start_goal_joint_distance_rad"] for row in task_rows])
    quartiles = np.quantile(joint_values, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartiles[-1] += 1.0e-9
    joint_bins = summarize_bins(
        task_rows,
        "start_goal_joint_distance_rad",
        quartiles.tolist(),
        ["Q1", "Q2", "Q3", "Q4"],
    )
    failures: dict[str, int] = defaultdict(int)
    representatives: dict[str, list[str]] = defaultdict(list)
    for row in task_rows:
        if row["failure_reason"] != "NONE":
            failures[row["failure_reason"]] += 1
            if len(representatives[row["failure_reason"]]) < 5:
                representatives[row["failure_reason"]].append(row["task_id"])
    summary = {
        "tasks": len(task_rows),
        "successes": sum(row["success_joint"] for row in task_rows),
        "success_rate": float(np.mean([row["success_joint"] for row in task_rows])),
        "success_wilson_95": wilson(
            sum(row["success_joint"] for row in task_rows), len(task_rows)
        ),
        "difficulty_definition": {
            "minimum_passage_clearance_proxy_m": (
                "minimum environment clearance of the independent minimum-jerk direct "
                "reference across grasp/place; a difficulty proxy, not passage ground truth"
            ),
            "obstacle_count": "n_objects - 1 (target excluded)",
            "start_goal_joint_distance_rad": "mean periodic L2 distance across grasp/place",
            "start_goal_tcp_distance_m": "mean TCP Euclidean distance across grasp/place",
            "goal_local_component_count_0p20m": "non-target geometry-token centers within 0.20 m",
        },
        "clearance_bins": clearance_bins,
        "obstacle_bins": obstacle_bins,
        "joint_distance_quartiles": joint_bins,
        "failure_counts": dict(sorted(failures.items())),
        "representative_task_ids": dict(sorted(representatives.items())),
        "allowed_failure_reasons": list(FAILURE_REASONS),
        "per_task_csv": str(args.output_csv.relative_to(ROOT)),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
