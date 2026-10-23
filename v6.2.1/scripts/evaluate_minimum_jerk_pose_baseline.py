"""Evaluate a quintic minimum-jerk Pose IK baseline on the frozen test split."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from h1_1111_grasp.trajectory_safety import (  # noqa: E402
    environment_clearance_batch,
    link_segments,
    self_clearance_batch,
)
from src.planning.ik_solver import quaternion_distance_rad  # noqa: E402
from src.planning.trajectory_math import (  # noqa: E402
    minimum_jerk_duration,
    minimum_jerk_trajectory,
)
from src.robot.torch_kinematics import DifferentiableZBotFK  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z / denominator * sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_rows(data_dir: Path, max_tasks: int) -> list[dict[str, np.ndarray]]:
    rows = []
    for path in sorted(data_dir.glob("test_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            shard = {
                key: np.ascontiguousarray(data[key])
                for key in data.files
                if key != "camera_view_names"
            }
        rows.extend(
            {key: value[index] for key, value in shard.items()}
            for index in range(len(shard["condition_grasp"]))
        )
    if not rows:
        raise FileNotFoundError(f"no test shards under {data_dir}")
    return rows[:max_tasks] if max_tasks else rows


def evaluate_stage(
    row: dict[str, np.ndarray],
    stage: str,
    fk_model: DifferentiableZBotFK,
    limits: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    q_start = np.asarray(
        row["q_start"] if stage == "grasp" else row["q_hold"], dtype=np.float64
    )
    candidate_mask = np.asarray(row[f"q_ik_candidate_mask_{stage}"], dtype=np.bool_)
    q_goals = np.asarray(row[f"q_ik_candidates_{stage}"], dtype=np.float64)[candidate_mask]
    if len(q_goals) == 0:
        return {"success": False, "failure_reason": "no_valid_ik_candidate"}

    cloud = np.asarray(row["scene_points"], dtype=np.float64)[
        np.asarray(row["scene_point_mask"], dtype=np.bool_)
    ]
    if stage == "place" and args.exclude_carried_target_from_cloud:
        target_center = np.asarray(row["object_center_foot0"], dtype=np.float64).reshape(3)
        target_size = float(np.max(np.asarray(row["object_size"], dtype=np.float64)))
        padding = 0.5 * target_size + args.carried_target_exclusion_padding_m
        cloud = cloud[np.any(np.abs(cloud - target_center[None]) > padding, axis=1)]
    goal_position = np.asarray(row[f"condition_{stage}"][-3:], dtype=np.float64)
    goal_quaternion = np.asarray(row[f"goal_quaternion_{stage}"], dtype=np.float64)
    metrics = []
    for index, q_goal_wrapped in enumerate(q_goals):
        q_goal = q_start + (q_goal_wrapped - q_start + np.pi) % (2.0 * np.pi) - np.pi
        duration, duration_ok, timing = minimum_jerk_duration(
            q_start,
            q_goal,
            np.asarray(limits["velocity"]),
            np.asarray(limits["acceleration"]),
            np.asarray(limits["jerk"]),
            args.min_duration_s,
            args.max_duration_s,
        )
        trajectory = minimum_jerk_trajectory(
            q_start, q_goal, duration, points=args.trajectory_points
        )
        segments = link_segments(fk_model, trajectory.q, device=args.device)
        environment = float(
            environment_clearance_batch(
                segments[None, :, 1:], cloud, capsule_radius_m=0.05
            )[0]
        )
        self_clearance = float(
            self_clearance_batch(segments, args.self_capsule_radius_m).min()
        )
        with torch.no_grad():
            position, quaternion = fk_model(
                torch.as_tensor(q_goal, dtype=torch.float32, device=args.device)[None]
            )
        terminal_error = float(
            np.linalg.norm(position[0].cpu().numpy() - goal_position)
        )
        orientation_error = float(
            quaternion_distance_rad(
                quaternion.cpu().numpy(), goal_quaternion[None]
            )[0]
        )
        dynamics_ok = bool(
            (np.max(np.abs(trajectory.dq), axis=0) <= np.asarray(limits["velocity"]) * (1 + 1e-6)).all()
            and (np.max(np.abs(trajectory.ddq), axis=0) <= np.asarray(limits["acceleration"]) * (1 + 1e-6)).all()
            and (np.max(np.abs(trajectory.dddq), axis=0) <= np.asarray(limits["jerk"]) * (1 + 1e-6)).all()
        )
        success = bool(
            terminal_error <= args.position_tolerance_m
            and orientation_error <= args.orientation_tolerance_rad
            and environment >= args.required_clearance_m
            and self_clearance >= args.required_clearance_m
            and dynamics_ok
            and duration_ok
        )
        metrics.append(
            {
                "candidate_index": index,
                "success": success,
                "terminal_error_m": terminal_error,
                "orientation_error_rad": orientation_error,
                "environment_clearance_m": environment,
                "self_clearance_m": self_clearance,
                "velocity_rms_rad_s": float(np.sqrt(np.mean(trajectory.dq**2))),
                "acceleration_rms_rad_s2": float(np.sqrt(np.mean(trajectory.ddq**2))),
                "jerk_rms_rad_s3": float(np.sqrt(np.mean(trajectory.dddq**2))),
                "duration_s": duration,
                "duration_ok": duration_ok,
                "dynamics_ok": dynamics_ok,
                "timing": timing,
            }
        )
    passing = [metric for metric in metrics if metric["success"]]
    return min(
        passing or metrics,
        key=lambda metric: (
            not metric["success"],
            metric["terminal_error_m"]
            + 0.1 * metric["orientation_error_rad"]
            + 40.0
            * max(
                0.0,
                args.required_clearance_m
                - min(metric["environment_clearance_m"], metric["self_clearance_m"]),
            ),
        ),
    )


def summarize(records: list[dict[str, object]], stage: str) -> dict[str, object]:
    values = [record[stage] for record in records]
    successes = [value for value in values if value["success"]]
    result: dict[str, object] = {
        "tasks": len(values),
        "successes": len(successes),
        "success_rate": len(successes) / max(1, len(values)),
        "success_rate_wilson_95": wilson(len(successes), len(values)),
    }
    for metric in (
        "terminal_error_m",
        "orientation_error_rad",
        "environment_clearance_m",
        "self_clearance_m",
        "velocity_rms_rad_s",
        "acceleration_rms_rad_s2",
        "jerk_rms_rad_s3",
        "duration_s",
    ):
        source = successes if successes else values
        metric_values = np.asarray([value.get(metric, np.nan) for value in source])
        result[f"mean_{metric}"] = float(np.nanmean(metric_values))
        result[f"p95_{metric}"] = float(np.nanquantile(metric_values, 0.95))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path, default=ROOT / "data/visual_v622_pose/left"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_v622/baselines/minimum_jerk.json",
    )
    parser.add_argument("--max_tasks", type=int, default=49)
    parser.add_argument("--trajectory_points", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--position_tolerance_m", type=float, default=0.03)
    parser.add_argument("--orientation_tolerance_rad", type=float, default=0.0872664626)
    parser.add_argument("--required_clearance_m", type=float, default=0.01)
    parser.add_argument("--self_capsule_radius_m", type=float, default=0.045)
    parser.add_argument("--min_duration_s", type=float, default=0.8)
    parser.add_argument("--max_duration_s", type=float, default=30.0)
    parser.add_argument(
        "--exclude_carried_target_from_cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--carried_target_exclusion_padding_m", type=float, default=0.015)
    args = parser.parse_args()

    rows = load_rows(args.data_dir, args.max_tasks)
    limits = load_yaml(ROOT / "configs/robot.yaml")["planning_soft_limits"]
    fk_model = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json").to(args.device)
    records = []
    started = perf_counter()
    for index, row in enumerate(rows):
        grasp = evaluate_stage(row, "grasp", fk_model, limits, args)
        place = evaluate_stage(row, "place", fk_model, limits, args)
        records.append(
            {
                "task_index": index,
                "sample_id": str(row["sample_id"]),
                "grasp": grasp,
                "place": place,
                "two_stage_success": bool(grasp["success"] and place["success"]),
            }
        )
        print(
            f"[BASELINE] {index + 1}/{len(rows)} grasp={grasp['success']} "
            f"place={place['success']}",
            flush=True,
        )

    two_stage_successes = int(sum(record["two_stage_success"] for record in records))
    report = {
        "method": "quintic_minimum_jerk_to_pose_ik",
        "tasks": len(records),
        "trajectory_points": args.trajectory_points,
        "grasp": summarize(records, "grasp"),
        "place": summarize(records, "place"),
        "two_stage": {
            "tasks": len(records),
            "successes": two_stage_successes,
            "success_rate": two_stage_successes / len(records),
            "success_rate_wilson_95": wilson(two_stage_successes, len(records)),
        },
        "elapsed_s": perf_counter() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
