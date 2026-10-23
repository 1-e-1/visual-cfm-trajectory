"""Evaluate paired RRT-Connect and STOMP baselines on pose test tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
from scipy.interpolate import CubicSpline
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.h1_1111_grasp.trajectory_safety import (
    environment_clearance_batch,
    link_segments,
    self_clearance_batch,
)
from src.planning.flow_postprocess import retime_sampled_path
from src.planning.ik_solver import quaternion_distance_rad
from src.robot.torch_kinematics import DifferentiableZBotFK


def wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def load_rows(data_dir: Path, max_tasks: int) -> list[dict[str, np.ndarray]]:
    required_keys = (
        "sample_id",
        "q_start",
        "q_hold",
        "q_ik_candidates_grasp",
        "q_ik_candidates_place",
        "p_goal",
        "p_place",
        "goal_quaternion_grasp",
        "goal_quaternion_place",
        "scene_points",
        "scene_point_mask",
        "object_center_foot0",
        "object_size",
    )
    rows = []
    for path in sorted(data_dir.glob("test_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in required_keys}
            for index in range(len(arrays["sample_id"])):
                rows.append(
                    {key: np.array(value[index], copy=True) for key, value in arrays.items()}
                )
                if max_tasks and len(rows) >= max_tasks:
                    return rows
    if not rows:
        raise FileNotFoundError(f"no test shards under {data_dir}")
    return rows


def resample_controls(controls: np.ndarray, points: int) -> np.ndarray:
    continuous = np.empty_like(controls, dtype=np.float64)
    continuous[0] = controls[0]
    for index in range(1, len(controls)):
        continuous[index] = continuous[index - 1] + wrap(
            controls[index] - continuous[index - 1]
        )
    source = np.linspace(0.0, 1.0, len(controls))
    target = np.linspace(0.0, 1.0, points)
    if len(controls) >= 4:
        spline = CubicSpline(
            source,
            continuous,
            axis=0,
            bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
        )
        path = spline(target)
    else:
        path = np.stack(
            [np.interp(target, source, continuous[:, joint]) for joint in range(6)],
            axis=1,
        )
    path[0] = continuous[0]
    path[-1] = continuous[-1]
    return wrap(path)


class CollisionChecker:
    def __init__(
        self,
        fk: DifferentiableZBotFK,
        cloud: np.ndarray,
        device: str,
        search_point_limit: int = 512,
    ) -> None:
        self.fk = fk
        self.device = device
        self.full_cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
        if len(self.full_cloud) > search_point_limit:
            indices = np.linspace(0, len(self.full_cloud) - 1, search_point_limit, dtype=int)
            self.search_cloud = self.full_cloud[indices]
        else:
            self.search_cloud = self.full_cloud

    def clearances(
        self, paths: np.ndarray, *, full_cloud: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(paths, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        count, points, _ = values.shape
        segments = link_segments(
            self.fk, values.reshape(-1, 6), device=self.device
        ).reshape(count, points, -1, 2, 3)
        cloud = self.full_cloud if full_cloud else self.search_cloud
        environment = environment_clearance_batch(
            segments[:, :, 1:], cloud, capsule_radius_m=0.05
        )
        self_values = self_clearance_batch(
            segments.reshape(-1, segments.shape[2], 2, 3), capsule_radius_m=0.045
        ).reshape(count, points).min(axis=1)
        return environment, self_values

    def edge_valid(self, start: np.ndarray, goal: np.ndarray, resolution: float) -> bool:
        delta = wrap(goal - start)
        points = max(2, int(np.ceil(np.linalg.norm(delta) / resolution)) + 1)
        path = wrap(start[None] + np.linspace(0.0, 1.0, points)[:, None] * delta[None])
        environment, self_values = self.clearances(path)
        return bool(environment[0] >= 0.01 and self_values[0] >= 0.01)


def trace_tree(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
    path = []
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    return path[::-1]


def rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    checker: CollisionChecker,
    rng: np.random.Generator,
    timeout_s: float,
    max_iterations: int = 800,
    step_size: float = 0.45,
    edge_resolution: float = 0.16,
) -> np.ndarray | None:
    if checker.edge_valid(start, goal, edge_resolution):
        return np.stack((start, goal))
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    active = 0
    started = perf_counter()

    def extend(tree: tuple[list[np.ndarray], list[int]], target: np.ndarray) -> tuple[str, int]:
        nodes, parents = tree
        distances = [np.linalg.norm(wrap(target - node)) for node in nodes]
        nearest = int(np.argmin(distances))
        distance = distances[nearest]
        if distance < 1.0e-8:
            return "reached", nearest
        q_new = wrap(nodes[nearest] + min(1.0, step_size / distance) * wrap(target - nodes[nearest]))
        if not checker.edge_valid(nodes[nearest], q_new, edge_resolution):
            return "trapped", nearest
        nodes.append(q_new)
        parents.append(nearest)
        return ("reached" if np.linalg.norm(wrap(target - q_new)) < 1.0e-5 else "advanced"), len(nodes) - 1

    for _ in range(max_iterations):
        if perf_counter() - started >= timeout_s:
            break
        q_random = goal if rng.random() < 0.15 else rng.uniform(-np.pi, np.pi, size=6)
        status, index_a = extend(trees[active], q_random)
        if status != "trapped":
            target = trees[active][0][index_a]
            while perf_counter() - started < timeout_s:
                status_b, index_b = extend(trees[1 - active], target)
                if status_b == "trapped":
                    break
                if status_b == "reached":
                    path_a = trace_tree(*trees[active], index_a)
                    path_b = trace_tree(*trees[1 - active], index_b)
                    if active == 0:
                        return np.asarray(path_a + path_b[-2::-1])
                    return np.asarray(path_b + path_a[-2::-1])
        active = 1 - active
    return None


def shortcut_path(
    path: np.ndarray,
    checker: CollisionChecker,
    rng: np.random.Generator,
    attempts: int = 80,
) -> np.ndarray:
    result = [np.asarray(q) for q in path]
    for _ in range(attempts):
        if len(result) <= 2:
            break
        left, right = sorted(rng.choice(len(result), size=2, replace=False).tolist())
        if right - left <= 1:
            continue
        if checker.edge_valid(result[left], result[right], resolution=0.10):
            result = result[: left + 1] + result[right:]
    return np.asarray(result)


def stomp(
    start: np.ndarray,
    goal: np.ndarray,
    checker: CollisionChecker,
    rng: np.random.Generator,
    timeout_s: float,
    control_points: int = 12,
    rollouts: int = 12,
    iterations: int = 10,
) -> np.ndarray | None:
    phase = np.linspace(0.0, 1.0, control_points)
    blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
    continuous_goal = start + wrap(goal - start)
    controls = start[None] + blend[:, None] * (continuous_goal - start)[None]

    def score(paths: np.ndarray) -> np.ndarray:
        environment, self_values = checker.clearances(paths)
        collision = 2500.0 * (
            np.maximum(0.0, 0.01 - environment)
            + np.maximum(0.0, 0.01 - self_values)
        )
        length = np.sum(np.linalg.norm(wrap(np.diff(paths, axis=1)), axis=-1), axis=1)
        smoothness = np.mean(np.diff(np.unwrap(paths, axis=1, period=2 * np.pi), n=2, axis=1) ** 2, axis=(1, 2))
        return collision + 0.08 * length + 0.3 * smoothness

    started = perf_counter()
    sigma = 0.38
    current_path = resample_controls(controls, 32)
    current_cost = float(score(current_path[None])[0])
    for _ in range(iterations):
        if perf_counter() - started >= timeout_s:
            break
        noise = rng.normal(0.0, sigma, size=(rollouts, control_points, 6))
        noise[:, 0] = 0.0
        noise[:, -1] = 0.0
        for _ in range(2):
            noise[:, 1:-1] = (
                0.25 * noise[:, :-2] + 0.5 * noise[:, 1:-1] + 0.25 * noise[:, 2:]
            )
        rollout_controls = controls[None] + noise
        paths = np.asarray([resample_controls(item, 32) for item in rollout_controls])
        costs = score(paths)
        span = float(np.ptp(costs))
        probabilities = np.exp(-10.0 * (costs - costs.min()) / max(span, 1.0e-9))
        probabilities /= probabilities.sum()
        proposal = controls + np.sum(probabilities[:, None, None] * noise, axis=0)
        proposal[0], proposal[-1] = start, continuous_goal
        proposal_path = resample_controls(proposal, 32)
        proposal_cost = float(score(proposal_path[None])[0])
        best = int(np.argmin(costs))
        if costs[best] < proposal_cost:
            proposal = rollout_controls[best]
            proposal[0], proposal[-1] = start, continuous_goal
            proposal_path = paths[best]
            proposal_cost = float(costs[best])
        if proposal_cost < current_cost:
            controls, current_path, current_cost = proposal, proposal_path, proposal_cost
        sigma *= 0.82
    final_path = resample_controls(controls, 64)
    environment, self_values = checker.clearances(final_path, full_cloud=True)
    if environment[0] < 0.01 or self_values[0] < 0.01:
        return None
    return final_path


def evaluate_geometric_path(
    path: np.ndarray | None,
    checker: CollisionChecker,
    limits: dict,
    planning_time_ms: float,
    fk: DifferentiableZBotFK,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    device: str,
) -> dict:
    empty = {
        "success": False,
        "planning_time_ms": planning_time_ms,
        "joint_path_length_rad": float("nan"),
        "environment_clearance_m": float("nan"),
        "self_clearance_m": float("nan"),
        "acceleration_rms_rad_s2": float("nan"),
        "jerk_rms_rad_s3": float("nan"),
        "duration_s": float("nan"),
        "terminal_position_error_m": float("nan"),
        "terminal_orientation_error_rad": float("nan"),
        "failure_reason": "NO_VALID_CANDIDATE",
        "_trajectory": None,
    }
    if path is None or len(path) < 2:
        return empty
    if len(path) < 4:
        path = resample_controls(path, 8)
    trajectory, timing = retime_sampled_path(
        q=path,
        predicted_duration=0.8,
        min_duration=0.8,
        nominal_max_duration=5.0,
        max_output_duration=30.0,
        velocity_limits=np.asarray(limits["velocity"]),
        acceleration_limits=np.asarray(limits["acceleration"]),
        jerk_limits=np.asarray(limits["jerk"]),
        timing_points=256,
    )
    environment, self_values = checker.clearances(trajectory.q, full_cloud=True)
    with torch.no_grad():
        terminal_position, terminal_quaternion = fk(
            torch.as_tensor(trajectory.q[-1:], dtype=torch.float32, device=device)
        )
    position_error = float(
        np.linalg.norm(terminal_position[0].cpu().numpy() - goal_position)
    )
    orientation_error = float(
        quaternion_distance_rad(
            terminal_quaternion.cpu().numpy(), goal_quaternion[None]
        )[0]
    )
    path_length = float(
        np.sum(np.linalg.norm(wrap(np.diff(trajectory.q, axis=0)), axis=1))
    )
    success = bool(
        environment[0] >= 0.01
        and self_values[0] >= 0.01
        and position_error <= 0.03
        and orientation_error <= 0.0872664626
        and timing["duration_valid"]
    )
    if environment[0] < 0.01:
        failure = "ENV_COLLISION"
    elif self_values[0] < 0.01:
        failure = "SELF_COLLISION"
    elif position_error > 0.03:
        failure = "TERMINAL_POSITION"
    elif orientation_error > 0.0872664626:
        failure = "TERMINAL_ORIENTATION"
    elif not timing["duration_valid"]:
        failure = "DYNAMIC_LIMIT"
    else:
        failure = "NONE"
    return {
        "success": success,
        "planning_time_ms": planning_time_ms,
        "joint_path_length_rad": path_length,
        "environment_clearance_m": float(environment[0]),
        "self_clearance_m": float(self_values[0]),
        "acceleration_rms_rad_s2": float(np.sqrt(np.mean(trajectory.ddq**2))),
        "jerk_rms_rad_s3": float(np.sqrt(np.mean(trajectory.dddq**2))),
        "duration_s": float(trajectory.duration),
        "terminal_position_error_m": position_error,
        "terminal_orientation_error_rad": orientation_error,
        "failure_reason": failure,
        "_trajectory": {
            "time": trajectory.time,
            "q": trajectory.q,
            "dq": trajectory.dq,
            "ddq": trajectory.ddq,
            "dddq": trajectory.dddq,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data/supplementary_unseen_v622_pose/left",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_a/classical_planners.json",
    )
    parser.add_argument("--methods", default="rrt_connect,stomp")
    parser.add_argument("--max_tasks", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--timeout_per_stage_s", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=None,
        help="Optional output directory for one replayable trajectory NPZ per task and method.",
    )
    args = parser.parse_args()

    rows = load_rows(args.data_dir, args.max_tasks)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - {"rrt_connect", "stomp"}
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    fk = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json").to(args.device)
    limits = yaml.safe_load((ROOT / "configs/robot.yaml").read_text(encoding="utf-8"))[
        "planning_soft_limits"
    ]
    records = []
    started_all = perf_counter()
    for task_index, row in enumerate(rows):
        cloud = np.asarray(row["scene_points"])[np.asarray(row["scene_point_mask"], dtype=bool)]
        target_center = np.asarray(row["object_center_foot0"], dtype=np.float32)
        target_size = float(np.max(np.asarray(row["object_size"], dtype=np.float32)))
        target_padding = 0.5 * target_size + 0.015
        clouds = {
            "grasp": cloud,
            "place": cloud[np.any(np.abs(cloud - target_center[None]) > target_padding, axis=1)],
        }
        starts = {"grasp": np.asarray(row["q_start"]), "place": np.asarray(row["q_hold"])}
        goals = {
            "grasp": np.asarray(row["q_ik_candidates_grasp"][0]),
            "place": np.asarray(row["q_ik_candidates_place"][0]),
        }
        goal_positions = {
            "grasp": np.asarray(row["p_goal"]),
            "place": np.asarray(row["p_place"]),
        }
        goal_quaternions = {
            "grasp": np.asarray(row["goal_quaternion_grasp"]),
            "place": np.asarray(row["goal_quaternion_place"]),
        }
        for method_index, method in enumerate(methods):
            stage_results = {}
            stage_trajectories = {}
            for stage_index, stage in enumerate(("grasp", "place")):
                checker = CollisionChecker(fk, clouds[stage], args.device)
                rng = np.random.default_rng(
                    args.seed + 100000 * method_index + 2 * task_index + stage_index
                )
                planning_started = perf_counter()
                if method == "rrt_connect":
                    geometric = rrt_connect(
                        starts[stage], goals[stage], checker, rng, args.timeout_per_stage_s
                    )
                    if geometric is not None:
                        geometric = shortcut_path(geometric, checker, rng)
                else:
                    geometric = stomp(
                        starts[stage], goals[stage], checker, rng, args.timeout_per_stage_s
                    )
                stage_results[stage] = evaluate_geometric_path(
                    geometric,
                    checker,
                    limits,
                    0.0,
                    fk,
                    goal_positions[stage],
                    goal_quaternions[stage],
                    args.device,
                )
                stage_results[stage]["planning_time_ms"] = 1000.0 * (
                    perf_counter() - planning_started
                )
                stage_trajectories[stage] = stage_results[stage].pop("_trajectory")
            joint_success = bool(
                stage_results["grasp"]["success"]
                and stage_results["place"]["success"]
            )
            failure = (
                "NONE"
                if joint_success
                else next(
                    stage_results[stage]["failure_reason"]
                    for stage in ("grasp", "place")
                    if not stage_results[stage]["success"]
                )
            )
            trajectory_file = ""
            if args.trajectory_dir is not None:
                args.trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = args.trajectory_dir / (
                    f"task_{task_index:04d}_{method}.npz"
                )
                payload = {
                    "task_index": np.int32(task_index),
                    "task_id": np.asarray(str(row["sample_id"])),
                    "method": np.asarray(method),
                    "seed": np.int64(args.seed),
                }
                for stage in ("grasp", "place"):
                    trajectory = stage_trajectories[stage]
                    if trajectory is None:
                        payload[f"{stage}_available"] = np.bool_(False)
                        continue
                    payload[f"{stage}_available"] = np.bool_(True)
                    for key, value in trajectory.items():
                        payload[f"{stage}_{key}"] = np.asarray(value)
                np.savez_compressed(trajectory_path, **payload)
                trajectory_file = str(trajectory_path.resolve())
            records.append(
                {
                    "task_index": task_index,
                    "task_id": str(row["sample_id"]),
                    "method": method,
                    "seed": args.seed,
                    "grasp": stage_results["grasp"],
                    "place": stage_results["place"],
                    "success_grasp": stage_results["grasp"]["success"],
                    "success_place": stage_results["place"]["success"],
                    "success_joint": joint_success,
                    "trajectory_file": trajectory_file,
                    "failure_reason": failure,
                }
            )
        print(f"[CLASSICAL] {task_index + 1}/{len(rows)}", flush=True)

    report = {
        "protocol": {
            "paired_tasks": True,
            "tasks": len(rows),
            "seed": args.seed,
            "methods": methods,
            "timeout_per_stage_s": args.timeout_per_stage_s,
            "required_clearance_m": 0.01,
            "self_capsule_radius_m": 0.045,
            "environment_capsule_radius_m": 0.05,
            "timing_points": 256,
            "max_output_duration_s": 30.0,
            "position_tolerance_m": 0.03,
            "orientation_tolerance_rad": 0.0872664626,
            "trajectory_export_enabled": args.trajectory_dir is not None,
            "curobo_status": "not installed in D:/conda_envs/env_isaaclab; not evaluated",
        },
        "elapsed_s": perf_counter() - started_all,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    flat_rows = []
    for record in records:
        flat = {
            "task_id": record["task_id"],
            "method": record["method"],
            "seed": record["seed"],
            "success_grasp": int(record["success_grasp"]),
            "success_place": int(record["success_place"]),
            "success_joint": int(record["success_joint"]),
            "trajectory_file": record["trajectory_file"],
            "failure_reason": record["failure_reason"],
        }
        for stage in ("grasp", "place"):
            for key, value in record[stage].items():
                if key not in {"success", "failure_reason"}:
                    flat[f"{key}_{stage}"] = value
        flat_rows.append(flat)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"tasks": len(rows), "methods": methods, "elapsed_s": report["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
