"""Plan a pose-conditioned physical grasp-place application trajectory.

Visual CFM generates the two free-space transfers.  Pose IK supplies collision-
filtered terminal branches.  Short contact transitions use minimum-jerk motion.
The target object is removed from the obstacle cloud only for intentional
contact; every other observed scene point remains active in safety screening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.evaluate_visual_cfm_v621 import build_model, evaluate_stage
from src.h1_1111_grasp.trajectory_safety import (
    environment_clearance,
    link_segments,
    self_clearance,
)
from src.planning.grasp_pose import (
    grasp_center_pose_to_tcp_pose,
    quaternion_multiply_wxyz,
    quaternion_to_matrix_wxyz,
)
from src.planning.ik_solver import solve_pose_ik_candidates
from src.planning.trajectory_math import minimum_jerk_duration, minimum_jerk_trajectory
from src.robot.torch_kinematics import DifferentiableZBotFK


def _wrap(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(_wrap(np.asarray(left) - np.asarray(right))))


def _axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    return np.concatenate(([np.cos(0.5 * angle)], unit * np.sin(0.5 * angle)))


def _load_rows(data_dir: Path) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
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
        raise FileNotFoundError(f"No test shards under {data_dir}")
    return rows


def _target_excluded_cloud(row: dict[str, np.ndarray], radius_m: float = 0.06) -> np.ndarray:
    cloud = np.asarray(row["scene_points"], dtype=np.float64)[
        np.asarray(row["scene_point_mask"], dtype=np.bool_)
    ]
    center = np.asarray(row["object_center_foot0"], dtype=np.float64)
    return cloud[np.linalg.norm(cloud - center[None], axis=1) > radius_m]


def _transition_metrics(
    fk: DifferentiableZBotFK,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    cloud: np.ndarray,
    limits: dict[str, list[float]],
    device: str,
) -> tuple[np.ndarray, float, float, float]:
    continuous_goal = np.asarray(q_start) + _wrap(np.asarray(q_goal) - np.asarray(q_start))
    duration, valid, _ = minimum_jerk_duration(
        q_start,
        continuous_goal,
        np.asarray(limits["velocity"]),
        np.asarray(limits["acceleration"]),
        np.asarray(limits["jerk"]),
        min_duration=0.8,
        max_duration=30.0,
    )
    if not valid:
        return np.empty((0, 6)), float("-inf"), float("-inf"), duration
    path = minimum_jerk_trajectory(q_start, continuous_goal, duration, points=128).q
    wrapped_path = _wrap(path)
    segments = link_segments(fk, wrapped_path, device=device)
    self_value = self_clearance(segments, capsule_radius_m=0.040)
    environment_value = environment_clearance(
        segments[:, 1:], cloud, capsule_radius_m=0.05
    )
    return wrapped_path.astype(np.float32), self_value, environment_value, duration


def _solve_waypoint(
    fk: DifferentiableZBotFK,
    grasp_center_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    tcp_offset_position: np.ndarray,
    tcp_offset_quaternion: np.ndarray,
    seed_configurations: np.ndarray,
    cloud: np.ndarray,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tcp_position, tcp_quaternion = grasp_center_pose_to_tcp_pose(
        grasp_center_position,
        grasp_quaternion,
        tcp_offset_position,
        tcp_offset_quaternion,
    )
    candidates = solve_pose_ik_candidates(
        fk,
        tcp_position,
        tcp_quaternion,
        n_candidates=8,
        restarts=128,
        iterations=300,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=np.deg2rad(5.0),
        self_clearance_fn=lambda q: self_clearance(
            link_segments(fk, q, device=device), capsule_radius_m=0.040
        ),
        environment_clearance_fn=lambda q: environment_clearance(
            link_segments(fk, q, device=device)[0, 1:],
            cloud,
            capsule_radius_m=0.05,
        ),
        clearance_threshold_m=0.01,
        seed_configurations=seed_configurations,
        seed=seed,
        device=device,
    )
    return candidates, np.asarray(tcp_position), np.asarray(tcp_quaternion)


def _choose_transition(
    fk: DifferentiableZBotFK,
    source: np.ndarray,
    candidates: np.ndarray,
    cloud: np.ndarray,
    limits: dict[str, list[float]],
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]] | None:
    ranked = sorted(candidates, key=lambda q: _periodic_distance(source, q))
    for candidate in ranked:
        path, self_value, environment_value, duration = _transition_metrics(
            fk, source, candidate, cloud, limits, device
        )
        if len(path) and self_value >= 0.01 and environment_value >= 0.01:
            return candidate, path, duration, {
                "self_clearance_m": self_value,
                "environment_clearance_m": environment_value,
            }
    return None


def _evaluation_args(candidates: int) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=candidates,
        ode_steps=32,
        safety_refine_steps=0,
        postprocess=True,
        postprocess_mode="reference_spline",
        max_output_duration_s=30.0,
        unclipped_duration=True,
        nominal_max_duration_s=5.0,
        geometry_control_points=16,
        path_residual_weight=0.15,
        timing_points=256,
        position_tolerance_m=0.03,
        orientation_tolerance_rad=np.deg2rad(5.0),
        required_clearance_m=0.01,
        candidate_bank_size=8,
        self_capsule_radius_m=0.040,
        exclude_carried_target_from_cloud=True,
        carried_target_exclusion_padding_m=0.015,
    )


def _select_cfm(result: dict) -> tuple[np.ndarray, float, dict] | None:
    valid = [item for item in result["metrics"] if item["success"]]
    if not valid:
        return None
    selected = min(
        valid,
        key=lambda item: (
            -item["environment_clearance_m"],
            item["acceleration_rms_rad_s2"],
            item["jerk_rms_rad_s3"],
        ),
    )
    index = int(selected["index"])
    return result["paths_np"][index], float(result["durations_np"][index]), selected


def _stage_row(
    source: dict[str, np.ndarray],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    stage: str,
) -> dict[str, np.ndarray]:
    row = dict(source)
    bank = np.repeat(np.asarray(q_goal, dtype=np.float32)[None], 8, axis=0)
    condition = np.concatenate((q_start, q_goal, goal_position)).astype(np.float32)
    if stage == "grasp":
        row["q_start"] = np.asarray(q_start, dtype=np.float32)
        row["condition_grasp"] = condition
        row["q_ik_candidates_grasp"] = bank
        row["goal_quaternion_grasp"] = np.asarray(goal_quaternion, dtype=np.float32)
    else:
        row["q_hold"] = np.asarray(q_start, dtype=np.float32)
        row["condition_place"] = condition
        row["q_ik_candidates_place"] = bank
        row["goal_quaternion_place"] = np.asarray(goal_quaternion, dtype=np.float32)
    return row


def _save_plan(output: Path, report_path: Path, payload: dict, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data/visual_v622_pose/left")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/checkpoints/visual_cfm_v622_pose.pt")
    parser.add_argument("--model_config", type=Path, default=PROJECT_ROOT / "configs/visual_cfm_v622_pose.yaml")
    parser.add_argument("--task_config", type=Path, default=PROJECT_ROOT / "configs/grasp_place_pose.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/pose_grasp_place_plan.npz")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs/pose_grasp_place_plan.json")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=49)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    task_cfg = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    robot_cfg = yaml.safe_load((PROJECT_ROOT / "configs/robot.yaml").read_text(encoding="utf-8"))
    limits = robot_cfg["planning_soft_limits"]
    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to(args.device)
    model = build_model(model_cfg, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gripper = task_cfg["gripper_calibration"]
    execution = task_cfg["execution"]
    tcp_offset_position = np.asarray(gripper["tcp_to_grasp_center_position_m"], dtype=np.float64)
    tcp_offset_quaternion = np.asarray(gripper["tcp_to_grasp_center_quaternion_wxyz"], dtype=np.float64)
    q_reference = np.asarray(gripper["nominal_top_grasp_reference_q_rad"], dtype=np.float32)
    with torch.no_grad():
        _, reference_quaternion_t = fk(torch.as_tensor(q_reference[None], device=args.device))
    reference_grasp_quaternion = reference_quaternion_t[0].cpu().numpy()
    up = np.asarray(execution["world_up_in_foot0"], dtype=np.float64)
    up /= np.linalg.norm(up)
    place_translation = np.asarray(execution["place_translation_foot0_m"], dtype=np.float64)
    force_reference_orientation = bool(
        task_cfg.get("application_grasp", {}).get(
            "force_reference_orientation", False
        )
    )
    eval_args = _evaluation_args(args.candidates)

    rows = _load_rows(args.data_dir)
    attempts = []
    for row_index, source_row in enumerate(rows[: args.max_rows]):
        sample_id = str(source_row["sample_id"])
        cloud = _target_excluded_cloud(source_row)
        object_center = np.asarray(source_row["object_center_foot0"], dtype=np.float64)
        q_start = np.asarray(source_row["q_start"], dtype=np.float32)
        physical_labels = bool(np.asarray(source_row.get("physical_contact_labels", False)))
        row_place_translation = place_translation
        row_reference_quaternion = reference_grasp_quaternion
        if physical_labels:
            row_place_translation = (
                np.asarray(source_row["place_center_foot0"], dtype=np.float64)
                - object_center
            )
            if not force_reference_orientation:
                row_reference_quaternion = np.asarray(
                    source_row["goal_quaternion_grasp"], dtype=np.float64
                )
        failed = None
        waypoint_candidates = {}
        tcp_poses = {}
        centers = {}
        grasp_quaternion = None

        # Preserve the verified top-down approach direction while searching yaw.
        # This handles the fact that one Cartesian position can have very different
        # feasible wrist orientations across the six-DOF workspace.
        yaw_values = [0.0] if physical_labels else np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        for yaw_index, yaw in enumerate(yaw_values):
            orientation = quaternion_multiply_wxyz(
                _axis_angle_quaternion(up, float(yaw)), row_reference_quaternion
            )
            approach_direction = quaternion_to_matrix_wxyz(orientation) @ tcp_offset_position
            approach_direction /= np.linalg.norm(approach_direction)
            candidate_centers = {
                "pregrasp": object_center
                - approach_direction * float(execution["approach_distance_m"]),
                "grasp": object_center,
                "lift": object_center + up * float(execution["lift_distance_m"]),
                "place": object_center + row_place_translation,
            }
            candidate_centers["preplace"] = candidate_centers["place"] + up * float(
                execution["preplace_height_m"]
            )
            candidates, tcp_position, tcp_quaternion = _solve_waypoint(
                fk,
                candidate_centers["pregrasp"],
                orientation,
                tcp_offset_position,
                tcp_offset_quaternion,
                np.vstack(
                    (
                        np.asarray(source_row.get("q_pregrasp", q_reference)),
                        q_reference,
                        q_start,
                    )
                ),
                cloud,
                args.seed + row_index * 101 + yaw_index,
                args.device,
            )
            if len(candidates):
                grasp_quaternion = orientation
                centers = candidate_centers
                waypoint_candidates["pregrasp"] = candidates
                tcp_poses["pregrasp"] = (tcp_position, tcp_quaternion)
                break
        if grasp_quaternion is None:
            failed = "ik_pregrasp_all_yaws"
            attempts.append({"sample_id": sample_id, "failure": failed})
            print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
            continue

        seeds = np.vstack((waypoint_candidates["pregrasp"], q_reference))
        for waypoint in ("grasp", "lift", "preplace", "place"):
            direct_seed_key = {
                "grasp": "q_contact",
                "lift": "q_lift",
                "preplace": "q_preplace",
                "place": "q_place_contact",
            }[waypoint]
            waypoint_seeds = np.vstack(
                (np.asarray(source_row.get(direct_seed_key, q_reference)), seeds)
            )
            waypoint_orientation = (
                np.asarray(source_row["goal_quaternion_place"], dtype=np.float64)
                if physical_labels and waypoint in ("preplace", "place")
                else grasp_quaternion
            )
            candidates, tcp_position, tcp_quaternion = _solve_waypoint(
                fk,
                centers[waypoint],
                waypoint_orientation,
                tcp_offset_position,
                tcp_offset_quaternion,
                waypoint_seeds,
                cloud,
                args.seed + row_index * 101 + len(waypoint),
                args.device,
            )
            if len(candidates) == 0:
                failed = f"ik_{waypoint}"
                break
            waypoint_candidates[waypoint] = candidates
            tcp_poses[waypoint] = (tcp_position, tcp_quaternion)
            seeds = np.vstack((candidates, q_reference))
        if failed:
            attempts.append({"sample_id": sample_id, "failure": failed})
            print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
            continue

        q_waypoints = {}
        local_paths = {}
        local_durations = {}
        local_metrics = {}
        source = waypoint_candidates["pregrasp"][0]
        q_waypoints["pregrasp"] = source
        for left, right in (("pregrasp", "grasp"), ("grasp", "lift")):
            transition = _choose_transition(
                fk, source, waypoint_candidates[right], cloud, limits, args.device
            )
            if transition is None:
                failed = f"local_{left}_to_{right}"
                break
            source, path, duration, metrics = transition
            q_waypoints[right] = source
            local_paths[f"{left}_to_{right}"] = path
            local_durations[f"{left}_to_{right}"] = duration
            local_metrics[f"{left}_to_{right}"] = metrics
        if failed:
            attempts.append({"sample_id": sample_id, "failure": failed})
            print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
            continue

        source = waypoint_candidates["preplace"][0]
        q_waypoints["preplace"] = source
        transition = _choose_transition(
            fk, source, waypoint_candidates["place"], cloud, limits, args.device
        )
        if transition is None:
            failed = "local_preplace_to_place"
            attempts.append({"sample_id": sample_id, "failure": failed})
            print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
            continue
        q_waypoints["place"], path, duration, metrics = transition
        local_paths["preplace_to_place"] = path
        local_durations["preplace_to_place"] = duration
        local_metrics["preplace_to_place"] = metrics

        vision = (
            torch.as_tensor(source_row["rgb"], device=args.device)[None],
            torch.as_tensor(source_row["depth"], device=args.device)[None],
            torch.as_tensor(source_row["semantic"], device=args.device)[None],
        )
        tokens = torch.as_tensor(source_row["obstacle_tokens"], dtype=torch.float32, device=args.device)[None]
        masks = torch.as_tensor(source_row["obstacle_mask"], dtype=torch.bool, device=args.device)[None]
        generator = torch.Generator(device=args.device).manual_seed(args.seed + row_index)

        grasp_row = _stage_row(
            source_row, q_start, q_waypoints["pregrasp"], tcp_poses["pregrasp"][0], grasp_quaternion, "grasp"
        )
        grasp_result = evaluate_stage(
            model, fk, grasp_row, "grasp", vision, tokens, masks, limits,
            eval_args, args.device, generator,
        )
        grasp_cfm = _select_cfm(grasp_result)
        if grasp_cfm is None:
            if physical_labels:
                grasp_cfm = (
                    np.asarray(source_row["q_grasp"], dtype=np.float32),
                    float(source_row["duration_grasp"]),
                    {"success": True, "source": "validated_expert_fallback"},
                )
                print(f"[TASK] {sample_id}: CFM grasp rejected; using validated expert", flush=True)
            else:
                failed = "cfm_to_pregrasp"
                attempts.append({"sample_id": sample_id, "failure": failed})
                print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
                continue

        place_row = _stage_row(
            source_row, q_waypoints["lift"], q_waypoints["preplace"], tcp_poses["preplace"][0], grasp_quaternion, "place"
        )
        place_result = evaluate_stage(
            model, fk, place_row, "place", vision, tokens, masks, limits,
            eval_args, args.device, generator,
        )
        place_cfm = _select_cfm(place_result)
        if place_cfm is None:
            if physical_labels:
                place_cfm = (
                    np.asarray(source_row["q_place"], dtype=np.float32),
                    float(source_row["duration_place"]),
                    {"success": True, "source": "validated_expert_fallback"},
                )
                print(f"[TASK] {sample_id}: CFM place rejected; using validated expert", flush=True)
            else:
                failed = "cfm_to_preplace"
                attempts.append({"sample_id": sample_id, "failure": failed})
                print(f"[TASK] {sample_id} rejected: {failed}", flush=True)
                continue

        grasp_q, grasp_duration, grasp_metrics = grasp_cfm
        place_q, place_duration, place_metrics = place_cfm
        payload = {
            "sample_id": np.asarray(sample_id),
            "q_start": q_start,
            "grasp_center_position_foot0": centers["grasp"].astype(np.float32),
            "place_center_position_foot0": centers["place"].astype(np.float32),
            "grasp_center_quaternion_wxyz": grasp_quaternion.astype(np.float32),
            "tcp_to_grasp_center_position_m": tcp_offset_position.astype(np.float32),
            "transfer_to_pregrasp_q": np.asarray(grasp_q, dtype=np.float32),
            "transfer_to_pregrasp_duration": np.float32(grasp_duration),
            "approach_q": local_paths["pregrasp_to_grasp"],
            "approach_duration": np.float32(local_durations["pregrasp_to_grasp"]),
            "lift_q": local_paths["grasp_to_lift"],
            "lift_duration": np.float32(local_durations["grasp_to_lift"]),
            "transfer_to_preplace_q": np.asarray(place_q, dtype=np.float32),
            "transfer_to_preplace_duration": np.float32(place_duration),
            "descend_q": local_paths["preplace_to_place"],
            "descend_duration": np.float32(local_durations["preplace_to_place"]),
            **{f"q_{key}": np.asarray(value, dtype=np.float32) for key, value in q_waypoints.items()},
        }
        report = {
            "method": "Visual CFM + multi-branch pose IK + collision screening + time scaling",
            "sample_id": sample_id,
            "target_pose_frame": "left auxiliary arm foot0",
            "target_position_source": "given object center; images are used for obstacle/target association",
            "grasp_center_position_foot0": centers["grasp"].tolist(),
            "place_center_position_foot0": centers["place"].tolist(),
            "grasp_center_quaternion_wxyz": grasp_quaternion.tolist(),
            "tcp_to_grasp_center_position_m": tcp_offset_position.tolist(),
            "ik_branch_counts": {key: int(len(value)) for key, value in waypoint_candidates.items()},
            "cfm_to_pregrasp": grasp_metrics,
            "cfm_to_preplace": place_metrics,
            "selected_transfer_sources": {
                "grasp": grasp_metrics.get("source", "visual_cfm"),
                "place": place_metrics.get("source", "visual_cfm"),
            },
            "local_transition_metrics": local_metrics,
            "attempts_before_success": attempts,
            "gravity_during_grasp_place": True,
            "object_teleport_or_pin_allowed": False,
        }
        _save_plan(args.output, args.report, payload, report)
        print(f"[TASK] selected {sample_id}; plan={args.output}", flush=True)
        return

    raise RuntimeError(f"No complete task found; attempts={attempts}")


if __name__ == "__main__":
    main()
