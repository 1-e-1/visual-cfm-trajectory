"""Rebuild the demonstration plan for a stable native-contact grasp pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_native_contact_plan import _smooth_cfm_transfer, _wrap
from src.h1_1111_grasp.trajectory_safety import link_segments, self_clearance
from src.planning.grasp_pose import quaternion_to_matrix_wxyz
from src.planning.ik_solver import quaternion_distance_rad, solve_pose_ik_candidates
from src.planning.trajectory_math import minimum_jerk_duration, minimum_jerk_trajectory
from src.robot.torch_kinematics import DifferentiableZBotFK


def _world_from_foot0(stage: Usd.Stage) -> np.ndarray:
    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/UnitreeH1/ZBot6S_left/foot_0")
    ).ComputeLocalToWorldTransform(0)
    origin = np.asarray(matrix.Transform((0.0, 0.0, 0.0)))
    return np.stack(
        [
            np.asarray(matrix.Transform(axis)) - origin
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ],
        axis=1,
    )


def _local_path(
    start: np.ndarray,
    goal: np.ndarray,
    limits: dict[str, list[float]],
    points: int = 256,
) -> tuple[np.ndarray, float]:
    continuous_goal = start + _wrap(goal - start)
    duration, valid, _ = minimum_jerk_duration(
        start,
        continuous_goal,
        np.asarray(limits["velocity"]),
        np.asarray(limits["acceleration"]),
        np.asarray(limits["jerk"]),
        min_duration=1.2,
        max_duration=30.0,
    )
    if not valid:
        raise RuntimeError("local pose transition violates dynamic limits")
    path = minimum_jerk_trajectory(start, continuous_goal, duration, points=points).q
    return _wrap(path).astype(np.float32), float(duration)


def _resample_path(path: np.ndarray, points: int) -> np.ndarray:
    """Resample a continuous joint path without changing its geometric trace."""
    source = np.asarray(path, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2:
        raise ValueError("path must have shape [N, dof] with N >= 2")
    phase_source = np.linspace(0.0, 1.0, len(source))
    phase_target = np.linspace(0.0, 1.0, int(points))
    return np.stack(
        [np.interp(phase_target, phase_source, source[:, joint]) for joint in range(source.shape[1])],
        axis=1,
    ).astype(np.float32)


def _append_timed_paths(
    first: np.ndarray,
    first_duration: float,
    second: np.ndarray,
    second_duration: float,
    points: int = 1024,
) -> tuple[np.ndarray, float]:
    """Concatenate paths while retaining each segment's commanded speed."""
    total_duration = float(first_duration + second_duration)
    first_points = int(round(points * first_duration / total_duration))
    first_points = int(np.clip(first_points, 2, points - 1))
    second_points = points - first_points + 1
    combined = np.concatenate(
        (
            _resample_path(first, first_points),
            _resample_path(second, second_points)[1:],
        ),
        axis=0,
    )
    return combined.astype(np.float32), total_duration


def _retime_sampled_path(
    path: np.ndarray,
    limits: dict[str, list[float]],
    min_duration: float = 3.0,
    max_duration: float = 30.0,
) -> tuple[float, dict[str, float]]:
    """Compute an unclipped time scale for a fixed-phase sampled path."""
    q = np.asarray(path, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, len(q))
    dq_du = np.gradient(q, phase, axis=0, edge_order=2)
    ddq_du2 = np.gradient(dq_du, phase, axis=0, edge_order=2)
    dddq_du3 = np.gradient(ddq_du2, phase, axis=0, edge_order=2)
    velocity = np.asarray(limits["velocity"], dtype=np.float64)
    acceleration = np.asarray(limits["acceleration"], dtype=np.float64)
    jerk = np.asarray(limits["jerk"], dtype=np.float64)
    required_velocity = float(np.max(np.abs(dq_du) / velocity[None]))
    required_acceleration = float(
        np.sqrt(np.max(np.abs(ddq_du2) / acceleration[None]))
    )
    required_jerk = float(np.cbrt(np.max(np.abs(dddq_du3) / jerk[None])))
    required = max(
        float(min_duration), required_velocity, required_acceleration, required_jerk
    )
    if required > max_duration:
        raise RuntimeError(
            f"segmented carry path requires {required:.3f} s, above {max_duration:.3f} s"
        )
    return required, {
        "required_velocity_s": required_velocity,
        "required_acceleration_s": required_acceleration,
        "required_jerk_s": required_jerk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "outputs/pose_grasp_place_plan_true_fingertips_lift.npz",
    )
    parser.add_argument(
        "--usd", type=Path, default=PROJECT_ROOT / "1111_copy_native_contact.usd"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_pose_plan.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_pose_plan.json",
    )
    parser.add_argument("--jaw_world_z", type=float, default=0.10)
    parser.add_argument(
        "--preserve_source_grasp_pose",
        action="store_true",
        help="Keep the source pregrasp, grasp, and lift segments unchanged.",
    )
    parser.add_argument(
        "--place_translation_foot0",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -0.18),
    )
    parser.add_argument(
        "--carry_path_mode",
        choices=("smooth_cfm", "segmented_pose"),
        default="smooth_cfm",
        help=(
            "Use the historical global joint-space spline or concatenate local "
            "pose-preserving IK transitions for strict carry orientation."
        ),
    )
    parser.add_argument(
        "--carry_max_duration",
        type=float,
        default=60.0,
        help="Maximum valid duration for a pose-preserving carry path.",
    )
    parser.add_argument(
        "--return_to_pickup",
        action="store_true",
        help=(
            "After native-contact lift, hold and descend along the exact reverse "
            "lift path to release at the task-specific pickup point."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as source:
        payload = {key: np.array(source[key], copy=True) for key in source.files}
    robot_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/robot.yaml").read_text(encoding="utf-8")
    )
    limits = robot_cfg["planning_soft_limits"]
    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to(
        args.device
    )
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open {args.usd}")
    rotation_world_foot0 = _world_from_foot0(stage)

    q_old_grasp = np.asarray(payload["q_grasp"], dtype=np.float64)
    with torch.no_grad():
        _, old_quaternion = fk(
            torch.as_tensor(q_old_grasp[None], dtype=torch.float32, device=args.device)
        )
    rotation_foot0_tcp = quaternion_to_matrix_wxyz(
        old_quaternion[0].cpu().numpy()
    )
    if args.preserve_source_grasp_pose:
        with torch.no_grad():
            _, lift_quaternion = fk(
                torch.as_tensor(
                    np.asarray(payload["q_lift"], dtype=np.float64)[None],
                    dtype=torch.float32,
                    device=args.device,
                )
            )
        rotation_foot0_tcp = quaternion_to_matrix_wxyz(
            lift_quaternion[0].cpu().numpy()
        )
    rotation_world_tcp = rotation_world_foot0 @ rotation_foot0_tcp
    old_jaw_world = np.asarray(
        [-0.8980241524403301, 0.2840195058594267, 0.33598443702825587]
    )
    if args.preserve_source_grasp_pose:
        target_jaw_world = old_jaw_world.copy()
        target_rotation_foot0 = rotation_foot0_tcp
    else:
        target_z = float(args.jaw_world_z)
        target_xy = old_jaw_world[:2] / np.linalg.norm(old_jaw_world[:2])
        target_jaw_world = np.concatenate(
            (target_xy * np.sqrt(1.0 - target_z**2), [target_z])
        )
        correction, _ = Rotation.align_vectors(
            target_jaw_world[None], old_jaw_world[None]
        )
        target_rotation_foot0 = rotation_world_foot0.T @ (
            correction.as_matrix() @ rotation_world_tcp
        )
    xyzw = Rotation.from_matrix(target_rotation_foot0).as_quat()
    target_quaternion = np.asarray([xyzw[3], *xyzw[:3]], dtype=np.float64)

    object_center = np.asarray(payload["grasp_center_position_foot0"], dtype=np.float64)
    place_center = object_center + np.asarray(
        args.place_translation_foot0, dtype=np.float64
    )
    tcp_offset = np.asarray(payload["tcp_to_grasp_center_position_m"], dtype=np.float64)
    up = np.asarray([-1.0, 0.0, 0.0])
    approach_direction = target_rotation_foot0 @ tcp_offset
    approach_direction /= np.linalg.norm(approach_direction)
    centers = {
        "pregrasp": object_center - 0.08 * approach_direction,
        "grasp": object_center,
        "lift": object_center + 0.10 * up,
        "preplace": place_center + 0.08 * up,
        "place": place_center,
    }

    def clearance_fn(q_values: np.ndarray) -> float:
        return self_clearance(
            link_segments(fk, q_values, device=args.device), capsule_radius_m=0.040
        )

    waypoint_candidates: dict[str, np.ndarray] = {}
    if args.preserve_source_grasp_pose:
        for name in ("pregrasp", "grasp", "lift"):
            waypoint_candidates[name] = np.asarray(payload[f"q_{name}"])[None]
    else:
        seeds = np.vstack(
            [payload[key] for key in ("q_pregrasp", "q_grasp", "q_lift", "q_preplace", "q_place")]
        )
        for index, name in enumerate(("pregrasp", "grasp", "lift")):
            tcp_position = centers[name] - target_rotation_foot0 @ tcp_offset
            candidates = solve_pose_ik_candidates(
                fk,
                tcp_position,
                target_quaternion,
                n_candidates=8,
                restarts=512,
                iterations=500,
                learning_rate=0.04,
                orientation_weight=0.15,
                position_tolerance_m=0.005,
                orientation_tolerance_rad=np.deg2rad(5.0),
                self_clearance_fn=clearance_fn,
                clearance_threshold_m=0.01,
                seed_configurations=seeds,
                seed=20260815 + index,
                device=args.device,
            )
            if not len(candidates):
                raise RuntimeError(f"no safe pose-IK candidate for {name}")
            waypoint_candidates[name] = candidates
            seeds = np.vstack((candidates, seeds))

    def choose_pair(
        left: str,
        right: str,
        source_bias: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        choices = []
        for q_left in waypoint_candidates[left]:
            for q_right in waypoint_candidates[right]:
                path, duration = _local_path(q_left, q_right, limits)
                clearance = clearance_fn(path)
                if clearance >= 0.01:
                    choices.append(
                        (
                            float(np.linalg.norm(_wrap(q_right - q_left))),
                            q_left,
                            q_right,
                            path,
                            duration,
                        )
                    )
        if not choices:
            raise RuntimeError(f"no safe local transition for {left}->{right}")
        def choice_cost(item) -> float:
            cost = float(item[0])
            if source_bias is not None:
                cost += float(np.linalg.norm(_wrap(item[1] - source_bias)))
            return cost

        _, q_left, q_right, path, duration = min(choices, key=choice_cost)
        return q_left, q_right, path, duration

    if args.preserve_source_grasp_pose:
        q_pregrasp = np.asarray(payload["q_pregrasp"], dtype=np.float32)
        q_grasp = np.asarray(payload["q_grasp"], dtype=np.float32)
        q_lift = np.asarray(payload["q_lift"], dtype=np.float32)
        approach_path = np.asarray(payload["approach_q"], dtype=np.float32)
        approach_duration = float(payload["approach_duration"])
        lift_path = np.asarray(payload["lift_q"], dtype=np.float32)
        lift_duration = float(payload["lift_duration"])
    else:
        q_pregrasp, q_grasp, approach_path, approach_duration = choose_pair(
            "pregrasp", "grasp"
        )
        lift_choices = []
        for q_lift in waypoint_candidates["lift"]:
            path, duration = _local_path(q_grasp, q_lift, limits)
            clearance = clearance_fn(path)
            if clearance >= 0.01:
                lift_choices.append(
                    (float(np.linalg.norm(_wrap(q_lift - q_grasp))), q_lift, path, duration)
                )
        if not lift_choices:
            raise RuntimeError("no safe local transition for grasp->lift")
        _, q_lift, lift_path, lift_duration = min(lift_choices, key=lambda item: item[0])
    # Preserve the validated Visual CFM trace. Retargeting all 1024 samples to a
    # new terminal pose can bend the early path through the supported object.
    # A separate local bridge changes only the final pregrasp posture and stays
    # outside the 80 mm approach envelope.
    source_pregrasp_path = np.asarray(
        payload["transfer_to_pregrasp_q"], dtype=np.float64
    )
    source_pregrasp_duration = float(payload["transfer_to_pregrasp_duration"])
    if args.preserve_source_grasp_pose:
        pregrasp_path = source_pregrasp_path.astype(np.float32)
        pregrasp_duration = source_pregrasp_duration
        pregrasp_bridge_duration = 0.0
    else:
        pregrasp_bridge, pregrasp_bridge_duration = _local_path(
            source_pregrasp_path[-1], q_pregrasp, limits
        )
        pregrasp_path, pregrasp_duration = _append_timed_paths(
            source_pregrasp_path,
            source_pregrasp_duration,
            pregrasp_bridge,
            pregrasp_bridge_duration,
            points=1024,
        )
    if args.return_to_pickup:
        transfer_path = np.stack((q_lift, q_lift)).astype(np.float32)
        transfer_duration = 2.0
        q_preplace = np.asarray(q_lift, dtype=np.float32)
        q_place = np.asarray(q_grasp, dtype=np.float32)
        descend_path = np.asarray(lift_path[::-1], dtype=np.float32).copy()
        descend_duration = float(lift_duration)
        maximum_carry_orientation_error = 0.0
        carry_waypoints = [np.asarray(q_lift, dtype=np.float32)]
        waypoint_candidates["preplace"] = q_preplace[None]
        waypoint_candidates["place"] = q_place[None]
        carry_retime = None
    else:
        carry_waypoints = [np.asarray(q_lift, dtype=np.float32)]
    previous_q = np.asarray(q_lift, dtype=np.float32)
    for waypoint_index, phase in enumerate(
        [] if args.return_to_pickup else np.linspace(0.0, 1.0, 25)[1:], start=1
    ):
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        center = (1.0 - blend) * centers["lift"] + blend * centers["preplace"]
        tcp_position = center - target_rotation_foot0 @ tcp_offset
        candidates = solve_pose_ik_candidates(
            fk,
            tcp_position,
            target_quaternion,
            n_candidates=8,
            restarts=128,
            iterations=350,
            learning_rate=0.04,
            orientation_weight=0.15,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=np.deg2rad(5.0),
            self_clearance_fn=clearance_fn,
            clearance_threshold_m=0.01,
            seed_configurations=previous_q[None],
            seed=20260900 + waypoint_index,
            device=args.device,
        )
        if not len(candidates):
            raise RuntimeError(
                f"no safe fixed-pose carry IK at waypoint {waypoint_index}"
            )
        previous_q = min(
            candidates,
            key=lambda candidate: float(
                np.linalg.norm(_wrap(candidate - previous_q))
            ),
        )
        carry_waypoints.append(np.asarray(previous_q, dtype=np.float32))
    if not args.return_to_pickup:
        q_preplace = np.asarray(carry_waypoints[-1], dtype=np.float32)
        waypoint_candidates["preplace"] = q_preplace[None]

    if not args.return_to_pickup:
        place_tcp_position = centers["place"] - target_rotation_foot0 @ tcp_offset
        place_candidates = solve_pose_ik_candidates(
            fk,
            place_tcp_position,
            target_quaternion,
            n_candidates=8,
            restarts=512,
            iterations=500,
            learning_rate=0.04,
            orientation_weight=0.15,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=np.deg2rad(5.0),
            self_clearance_fn=clearance_fn,
            clearance_threshold_m=0.01,
            seed_configurations=q_preplace[None],
            seed=20261000,
            device=args.device,
        )
        if not len(place_candidates):
            raise RuntimeError("no safe place IK on the continuous carry branch")
        waypoint_candidates["place"] = place_candidates
        place_choices = []
        for candidate in place_candidates:
            path, duration = _local_path(q_preplace, candidate, limits)
            clearance = clearance_fn(path)
            if clearance >= 0.01:
                place_choices.append(
                    (
                        float(np.linalg.norm(_wrap(candidate - q_preplace))),
                        candidate,
                        path,
                        duration,
                    )
                )
        if not place_choices:
            raise RuntimeError("no safe continuous preplace->place transition")
        _, q_place, descend_path, descend_duration = min(
            place_choices, key=lambda item: item[0]
        )
    carry_retime = None
    if not args.return_to_pickup and (
        args.preserve_source_grasp_pose or args.carry_path_mode == "segmented_pose"
    ):
        transfer_segments = []
        for start, goal in zip(carry_waypoints[:-1], carry_waypoints[1:]):
            segment, _ = _local_path(start, goal, limits, points=48)
            transfer_segments.append(segment if not transfer_segments else segment[1:])
        transfer_path = np.concatenate(transfer_segments, axis=0).astype(np.float32)
        transfer_duration, carry_retime = _retime_sampled_path(
            transfer_path,
            limits,
            min_duration=6.0,
            max_duration=float(args.carry_max_duration),
        )
    elif not args.return_to_pickup:
        transfer_path, transfer_duration, _ = _smooth_cfm_transfer(
            q_lift,
            np.asarray(carry_waypoints, dtype=np.float32),
            q_preplace,
            limits,
            points=1024,
            nominal_duration=8.0,
            geometry_control_points=len(carry_waypoints),
        )
    if not args.return_to_pickup:
        with torch.no_grad():
            _, transfer_quaternions = fk(
                torch.as_tensor(transfer_path, dtype=torch.float32, device=args.device)
            )
        carry_orientation_errors = quaternion_distance_rad(
            transfer_quaternions,
            torch.as_tensor(
                target_quaternion, dtype=torch.float32, device=args.device
            ),
        )
        maximum_carry_orientation_error = float(torch.max(carry_orientation_errors).item())
    if not args.return_to_pickup and maximum_carry_orientation_error > np.deg2rad(8.0):
        maximum_error_index = int(torch.argmax(carry_orientation_errors).item())
        debug_path = args.output.with_suffix(".debug.npz")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            debug_path,
            transfer_path=transfer_path,
            carry_waypoints=np.asarray(carry_waypoints, dtype=np.float32),
            target_quaternion=target_quaternion.astype(np.float32),
            orientation_error_rad=carry_orientation_errors.detach().cpu().numpy(),
        )
        raise RuntimeError(
            "smoothed carry path does not preserve the fixed gripper pose: "
            f"{np.rad2deg(maximum_carry_orientation_error):.3f} deg at "
            f"sample {maximum_error_index}; debug={debug_path}"
        )
    for name, path in (
        ("transfer_to_pregrasp", pregrasp_path),
        ("approach", approach_path),
        ("lift", lift_path),
        ("transfer_to_preplace", transfer_path),
        ("descend", descend_path),
    ):
        clearance = clearance_fn(path)
        if clearance < 0.01:
            raise RuntimeError(f"{name} self-clearance is only {clearance:.6f} m")

    payload.update(
        {
            "grasp_center_quaternion_wxyz": target_quaternion.astype(np.float32),
            "place_center_position_foot0": place_center.astype(np.float32),
            "transfer_to_pregrasp_q": pregrasp_path,
            "transfer_to_pregrasp_duration": np.float32(pregrasp_duration),
            "approach_q": approach_path,
            "approach_duration": np.float32(approach_duration),
            "lift_q": lift_path,
            "lift_duration": np.float32(lift_duration),
            "transfer_to_preplace_q": transfer_path,
            "transfer_to_preplace_duration": np.float32(transfer_duration),
            "descend_q": descend_path,
            "descend_duration": np.float32(descend_duration),
            "q_pregrasp": q_pregrasp.astype(np.float32),
            "q_grasp": q_grasp.astype(np.float32),
            "q_lift": q_lift.astype(np.float32),
            "q_preplace": q_preplace.astype(np.float32),
            "q_place": q_place.astype(np.float32),
        }
    )
    payload.pop("terminal_place_correction_q", None)
    payload.pop("terminal_place_correction_duration", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "jaw_world_axis": target_jaw_world.tolist(),
        "grasp_quaternion_foot0_wxyz": target_quaternion.tolist(),
        "ik_branch_counts": {
            key: int(len(value)) for key, value in waypoint_candidates.items()
        },
        "durations_s": {
            "visual_cfm_to_pregrasp": pregrasp_duration,
            "visual_cfm_original": source_pregrasp_duration,
            "pregrasp_pose_bridge": pregrasp_bridge_duration,
            "approach": approach_duration,
            "lift": lift_duration,
            "visual_cfm_to_preplace": transfer_duration,
            "descend": descend_duration,
        },
        "carry_pose_waypoints": int(len(carry_waypoints)),
        "carry_path_mode": args.carry_path_mode,
        "return_to_pickup": bool(args.return_to_pickup),
        "carry_retime": carry_retime,
        "carry_max_duration_s": float(args.carry_max_duration),
        "maximum_carry_orientation_error_deg": float(
            np.rad2deg(maximum_carry_orientation_error)
        ),
        "self_clearance_m": {
            name: float(clearance_fn(path))
            for name, path in (
                ("visual_cfm_to_pregrasp", pregrasp_path),
                ("approach", approach_path),
                ("lift", lift_path),
                ("visual_cfm_to_preplace", transfer_path),
                ("descend", descend_path),
            )
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
