"""Diagnose pose-IK reachability for a dataset target before physical grasping."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.h1_1111_grasp.trajectory_safety import environment_clearance, link_segments, self_clearance
from src.planning.grasp_pose import (
    grasp_center_pose_to_tcp_pose,
    quaternion_multiply_wxyz,
    quaternion_to_matrix_wxyz,
)
from src.planning.ik_solver import solve_pose_ik_candidates
from src.robot.torch_kinematics import DifferentiableZBotFK


def axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/grasp_pose_ik_diagnostic.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs/grasp_place_pose.yaml").read_text(encoding="utf-8"))
    gripper = config["gripper_calibration"]
    execution = config["execution"]
    with np.load(sorted(glob.glob(str(ROOT / "data/visual_v622_pose/left/test_shard_*.npz")))[0]) as data:
        row = {key: data[key][args.row] for key in data.files if key != "camera_view_names"}
    cloud = np.asarray(row["scene_points"])[np.asarray(row["scene_point_mask"], dtype=np.bool_)]
    object_center = np.asarray(row["object_center_foot0"], dtype=np.float64)
    cloud = cloud[np.linalg.norm(cloud - object_center[None], axis=1) > 0.06]

    fk = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json").to(args.device)
    q_reference = np.asarray(gripper["nominal_top_grasp_reference_q_rad"], dtype=np.float32)
    with torch.no_grad():
        _, reference_quaternion = fk(torch.as_tensor(q_reference[None], device=args.device))
    reference_quaternion = reference_quaternion[0].cpu().numpy()
    offset = np.asarray(gripper["tcp_to_grasp_center_position_m"], dtype=np.float64)
    offset_quaternion = np.asarray(gripper["tcp_to_grasp_center_quaternion_wxyz"], dtype=np.float64)
    up = np.asarray(execution["world_up_in_foot0"], dtype=np.float64)

    records = []
    for yaw_index, yaw in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
        orientation = quaternion_multiply_wxyz(
            axis_angle_quaternion(up, float(yaw)), reference_quaternion
        )
        forward = quaternion_to_matrix_wxyz(orientation) @ offset
        forward /= np.linalg.norm(forward)
        pregrasp_center = object_center - forward * float(execution["approach_distance_m"])
        tcp_position, tcp_quaternion = grasp_center_pose_to_tcp_pose(
            pregrasp_center, orientation, offset, offset_quaternion
        )
        unconstrained, unconstrained_diag = solve_pose_ik_candidates(
            fk,
            tcp_position,
            tcp_quaternion,
            n_candidates=8,
            restarts=args.restarts,
            iterations=args.iterations,
            seed_configurations=np.vstack((q_reference, row["q_start"])),
            seed=20260815 + yaw_index,
            device=args.device,
            return_diagnostics=True,
        )
        safe, safe_diag = solve_pose_ik_candidates(
            fk,
            tcp_position,
            tcp_quaternion,
            n_candidates=8,
            restarts=args.restarts,
            iterations=args.iterations,
            self_clearance_fn=lambda q: self_clearance(
                link_segments(fk, q, device=args.device), capsule_radius_m=0.045
            ),
            environment_clearance_fn=lambda q: environment_clearance(
                link_segments(fk, q, device=args.device)[0, 1:], cloud, capsule_radius_m=0.05
            ),
            clearance_threshold_m=0.01,
            seed_configurations=np.vstack((q_reference, row["q_start"])),
            seed=20260815 + yaw_index,
            device=args.device,
            return_diagnostics=True,
        )
        self_values = [
            self_clearance(link_segments(fk, q[None], device=args.device), capsule_radius_m=0.045)
            for q in unconstrained
        ]
        environment_values = [
            environment_clearance(
                link_segments(fk, q[None], device=args.device)[0, 1:],
                cloud,
                capsule_radius_m=0.05,
            )
            for q in unconstrained
        ]
        record = {
            "yaw_deg": float(np.rad2deg(yaw)),
            "pregrasp_center": pregrasp_center.tolist(),
            "tcp_position": tcp_position.tolist(),
            "unconstrained_branches": int(len(unconstrained)),
            "safe_branches": int(len(safe)),
            "best_position_error_m": None if not unconstrained_diag else min(item.position_error_m for item in unconstrained_diag),
            "best_orientation_error_rad": None if not unconstrained_diag else min(item.orientation_error_rad for item in unconstrained_diag),
            "maximum_self_clearance_m": None if not self_values else max(self_values),
            "maximum_environment_clearance_m": None if not environment_values else max(environment_values),
            "branches_self_safe": int(sum(value >= 0.01 for value in self_values)),
            "branches_environment_safe": int(sum(value >= 0.01 for value in environment_values)),
        }
        records.append(record)
        print(record, flush=True)

    report = {"sample_id": str(row["sample_id"]), "object_center_foot0": object_center.tolist(), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
