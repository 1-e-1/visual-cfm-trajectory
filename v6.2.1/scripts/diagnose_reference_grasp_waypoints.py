"""Check local pose reachability around the calibrated reference grasp."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.h1_1111_grasp.trajectory_safety import link_segments, self_clearance
from src.planning.grasp_pose import (
    grasp_center_pose_to_tcp_pose,
    quaternion_to_matrix_wxyz,
    tcp_pose_to_grasp_center_pose,
)
from src.planning.ik_solver import solve_pose_ik_candidates
from src.robot.torch_kinematics import DifferentiableZBotFK


def main() -> None:
    device = "cuda:0"
    q_contact = np.array([-2.1070001, 0.96, 0.597, 0.476, 0.098, -0.029], np.float32)
    offset = np.array([0.0000126, -0.0530952, 0.1405786])
    offset_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    fk = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json").to(device)
    with torch.no_grad():
        tcp_position_t, quaternion_t = fk(torch.as_tensor(q_contact[None], device=device))
    tcp_position = tcp_position_t[0].cpu().numpy()
    quaternion = quaternion_t[0].cpu().numpy()
    center, _ = tcp_pose_to_grasp_center_pose(
        tcp_position, quaternion, offset, offset_quaternion
    )
    forward = quaternion_to_matrix_wxyz(quaternion) @ offset
    forward /= np.linalg.norm(forward)
    targets = {
        "contact": center,
        "pregrasp": center - forward * 0.08,
        "lift": center + np.array([-0.10, 0.0, 0.0]),
        "preplace": center + np.array([-0.08, 0.0, 0.0]),
    }
    rng = np.random.default_rng(20260815)
    local_seeds = q_contact[None] + rng.normal(0.0, 0.12, size=(128, 6)).astype(np.float32)
    local_seeds[0] = q_contact
    records = {}
    for index, (name, grasp_center) in enumerate(targets.items()):
        goal_position, goal_quaternion = grasp_center_pose_to_tcp_pose(
            grasp_center, quaternion, offset, offset_quaternion
        )
        unconstrained = solve_pose_ik_candidates(
            fk,
            goal_position,
            goal_quaternion,
            n_candidates=8,
            restarts=128,
            iterations=500,
            seed_configurations=local_seeds,
            seed=20260815 + index,
            device=device,
        )
        clearances = [
            self_clearance(
                link_segments(fk, q[None], device=device), capsule_radius_m=0.045
            )
            for q in unconstrained
        ]
        records[name] = {
            "grasp_center": grasp_center.tolist(),
            "tcp_position": goal_position.tolist(),
            "branches": int(len(unconstrained)),
            "self_clearances_m": clearances,
            "safe_branches": int(sum(value >= 0.01 for value in clearances)),
        }
        print(name, records[name], flush=True)
    output = ROOT / "outputs/reference_grasp_waypoint_diagnostic.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
