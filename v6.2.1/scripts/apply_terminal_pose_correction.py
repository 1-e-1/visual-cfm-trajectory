"""Apply a measured task-space place residual as a pose-preserving IK correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.h1_1111_grasp.trajectory_safety import link_segments, self_clearance
from src.planning.ik_solver import solve_pose_ik_candidates
from src.robot.torch_kinematics import DifferentiableZBotFK


def _minimum_jerk(start: np.ndarray, goal: np.ndarray, points: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, points, dtype=np.float64)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None] + blend[:, None] * (goal - start)[None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--world_residual", type=float, nargs=3, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=129)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as source:
        plan = {key: np.array(source[key], copy=True) for key in source.files}
    q_old = np.asarray(plan["q_place"], dtype=np.float64)

    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to(args.device)
    with torch.no_grad():
        position, quaternion = fk(
            torch.as_tensor(q_old[None], dtype=torch.float32, device=args.device)
        )
    position = position[0].cpu().numpy()
    quaternion = quaternion[0].cpu().numpy()

    stage = Usd.Stage.Open(str(PROJECT_ROOT / "1111_copy.usd"))
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/UnitreeH1/ZBot6S_left/foot_0")
    ).ComputeLocalToWorldTransform(0)
    columns = []
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        direction = transform.TransformDir(Gf.Vec3d(*axis))
        columns.append([direction[0], direction[1], direction[2]])
    rotation_foot0_to_world = np.asarray(columns, dtype=np.float64).T
    world_residual = np.asarray(args.world_residual, dtype=np.float64)
    desired_world_correction = -world_residual
    foot0_correction = rotation_foot0_to_world.T @ desired_world_correction
    goal_position = position + foot0_correction

    candidates = solve_pose_ik_candidates(
        fk,
        goal_position,
        quaternion,
        n_candidates=8,
        restarts=192,
        iterations=400,
        position_tolerance_m=0.002,
        orientation_tolerance_rad=np.deg2rad(2.0),
        seed_configurations=q_old[None],
        seed=args.seed,
        device=args.device,
    )
    if not len(candidates):
        raise RuntimeError("terminal pose correction produced no IK candidate")

    ranked = sorted(
        candidates,
        key=lambda q: np.linalg.norm((q - q_old + np.pi) % (2.0 * np.pi) - np.pi),
    )
    selected = None
    selected_path = None
    selected_clearance = None
    for candidate in ranked:
        path = _minimum_jerk(q_old, np.asarray(candidate), args.points)
        clearance = self_clearance(
            link_segments(fk, path, device=args.device), capsule_radius_m=0.040
        )
        if clearance >= 0.01:
            selected = np.asarray(candidate, dtype=np.float32)
            selected_path = path.astype(np.float32)
            selected_clearance = float(clearance)
            break
    if selected is None:
        raise RuntimeError("all terminal correction paths failed self-collision screening")

    plan["q_place"] = selected
    plan["terminal_place_correction_q"] = selected_path
    plan["terminal_place_correction_duration"] = np.asarray(args.duration, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **plan)

    with torch.no_grad():
        final_position, final_quaternion = fk(
            torch.as_tensor(selected[None], dtype=torch.float32, device=args.device)
        )
    position_error = float(np.linalg.norm(final_position[0].cpu().numpy() - goal_position))
    report = {
        "source_plan": str(args.input),
        "output_plan": str(args.output),
        "measured_world_residual_m": world_residual.tolist(),
        "applied_world_correction_m": desired_world_correction.tolist(),
        "applied_foot0_correction_m": foot0_correction.tolist(),
        "candidate_count": int(len(candidates)),
        "selected_q_place": selected.tolist(),
        "periodic_joint_distance_rad": float(
            np.linalg.norm((selected - q_old + np.pi) % (2.0 * np.pi) - np.pi)
        ),
        "terminal_fk_position_error_m": position_error,
        "self_clearance_m": selected_clearance,
        "environment_clearance_recomputed": False,
        "environment_note": "The source point cloud was not archived in the old plan; PhysX replay is required.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
