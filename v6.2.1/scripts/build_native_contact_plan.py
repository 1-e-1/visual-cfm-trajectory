"""Build a continuous native-contact plan from the frozen fingertip plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.h1_1111_grasp.trajectory_safety import link_segments, self_clearance
from src.planning.trajectory_math import minimum_jerk_duration, minimum_jerk_trajectory
from src.robot.torch_kinematics import DifferentiableZBotFK


def _wrap(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _unwrap_path(path: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    output = np.empty_like(values)
    output[0] = reference + _wrap(values[0] - reference)
    for index in range(1, len(values)):
        output[index] = output[index - 1] + _wrap(values[index] - output[index - 1])
    return output


def _bridge(
    start: np.ndarray,
    goal: np.ndarray,
    limits: dict[str, list[float]],
) -> tuple[np.ndarray, float]:
    continuous_goal = start + _wrap(goal - start)
    duration, valid, _ = minimum_jerk_duration(
        start,
        continuous_goal,
        np.asarray(limits["velocity"]),
        np.asarray(limits["acceleration"]),
        np.asarray(limits["jerk"]),
        min_duration=0.8,
        max_duration=30.0,
    )
    if not valid:
        raise RuntimeError("a native-contact continuity bridge violates dynamic limits")
    return minimum_jerk_trajectory(start, continuous_goal, duration, points=128).q, duration


def _resample_segments(
    segments: list[tuple[np.ndarray, float]], points: int
) -> tuple[np.ndarray, float]:
    merged_time: list[np.ndarray] = []
    merged_path: list[np.ndarray] = []
    offset = 0.0
    reference = np.asarray(segments[0][0][0], dtype=np.float64)
    for index, (path, duration) in enumerate(segments):
        continuous = _unwrap_path(path, reference)
        time = np.linspace(offset, offset + float(duration), len(continuous))
        if index:
            time = time[1:]
            continuous = continuous[1:]
        merged_time.append(time)
        merged_path.append(continuous)
        offset += float(duration)
        reference = continuous[-1]
    source_time = np.concatenate(merged_time)
    source_path = np.concatenate(merged_path)
    target_time = np.linspace(0.0, offset, points)
    result = np.stack(
        [np.interp(target_time, source_time, source_path[:, joint]) for joint in range(6)],
        axis=1,
    )
    return _wrap(result).astype(np.float32), offset


def _dynamic_metrics(path: np.ndarray, duration: float) -> dict[str, float]:
    dt = float(duration) / max(1, len(path) - 1)
    velocity = _wrap(np.diff(path, axis=0)) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    return {
        "maximum_periodic_step_rad": float(np.max(np.abs(_wrap(np.diff(path, axis=0))))),
        "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "maximum_jerk_rad_s3": float(np.max(np.abs(jerk))),
    }


def _smooth_cfm_transfer(
    start: np.ndarray,
    cfm_path: np.ndarray,
    goal: np.ndarray,
    limits: dict[str, list[float]],
    points: int,
    nominal_duration: float,
    geometry_control_points: int = 16,
) -> tuple[np.ndarray, float, int]:
    continuous_cfm = _unwrap_path(cfm_path, start)
    indices = np.unique(
        np.round(
            np.linspace(
                0,
                len(continuous_cfm) - 1,
                min(int(geometry_control_points), len(continuous_cfm)),
            )
        ).astype(int)
    )
    continuous_goal = continuous_cfm[-1] + _wrap(goal - continuous_cfm[-1])
    controls = np.vstack((start, continuous_cfm[indices], continuous_goal))
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(controls, axis=0), axis=1) > 1.0e-8)
    )
    controls = controls[keep]
    chord = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(controls, axis=0), axis=1)))
    )
    parameter = chord / chord[-1]
    spline = CubicSpline(
        parameter,
        controls,
        axis=0,
        bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
    )
    sample_parameter = np.linspace(0.0, 1.0, points)
    path = spline(sample_parameter)
    velocity_limit = np.asarray(limits["velocity"], dtype=np.float64)
    acceleration_limit = np.asarray(limits["acceleration"], dtype=np.float64)
    jerk_limit = np.asarray(limits["jerk"], dtype=np.float64)
    derivative_1 = np.max(np.abs(spline(sample_parameter, 1)), axis=0)
    derivative_2 = np.max(np.abs(spline(sample_parameter, 2)), axis=0)
    derivative_3 = np.max(np.abs(spline(sample_parameter, 3)), axis=0)
    required_duration = max(
        float(nominal_duration),
        float(np.max(derivative_1 / velocity_limit)),
        float(np.max(np.sqrt(derivative_2 / acceleration_limit))),
        float(np.max(np.cbrt(derivative_3 / jerk_limit))),
    )
    duration = 1.2 * required_duration
    return _wrap(path).astype(np.float32), duration, len(controls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "outputs/pose_grasp_place_plan_true_fingertips_lift.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_plan.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/native_contact/native_contact_plan.json",
    )
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as source:
        payload = {key: np.array(source[key], copy=True) for key in source.files}
    robot_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/robot.yaml").read_text(encoding="utf-8")
    )
    limits = robot_cfg["planning_soft_limits"]

    q_lift = np.asarray(payload["q_lift"], dtype=np.float64)
    q_preplace = np.asarray(payload["q_preplace"], dtype=np.float64)
    original_path = np.asarray(payload["transfer_to_preplace_q"], dtype=np.float64)
    original_duration = float(payload["transfer_to_preplace_duration"])
    original_start_error = float(np.linalg.norm(_wrap(original_path[0] - q_lift)))
    original_end_error = float(np.linalg.norm(_wrap(original_path[-1] - q_preplace)))
    repaired_path, repaired_duration, control_count = _smooth_cfm_transfer(
        q_lift,
        original_path,
        q_preplace,
        limits,
        points=args.points,
        nominal_duration=original_duration,
    )
    payload["transfer_to_preplace_q"] = repaired_path
    payload["transfer_to_preplace_duration"] = np.float32(repaired_duration)

    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to(
        args.device
    )
    with torch.no_grad():
        segments = link_segments(fk, repaired_path, device=args.device)
    clearance = self_clearance(segments, capsule_radius_m=0.040)
    if clearance < 0.01:
        raise RuntimeError(f"repaired transfer self-clearance is only {clearance:.6f} m")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "method": "Visual CFM geometry control points with clamped cubic smoothing and dynamic time scaling",
        "points": int(len(repaired_path)),
        "geometry_control_points": int(control_count),
        "original_duration_s": original_duration,
        "total_duration_s": float(repaired_duration),
        "original_start_error_rad": original_start_error,
        "original_end_error_rad": original_end_error,
        "start_error_rad": float(np.linalg.norm(_wrap(repaired_path[0] - q_lift))),
        "end_error_rad": float(np.linalg.norm(_wrap(repaired_path[-1] - q_preplace))),
        "self_clearance_m": float(clearance),
        "dynamic_metrics": _dynamic_metrics(repaired_path, repaired_duration),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
