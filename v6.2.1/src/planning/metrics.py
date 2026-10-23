"""Trajectory metrics. Motion cost is a kinematic proxy, not real energy."""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:
    _trapezoid = np.trapz

from .trajectory_math import JointTrajectory


def trajectory_metrics(trajectory: JointTrajectory, joint_weights: np.ndarray) -> dict[str, float | list[float]]:
    weights = np.asarray(joint_weights, dtype=np.float64)
    if weights.shape != (6,) or np.any(weights < 0.0):
        raise ValueError("joint_weights must have shape (6,) and be non-negative")
    time = trajectory.time
    weighted_velocity_sq = np.sum(weights[None, :] * trajectory.dq**2, axis=1)
    acceleration_sq = np.sum(trajectory.ddq**2, axis=1)
    jerk_sq = np.sum(trajectory.dddq**2, axis=1)
    joint_path = np.sum(np.abs(np.diff(trajectory.q, axis=0)), axis=0)
    return {
        "duration_s": trajectory.duration,
        "joint_path_length_l1_rad": float(np.sum(joint_path)),
        "per_joint_path_length_rad": [float(v) for v in joint_path],
        "motion_cost_proxy_integral": float(_trapezoid(weighted_velocity_sq, time)),
        "acceleration_squared_integral": float(_trapezoid(acceleration_sq, time)),
        "jerk_squared_integral": float(_trapezoid(jerk_sq, time)),
        "max_abs_velocity_rad_s": float(np.max(np.abs(trajectory.dq))),
        "max_abs_acceleration_rad_s2": float(np.max(np.abs(trajectory.ddq))),
        "max_abs_jerk_rad_s3": float(np.max(np.abs(trajectory.dddq))),
    }
