"""Low-dimensional cubic-spline optimizer for expert joint trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:
    _trapezoid = np.trapz
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .trajectory_math import JointTrajectory, minimum_jerk_trajectory


@dataclass(frozen=True)
class ExpertCandidate:
    trajectory: JointTrajectory
    objective: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    planning_time_s: float
    initialization_index: int
    soft_limits_satisfied: bool


def _spline_trajectory(control_points: np.ndarray, duration: float, points: int) -> JointTrajectory:
    knot_u = np.linspace(0.0, 1.0, control_points.shape[0])
    sample_u = np.linspace(0.0, 1.0, points)
    spline = CubicSpline(knot_u, control_points, axis=0, bc_type=((1, np.zeros(6)), (1, np.zeros(6))))
    q = spline(sample_u)
    dq = spline(sample_u, 1) / duration
    ddq = spline(sample_u, 2) / duration**2
    dddq = spline(sample_u, 3) / duration**3
    q[0] = control_points[0]
    q[-1] = control_points[-1]
    trajectory = JointTrajectory(
        time=sample_u * duration,
        q=q,
        dq=dq,
        ddq=ddq,
        dddq=dddq,
        duration=float(duration),
        method="optimized_cubic_spline",
    )
    trajectory.validate(points)
    return trajectory


def optimize_expert_candidates(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    initial_duration: float,
    points: int,
    control_point_count: int,
    min_duration: float,
    max_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    joint_weights: np.ndarray,
    weights: dict[str, float],
    num_initializations: int,
    max_iterations: int,
    seed: int,
    initialization_indices: list[int] | None = None,
) -> list[ExpertCandidate]:
    q_start = np.asarray(q_start, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
    acceleration_limits = np.asarray(acceleration_limits, dtype=np.float64)
    jerk_limits = np.asarray(jerk_limits, dtype=np.float64)
    joint_weights = np.asarray(joint_weights, dtype=np.float64)
    if control_point_count < 4:
        raise ValueError("control_point_count must be at least 4")

    knot_u = np.linspace(0.0, 1.0, control_point_count)
    initial_curve = minimum_jerk_trajectory(q_start, q_goal, initial_duration, points=control_point_count)
    base_controls = initial_curve.q
    internal_shape = (control_point_count - 2, 6)
    trust_radius = 2.0 * np.pi

    def decode(x: np.ndarray) -> tuple[np.ndarray, float]:
        controls = np.vstack((q_start, x[:-1].reshape(internal_shape), q_goal))
        return controls, float(x[-1])

    def soft_limit_residuals(trajectory: JointTrajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.maximum(np.max(np.abs(trajectory.dq), axis=0) / velocity_limits - 1.0, 0.0),
            np.maximum(np.max(np.abs(trajectory.ddq), axis=0) / acceleration_limits - 1.0, 0.0),
            np.maximum(np.max(np.abs(trajectory.dddq), axis=0) / jerk_limits - 1.0, 0.0),
        )

    def objective(x: np.ndarray) -> float:
        controls, duration = decode(x)
        trajectory = _spline_trajectory(controls, duration, points)
        dt = trajectory.time
        joint_length = np.sum(np.linalg.norm(np.diff(trajectory.q, axis=0), axis=1))
        motion_proxy = _trapezoid(np.sum(joint_weights[None, :] * trajectory.dq**2, axis=1), dt)
        velocity_cost = _trapezoid(np.sum(trajectory.dq**2, axis=1), dt)
        acceleration_cost = _trapezoid(np.sum(trajectory.ddq**2, axis=1), dt)
        jerk_cost = _trapezoid(np.sum(trajectory.dddq**2, axis=1), dt)
        residuals = soft_limit_residuals(trajectory)
        constraint_penalty = 1.0e4 * sum(float(np.sum(item**2)) for item in residuals)
        return float(
            weights["w_joint_length"] * joint_length
            + weights["w_motion_proxy"] * motion_proxy
            + weights["w_velocity"] * velocity_cost
            + weights["w_acceleration"] * acceleration_cost
            + weights["w_jerk"] * jerk_cost
            + weights["w_duration"] * duration
            + constraint_penalty
        )

    candidates = []
    indices = initialization_indices if initialization_indices is not None else list(range(num_initializations))
    for init_index in indices:
        if init_index < 0 or init_index >= num_initializations:
            raise ValueError(f"initialization index {init_index} is outside [0, {num_initializations})")
        controls = base_controls.copy()
        if init_index > 0:
            init_rng = np.random.default_rng(np.random.SeedSequence([seed, init_index]))
            controls[1:-1] += init_rng.normal(0.0, 0.15 * init_index, size=internal_shape)
        x0 = np.concatenate((controls[1:-1].reshape(-1), [initial_duration]))
        duration_lower_bound = max(min_duration, initial_duration)
        lower = np.concatenate((base_controls[1:-1].reshape(-1) - trust_radius, [duration_lower_bound]))
        upper = np.concatenate((base_controls[1:-1].reshape(-1) + trust_radius, [max_duration]))
        started = perf_counter()
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            options={"maxiter": max_iterations, "ftol": 1.0e-7, "disp": False},
        )
        elapsed = perf_counter() - started
        optimized_controls, optimized_duration = decode(result.x)
        trajectory = _spline_trajectory(optimized_controls, optimized_duration, points)
        residuals = soft_limit_residuals(trajectory)
        limits_ok = all(np.max(item) <= 1.0e-5 for item in residuals)
        candidates.append(
            ExpertCandidate(
                trajectory=trajectory,
                objective=float(result.fun),
                optimizer_success=bool(result.success),
                optimizer_status=int(result.status),
                optimizer_message=str(result.message),
                iterations=int(result.nit),
                planning_time_s=elapsed,
                initialization_index=init_index,
                soft_limits_satisfied=limits_ok,
            )
        )
    return sorted(candidates, key=lambda item: item.objective)
