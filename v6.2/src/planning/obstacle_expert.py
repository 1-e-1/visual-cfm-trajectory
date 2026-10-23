"""Time-aware obstacle task sampling and smooth joint-space expert planning."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.planning.expert_optimizer import _spline_trajectory
from src.planning.metrics import trajectory_metrics
from src.planning.trajectory_math import JointTrajectory, minimum_jerk_duration, minimum_jerk_trajectory


@dataclass(frozen=True)
class ObstacleExpertResult:
    trajectory: JointTrajectory
    validation: dict[str, Any]
    candidate_index: int
    candidates_generated: int
    candidates_kinematically_safe: int
    planning_time_s: float
    minimum_predicted_clearance_m: float


def _signed_distance_to_axis_aligned_box(
    points: torch.Tensor,
    centers: torch.Tensor,
    half_extents: torch.Tensor,
) -> torch.Tensor:
    delta = torch.abs(points - centers) - half_extents
    outside = torch.linalg.norm(torch.clamp_min(delta, 0.0), dim=-1)
    inside = torch.clamp_max(torch.amax(delta, dim=-1), 0.0)
    return outside + inside


def _trajectory_with_scaled_duration(
    controls: np.ndarray,
    points: int,
    min_duration: float,
    max_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
) -> JointTrajectory | None:
    unit = _spline_trajectory(controls, 1.0, points)
    scale = max(
        float(np.max(np.abs(unit.dq) / velocity_limits[None, :])),
        float(np.sqrt(np.max(np.abs(unit.ddq) / acceleration_limits[None, :]))),
        float(np.cbrt(np.max(np.abs(unit.dddq) / jerk_limits[None, :]))),
        min_duration,
    )
    duration = max(min_duration, scale * 1.03)
    if duration > max_duration:
        return None
    return _spline_trajectory(controls, duration, points)


def generate_smooth_candidates(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    points: int,
    control_points: int,
    count: int,
    min_duration: float,
    max_duration: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    seed: int,
) -> list[JointTrajectory]:
    """Create reproducible multi-homotopy cubic splines with exact endpoints."""
    base_duration, feasible, _ = minimum_jerk_duration(
        q_start,
        q_goal,
        velocity_limits,
        acceleration_limits,
        jerk_limits,
        min_duration,
        max_duration,
    )
    if not feasible:
        return []
    base = minimum_jerk_trajectory(q_start, q_goal, base_duration, points=control_points).q
    u = np.linspace(0.0, 1.0, control_points)
    bump_1 = np.sin(np.pi * u) ** 2
    bump_2 = np.sin(2.0 * np.pi * u) * np.sin(np.pi * u) ** 2
    rng = np.random.default_rng(seed)
    offsets = [np.zeros((2, 6), dtype=np.float64)]
    for joint in range(6):
        for sign in (-1.0, 1.0):
            primary = np.zeros(6)
            primary[joint] = sign * 0.9
            offsets.append(np.stack((primary, np.zeros(6))))
    while len(offsets) < count:
        primary = rng.normal(size=6)
        primary /= max(float(np.linalg.norm(primary)), 1.0e-12)
        primary *= rng.uniform(0.45, 2.5)
        secondary = rng.normal(size=6)
        secondary /= max(float(np.linalg.norm(secondary)), 1.0e-12)
        secondary *= rng.uniform(0.0, 0.7)
        offsets.append(np.stack((primary, secondary)))

    trajectories = []
    for primary, secondary in offsets[:count]:
        controls = base + bump_1[:, None] * primary[None, :] + bump_2[:, None] * secondary[None, :]
        controls[0] = q_start
        controls[-1] = q_goal
        trajectory = _trajectory_with_scaled_duration(
            controls,
            points,
            min_duration,
            max_duration,
            velocity_limits,
            acceleration_limits,
            jerk_limits,
        )
        if trajectory is not None:
            trajectories.append(trajectory)
    return trajectories


def predicted_clearance(
    fk_model: Any,
    trajectory: JointTrajectory,
    static_position: np.ndarray,
    static_size: np.ndarray,
    dynamic_spec: dict[str, Any],
    link_radius_m: float,
    device: str,
) -> float:
    q = torch.as_tensor(trajectory.q, dtype=torch.float32, device=device)
    with torch.no_grad():
        link_positions, _ = fk_model.link_poses(q)
    time = torch.as_tensor(trajectory.time, dtype=torch.float32, device=device)
    static_center = torch.as_tensor(static_position, dtype=torch.float32, device=device)
    static_half = 0.5 * torch.as_tensor(static_size, dtype=torch.float32, device=device)
    axis = torch.as_tensor(dynamic_spec["axis_xyz"], dtype=torch.float32, device=device)
    axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-12)
    omega = 2.0 * np.pi * float(dynamic_spec["frequency_hz"])
    phase = omega * time + float(dynamic_spec["phase_rad"])
    dynamic_center = torch.as_tensor(dynamic_spec["center_m"], dtype=torch.float32, device=device)
    dynamic_positions = dynamic_center[None, :] + (
        axis[None, :] * float(dynamic_spec["amplitude_m"]) * torch.sin(phase)[:, None]
    )
    dynamic_half = 0.5 * torch.as_tensor(dynamic_spec["size_m"], dtype=torch.float32, device=device)
    static_distance = _signed_distance_to_axis_aligned_box(
        link_positions,
        static_center[None, None, :],
        static_half[None, None, :],
    )
    dynamic_distance = _signed_distance_to_axis_aligned_box(
        link_positions,
        dynamic_positions[:, None, :],
        dynamic_half[None, None, :],
    )
    return float(torch.minimum(static_distance.min(), dynamic_distance.min()).item() - link_radius_m)


def _obstacle_positions_torch(
    spec: dict[str, Any],
    time: torch.Tensor,
    device: str,
) -> torch.Tensor:
    motion_model = spec.get("motion_model", "static")
    if motion_model == "static":
        position = torch.as_tensor(spec["position_m"], dtype=torch.float32, device=device)
        return position[None, :].expand(len(time), -1)
    center = torch.as_tensor(spec["center_m"], dtype=torch.float32, device=device)
    axis = torch.as_tensor(spec["axis_xyz"], dtype=torch.float32, device=device)
    axis = axis / torch.linalg.norm(axis).clamp_min(1.0e-12)
    if motion_model == "sinusoidal":
        omega = 2.0 * np.pi * float(spec["frequency_hz"])
        phase = omega * time + float(spec["phase_rad"])
        return center[None, :] + axis[None, :] * float(spec["amplitude_m"]) * torch.sin(phase)[:, None]
    if motion_model == "linear_ping_pong":
        half_length = max(float(spec["corridor_half_length_m"]), 1.0e-6)
        speed = max(float(spec["speed_m_s"]), 1.0e-6)
        period = 4.0 * half_length / speed
        cycle = torch.remainder(
            time / period + float(spec.get("phase_rad", 0.0)) / (2.0 * np.pi),
            1.0,
        )
        displacement = torch.where(
            cycle < 0.25,
            4.0 * cycle,
            torch.where(cycle < 0.75, 2.0 - 4.0 * cycle, 4.0 * cycle - 4.0),
        )
        return center[None, :] + axis[None, :] * (half_length * displacement[:, None])
    raise ValueError(f"Unsupported obstacle motion model: {motion_model}")


def predicted_multi_obstacle_clearance(
    fk_model: Any,
    trajectory: JointTrajectory,
    obstacle_specs: list[dict[str, Any]],
    link_radius_m: float,
    device: str,
) -> float:
    """Conservative link-center clearance against arbitrary time-aware obstacle tokens."""
    q = torch.as_tensor(trajectory.q, dtype=torch.float32, device=device)
    with torch.no_grad():
        link_positions, _ = fk_model.link_poses(q)
    time = torch.as_tensor(trajectory.time, dtype=torch.float32, device=device)
    minimum = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    for spec in obstacle_specs:
        centers = _obstacle_positions_torch(spec, time, device)
        half_extents = 0.5 * torch.as_tensor(spec["size_m"], dtype=torch.float32, device=device)
        distance = _signed_distance_to_axis_aligned_box(
            link_positions,
            centers[:, None, :],
            half_extents[None, None, :],
        )
        minimum = torch.minimum(minimum, distance.min())
    return float(minimum.item() - link_radius_m)


def plan_obstacle_expert(
    scene: Any,
    fk_model: Any,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    static_position: np.ndarray,
    static_size: np.ndarray,
    dynamic_spec: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    robot_cfg: dict[str, Any],
    candidate_count: int,
    seed: int,
    device: str,
) -> ObstacleExpertResult:
    started = perf_counter()
    trajectory_cfg = optimizer_cfg["trajectory"]
    soft_limits = robot_cfg["planning_soft_limits"]
    trajectories = generate_smooth_candidates(
        q_start=q_start,
        q_goal=q_goal,
        points=int(trajectory_cfg["points"]),
        control_points=max(10, int(optimizer_cfg["optimization"]["bspline_control_points"])),
        count=candidate_count,
        min_duration=float(trajectory_cfg["min_duration_s"]),
        max_duration=float(trajectory_cfg["max_duration_s"]),
        velocity_limits=np.asarray(soft_limits["velocity"], dtype=np.float64),
        acceleration_limits=np.asarray(soft_limits["acceleration"], dtype=np.float64),
        jerk_limits=np.asarray(soft_limits["jerk"], dtype=np.float64),
        seed=seed,
    )
    link_radius = 0.055
    safe_candidates = []
    joint_weights = np.asarray(optimizer_cfg["motion_cost_proxy"]["joint_weights"], dtype=np.float64)
    for candidate_index, trajectory in enumerate(trajectories):
        clearance = predicted_clearance(
            fk_model,
            trajectory,
            static_position,
            static_size,
            dynamic_spec,
            link_radius,
            device,
        )
        if clearance < 0.008:
            continue
        metrics = trajectory_metrics(trajectory, joint_weights)
        score = (
            metrics["joint_path_length_l1_rad"]
            + 0.4 * metrics["motion_cost_proxy_integral"]
            + 2.0 * trajectory.duration
            - 0.2 * min(clearance, 0.20)
        )
        safe_candidates.append((score, candidate_index, clearance, trajectory))
    safe_candidates.sort(key=lambda item: item[0])
    for _, candidate_index, clearance, trajectory in safe_candidates:
        validation = scene.validate_timed_path(
            trajectory.q,
            trajectory.time,
            settle_steps=1,
            stop_on_collision=True,
        )
        if validation["collision_free"]:
            return ObstacleExpertResult(
                trajectory=trajectory,
                validation=validation,
                candidate_index=candidate_index,
                candidates_generated=len(trajectories),
                candidates_kinematically_safe=len(safe_candidates),
                planning_time_s=perf_counter() - started,
                minimum_predicted_clearance_m=clearance,
            )
    raise RuntimeError(
        f"No obstacle-free expert: generated={len(trajectories)}, "
        f"predicted_safe={len(safe_candidates)}"
    )


def plan_randomized_obstacle_expert(
    scene: Any,
    fk_model: Any,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    obstacle_specs: list[dict[str, Any]],
    optimizer_cfg: dict[str, Any],
    robot_cfg: dict[str, Any],
    candidate_count: int,
    seed: int,
    device: str,
) -> ObstacleExpertResult:
    """Plan and PhysX-validate an expert path against shape-randomized space obstacles."""
    started = perf_counter()
    trajectory_cfg = optimizer_cfg["trajectory"]
    soft_limits = robot_cfg["planning_soft_limits"]
    trajectories = generate_smooth_candidates(
        q_start=q_start,
        q_goal=q_goal,
        points=int(trajectory_cfg["points"]),
        control_points=max(10, int(optimizer_cfg["optimization"]["bspline_control_points"])),
        count=candidate_count,
        min_duration=float(trajectory_cfg["min_duration_s"]),
        max_duration=float(trajectory_cfg["max_duration_s"]),
        velocity_limits=np.asarray(soft_limits["velocity"], dtype=np.float64),
        acceleration_limits=np.asarray(soft_limits["acceleration"], dtype=np.float64),
        jerk_limits=np.asarray(soft_limits["jerk"], dtype=np.float64),
        seed=seed,
    )
    joint_weights = np.asarray(optimizer_cfg["motion_cost_proxy"]["joint_weights"], dtype=np.float64)
    safe_candidates = []
    for candidate_index, trajectory in enumerate(trajectories):
        clearance = predicted_multi_obstacle_clearance(
            fk_model,
            trajectory,
            obstacle_specs,
            link_radius_m=0.055,
            device=device,
        )
        if clearance < 0.008:
            continue
        metrics = trajectory_metrics(trajectory, joint_weights)
        score = (
            metrics["joint_path_length_l1_rad"]
            + 0.4 * metrics["motion_cost_proxy_integral"]
            + 2.0 * trajectory.duration
            - 0.2 * min(clearance, 0.20)
        )
        safe_candidates.append((score, candidate_index, clearance, trajectory))
    safe_candidates.sort(key=lambda item: item[0])
    for _, candidate_index, clearance, trajectory in safe_candidates:
        validation = scene.validate_timed_path(
            trajectory.q,
            trajectory.time,
            settle_steps=1,
            stop_on_collision=True,
        )
        if validation["collision_free"]:
            return ObstacleExpertResult(
                trajectory=trajectory,
                validation=validation,
                candidate_index=candidate_index,
                candidates_generated=len(trajectories),
                candidates_kinematically_safe=len(safe_candidates),
                planning_time_s=perf_counter() - started,
                minimum_predicted_clearance_m=clearance,
            )
    raise RuntimeError(
        f"No randomized-obstacle expert: generated={len(trajectories)}, "
        f"predicted_safe={len(safe_candidates)}"
    )


def sample_blocking_obstacles(
    fk_model: Any,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    obstacle_cfg: dict[str, Any],
    rng: np.random.Generator,
    device: str,
) -> dict[str, Any]:
    """Place obstacles along the direct path while keeping every task reproducible."""
    direct = minimum_jerk_trajectory(q_start, q_goal, duration, points=64)
    with torch.no_grad():
        tcp_positions, _ = fk_model(torch.as_tensor(direct.q, dtype=torch.float32, device=device))
    tcp = tcp_positions.detach().cpu().numpy()
    workspace = obstacle_cfg["scene"]["workspace"]
    challenge_mode = rng.choice(["static", "dynamic", "both"], p=[0.4, 0.4, 0.2])
    lateral_angle = rng.uniform(0.0, 2.0 * np.pi)
    lateral = np.asarray([np.cos(lateral_angle), np.sin(lateral_angle), 0.0], dtype=np.float64)
    static_offset = 0.0 if challenge_mode in {"static", "both"} else rng.uniform(0.14, 0.22)
    static_position = tcp[21] + static_offset * lateral + rng.uniform(-0.02, 0.02, size=3)
    static_position = np.clip(
        static_position,
        [workspace["x"][0], workspace["y"][0], workspace["z"][0]],
        [workspace["x"][1], workspace["y"][1], workspace["z"][1]],
    )
    angle = rng.uniform(0.0, 2.0 * np.pi)
    axis = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    amplitude = rng.uniform(0.12, 0.24)
    frequency = rng.uniform(0.12, 0.20)
    phase = 0.0
    target_time = direct.time[42]
    dynamic_offset = 0.0 if challenge_mode in {"dynamic", "both"} else rng.uniform(0.14, 0.22)
    target_position = tcp[42] - dynamic_offset * lateral + rng.uniform(-0.02, 0.02, size=3)
    center = target_position - axis * amplitude * np.sin(2.0 * np.pi * frequency * target_time + phase)
    dynamic_default = obstacle_cfg["dynamic_obstacles"]["default"]
    return {
        "challenge_mode": challenge_mode,
        "static_position_m": static_position,
        "static_size_m": np.asarray(obstacle_cfg["static_obstacles"]["default"]["size_m"], dtype=np.float64),
        "dynamic": {
            "size_m": np.asarray(dynamic_default["size_m"], dtype=np.float64),
            "center_m": center,
            "axis_xyz": axis,
            "amplitude_m": amplitude,
            "frequency_hz": frequency,
            "phase_rad": phase,
            "quaternion_wxyz": np.asarray(dynamic_default["quaternion_wxyz"], dtype=np.float64),
        },
    }


def _numpy_motion_position(spec: dict[str, Any], time_s: float) -> np.ndarray:
    motion_model = spec["motion_model"]
    if motion_model == "static":
        return np.asarray(spec["position_m"], dtype=np.float64)
    center = np.asarray(spec["center_m"], dtype=np.float64)
    axis = np.asarray(spec["axis_xyz"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    if motion_model == "sinusoidal":
        phase = 2.0 * np.pi * float(spec["frequency_hz"]) * time_s + float(spec["phase_rad"])
        return center + axis * float(spec["amplitude_m"]) * np.sin(phase)
    if motion_model == "linear_ping_pong":
        half_length = float(spec["corridor_half_length_m"])
        speed = float(spec["speed_m_s"])
        period = 4.0 * half_length / speed
        cycle = (time_s / period + float(spec["phase_rad"]) / (2.0 * np.pi)) % 1.0
        if cycle < 0.25:
            displacement = 4.0 * cycle
        elif cycle < 0.75:
            displacement = 2.0 - 4.0 * cycle
        else:
            displacement = 4.0 * cycle - 4.0
        return center + axis * half_length * displacement
    raise ValueError(f"Unsupported obstacle motion model: {motion_model}")


def _scene_passes_geometric_filters(
    instances: list[dict[str, Any]],
    q_start_tcp: np.ndarray,
    q_goal_tcp: np.ndarray,
    duration: float,
    obstacle_cfg: dict[str, Any],
) -> bool:
    workspace = obstacle_cfg["scene"]["workspace"]
    lower = np.asarray([workspace["x"][0], workspace["y"][0], workspace["z"][0]], dtype=np.float64)
    upper = np.asarray([workspace["x"][1], workspace["y"][1], workspace["z"][1]], dtype=np.float64)
    workspace_volume = float(np.prod(upper - lower))
    occupied_volume = sum(float(np.prod(item["size_m"])) for item in instances)
    task_cfg = obstacle_cfg["task_sampling"]
    if occupied_volume / workspace_volume > float(task_cfg["max_workspace_occupancy_ratio"]):
        return False
    max_diameter = float(task_cfg["max_obstacle_diameter_m"])
    endpoint_clearance = float(task_cfg["min_tcp_obstacle_clearance_m"])
    base_exclusion = float(obstacle_cfg["scene"]["base_exclusion_radius_m"])
    for item in instances:
        size = np.asarray(item["size_m"], dtype=np.float64)
        if float(size.max()) > max_diameter:
            return False
        bounding_radius = 0.5 * float(np.linalg.norm(size))
        sample_times = np.linspace(0.0, duration, 25)
        positions = np.stack([_numpy_motion_position(item, float(time_s)) for time_s in sample_times])
        if np.any(positions < lower[None, :] + bounding_radius) or np.any(
            positions > upper[None, :] - bounding_radius
        ):
            return False
        near_base_height = positions[:, 2] - bounding_radius < 0.16
        if np.any(near_base_height) and float(
            np.min(np.linalg.norm(positions[near_base_height, :2], axis=1))
        ) < base_exclusion + bounding_radius:
            return False
        start_clearance = np.linalg.norm(positions[0] - q_start_tcp) - bounding_radius
        goal_clearance = np.linalg.norm(positions[-1] - q_goal_tcp) - bounding_radius
        if min(start_clearance, goal_clearance) < endpoint_clearance:
            return False
    return True


def sample_randomized_space_obstacles(
    fk_model: Any,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    obstacle_cfg: dict[str, Any],
    rng: np.random.Generator,
    device: str,
) -> dict[str, Any]:
    """Sample bounded shape variants and 3-D flyby debris with solvability pre-filters."""
    direct = minimum_jerk_trajectory(q_start, q_goal, duration, points=64)
    with torch.no_grad():
        tcp_positions, _ = fk_model(torch.as_tensor(direct.q, dtype=torch.float32, device=device))
    tcp = tcp_positions.detach().cpu().numpy()
    variants = obstacle_cfg["shape_pool"]["variants"]
    by_role = {
        role: [item for item in variants if item["role"] == role]
        for role in ("static", "dynamic", "debris")
    }
    workspace = obstacle_cfg["scene"]["workspace"]
    lower = np.asarray([workspace["x"][0], workspace["y"][0], workspace["z"][0]], dtype=np.float64)
    upper = np.asarray([workspace["x"][1], workspace["y"][1], workspace["z"][1]], dtype=np.float64)
    max_attempts = int(obstacle_cfg["task_sampling"]["max_scene_resample_attempts"])
    debris_cfg = obstacle_cfg["space_debris"]
    dynamic_cfg = obstacle_cfg["dynamic_obstacles"]
    for _ in range(max_attempts):
        static_variant = dict(by_role["static"][int(rng.integers(len(by_role["static"])))])
        dynamic_variant = dict(by_role["dynamic"][int(rng.integers(len(by_role["dynamic"])))])
        lateral_angle = rng.uniform(0.0, 2.0 * np.pi)
        lateral = np.asarray([np.cos(lateral_angle), np.sin(lateral_angle), 0.0], dtype=np.float64)
        static_position = np.clip(
            tcp[22] + lateral * rng.uniform(0.06, 0.13) + rng.uniform(-0.015, 0.015, size=3),
            lower + 0.5 * np.asarray(static_variant["size_m"], dtype=np.float64),
            upper - 0.5 * np.asarray(static_variant["size_m"], dtype=np.float64),
        )
        dynamic_angle = rng.uniform(0.0, 2.0 * np.pi)
        dynamic_axis = np.asarray([np.cos(dynamic_angle), np.sin(dynamic_angle), 0.0], dtype=np.float64)
        amplitude = rng.uniform(0.10, 0.18)
        frequency = rng.uniform(0.12, 0.20)
        target_time = float(direct.time[42])
        target_position = tcp[42] - lateral * rng.uniform(0.05, 0.12)
        dynamic_center = target_position - dynamic_axis * amplitude * np.sin(
            2.0 * np.pi * frequency * target_time
        )
        dynamic_radius = 0.5 * float(np.linalg.norm(dynamic_variant["size_m"]))
        dynamic_margin = np.abs(dynamic_axis) * amplitude + dynamic_radius
        dynamic_center = np.clip(dynamic_center, lower + dynamic_margin, upper - dynamic_margin)
        instances: list[dict[str, Any]] = [
            {
                "pool_id": static_variant["id"],
                "type": "static",
                "shape": static_variant["shape"],
                "size_m": np.asarray(static_variant["size_m"], dtype=np.float64),
                "position_m": static_position,
                "quaternion_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                "motion_model": "static",
            },
            {
                "pool_id": dynamic_variant["id"],
                "type": "dynamic",
                "shape": dynamic_variant["shape"],
                "size_m": np.asarray(dynamic_variant["size_m"], dtype=np.float64),
                "center_m": dynamic_center,
                "axis_xyz": dynamic_axis,
                "amplitude_m": amplitude,
                "frequency_hz": frequency,
                "phase_rad": 0.0,
                "quaternion_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                "motion_model": "sinusoidal",
            },
        ]
        debris_count = int(rng.integers(int(debris_cfg["count_range"][0]), int(debris_cfg["count_range"][1]) + 1))
        debris_variants = rng.choice(by_role["debris"], size=debris_count, replace=False)
        for debris_index, variant in enumerate(debris_variants):
            axis = rng.normal(size=3)
            axis[2] *= 0.65
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            half_length = rng.uniform(*debris_cfg["corridor_half_length_range_m"])
            speed = rng.uniform(*debris_cfg["speed_range_m_s"])
            crossing_index = 30 + 8 * debris_index
            crossing_time = float(direct.time[crossing_index])
            center = tcp[crossing_index] + lateral * rng.uniform(0.07, 0.14)
            debris_radius = 0.5 * float(np.linalg.norm(variant["size_m"]))
            debris_margin = np.abs(axis) * half_length + debris_radius
            center = np.clip(center, lower + debris_margin, upper - debris_margin)
            period = 4.0 * half_length / speed
            phase_fraction = (-crossing_time / period) % 1.0
            instances.append(
                {
                    "pool_id": variant["id"],
                    "type": "debris",
                    "shape": variant["shape"],
                    "size_m": np.asarray(variant["size_m"], dtype=np.float64),
                    "center_m": center,
                    "axis_xyz": axis,
                    "corridor_half_length_m": half_length,
                    "speed_m_s": speed,
                    "phase_rad": 2.0 * np.pi * phase_fraction,
                    "quaternion_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "motion_model": "linear_ping_pong",
                }
            )
        if _scene_passes_geometric_filters(instances, tcp[0], tcp[-1], duration, obstacle_cfg):
            return {
                "challenge_mode": "space_mixed",
                "instances": instances,
                "active_count": len(instances),
                "workspace_occupancy_ratio": sum(float(np.prod(item["size_m"])) for item in instances)
                / float(np.prod(upper - lower)),
            }
    raise RuntimeError("randomized_space_obstacle_sampling_exhausted")
