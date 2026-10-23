"""Camera-derived obstacle reconstruction and trajectory clearance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CameraObstacle:
    center_m: np.ndarray
    size_m: np.ndarray
    velocity_m_s: np.ndarray
    point_count: int

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        for key in ("center_m", "size_m", "velocity_m_s"):
            report[key] = np.asarray(report[key]).tolist()
        return report


def look_at_rotation(position_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    """Return the ROS optical-frame to world rotation used by the fixed cameras."""
    forward = np.asarray(target_m, dtype=np.float64) - np.asarray(position_m, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    down = np.cross(forward, right)
    down /= max(float(np.linalg.norm(down)), 1.0e-12)
    return np.column_stack((right, down, forward))


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3)
    indices = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.zeros(len(sums), dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    return sums / counts[:, None]


def _point_segment_distances(
    points: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> np.ndarray:
    direction = segment_end - segment_start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-12:
        return np.linalg.norm(points - segment_start[None, :], axis=1)
    fraction = np.clip(
        ((points - segment_start[None, :]) @ direction) / denominator,
        0.0,
        1.0,
    )
    closest = segment_start[None, :] + fraction[:, None] * direction[None, :]
    return np.linalg.norm(points - closest, axis=1)


def remove_robot_points(
    points: np.ndarray,
    link_positions_m: np.ndarray,
    radius_m: float = 0.065,
) -> np.ndarray:
    """Remove points close to the known current robot kinematic chain."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    links = np.asarray(link_positions_m, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(links) == 0:
        return points
    chain = np.concatenate((np.zeros((1, 3), dtype=np.float64), links), axis=0)
    minimum = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in zip(chain[:-1], chain[1:]):
        minimum = np.minimum(minimum, _point_segment_distances(points, start, end))
    return points[minimum > float(radius_m)]


def _connected_components(
    points: np.ndarray,
    connection_radius_m: float,
    min_points: int,
) -> list[np.ndarray]:
    if len(points) < min_points:
        return []
    parent = np.arange(len(points), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in cKDTree(points).query_pairs(float(connection_radius_m)):
        union(int(left), int(right))
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    return [
        points[np.asarray(indices, dtype=np.int64)]
        for indices in groups.values()
        if len(indices) >= int(min_points)
    ]


def _fixed_view_points(
    depth: np.ndarray,
    semantic: np.ndarray,
    view_names: list[str],
    obstacle_cfg: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    camera_cfg = obstacle_cfg["camera"]
    pinhole = camera_cfg["pinhole"]
    width = int(camera_cfg["width"])
    height = int(camera_cfg["height"])
    focal_pixels = (
        float(pinhole["focal_length_mm"])
        / float(pinhole["horizontal_aperture_mm"])
        * width
    )
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    clip_min, clip_max = (float(value) for value in pinhole["clipping_range_m"])
    fixed_cfg = obstacle_cfg["fixed_cameras"]["cameras"]
    world_points: list[np.ndarray] = []
    for view_index, view_name in enumerate(view_names):
        if view_name == "wrist" or view_name not in fixed_cfg:
            continue
        item = fixed_cfg[view_name]
        position = np.asarray(item["position_m"], dtype=np.float64)
        rotation = look_at_rotation(position, np.asarray(item["look_at_m"], dtype=np.float64))
        depth_image = np.asarray(depth[view_index, frame_index, ..., 0], dtype=np.float64)
        semantic_image = np.asarray(semantic[view_index, frame_index, ..., 0])
        valid = (
            np.isfinite(depth_image)
            & (depth_image > clip_min)
            & (depth_image < clip_max - 1.0e-3)
            # Isaac Sim assigns 0 to background and 1 to the robot in this scene.
            & (semantic_image > 1)
        )
        if not np.any(valid):
            continue
        rows, columns = np.indices(depth_image.shape)
        z = depth_image[valid]
        x = (columns[valid] - center_x) * z / focal_pixels
        y = (rows[valid] - center_y) * z / focal_pixels
        camera_points = np.stack((x, y, z), axis=-1)
        world_points.append(camera_points @ rotation.T + position[None, :])
    if not world_points:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(world_points, axis=0)


def _cluster_obstacle_points(
    points: np.ndarray,
    link_positions_m: np.ndarray,
    obstacle_cfg: dict[str, Any],
    voxel_size_m: float,
    connection_radius_m: float,
    min_cluster_points: int,
) -> list[np.ndarray]:
    workspace = obstacle_cfg["scene"]["workspace"]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = (
        (points[:, 0] >= float(workspace["x"][0]))
        & (points[:, 0] <= float(workspace["x"][1]))
        & (points[:, 1] >= float(workspace["y"][0]))
        & (points[:, 1] <= float(workspace["y"][1]))
        & (points[:, 2] >= float(workspace["z"][0]))
        & (points[:, 2] <= float(workspace["z"][1]))
    )
    points = remove_robot_points(points[inside], link_positions_m)
    points = _voxel_downsample(points, voxel_size_m)
    return _connected_components(points, connection_radius_m, min_cluster_points)


def reconstruct_fixed_camera_obstacles(
    depth: np.ndarray,
    semantic: np.ndarray,
    view_names: list[str],
    link_positions_m: np.ndarray,
    obstacle_cfg: dict[str, Any],
    voxel_size_m: float = 0.018,
    connection_radius_m: float = 0.055,
    min_cluster_points: int = 4,
    inflation_m: float = 0.025,
) -> tuple[list[CameraObstacle], dict[str, Any]]:
    """Reconstruct moving AABBs from two frames of the three fixed cameras."""
    frame_interval_s = float(obstacle_cfg["camera"]["frame_interval_s"])
    frame_clusters: list[list[np.ndarray]] = []
    for frame_index in (0, 1):
        points = _fixed_view_points(
            depth,
            semantic,
            view_names,
            obstacle_cfg,
            frame_index,
        )
        frame_clusters.append(
            _cluster_obstacle_points(
                points,
                link_positions_m,
                obstacle_cfg,
                voxel_size_m,
                connection_radius_m,
                min_cluster_points,
            )
        )
    previous, current = frame_clusters
    previous_centers = np.asarray(
        [0.5 * (cluster.min(axis=0) + cluster.max(axis=0)) for cluster in previous],
        dtype=np.float64,
    ).reshape(-1, 3)
    current_centers = np.asarray(
        [0.5 * (cluster.min(axis=0) + cluster.max(axis=0)) for cluster in current],
        dtype=np.float64,
    ).reshape(-1, 3)
    matches: dict[int, int] = {}
    if len(previous_centers) and len(current_centers):
        costs = np.linalg.norm(
            previous_centers[:, None, :] - current_centers[None, :, :],
            axis=-1,
        )
        previous_indices, current_indices = linear_sum_assignment(costs)
        for previous_index, current_index in zip(previous_indices, current_indices):
            if costs[previous_index, current_index] <= 0.12:
                matches[int(current_index)] = int(previous_index)

    obstacles: list[CameraObstacle] = []
    for current_index, cluster in enumerate(current):
        minimum = cluster.min(axis=0)
        maximum = cluster.max(axis=0)
        center = 0.5 * (minimum + maximum)
        size = np.maximum(maximum - minimum + 2.0 * float(inflation_m), 0.045)
        velocity = np.zeros(3, dtype=np.float64)
        if current_index in matches:
            previous_center = previous_centers[matches[current_index]]
            velocity = (center - previous_center) / frame_interval_s
            speed = float(np.linalg.norm(velocity))
            if speed < 0.02:
                velocity[:] = 0.0
            elif speed > 0.30:
                velocity *= 0.30 / speed
        obstacles.append(
            CameraObstacle(
                center_m=center,
                size_m=size,
                velocity_m_s=velocity,
                point_count=len(cluster),
            )
        )
    report = {
        "view_names": [name for name in view_names if name != "wrist"],
        "frame_interval_s": frame_interval_s,
        "frame_cluster_counts": [len(previous), len(current)],
        "obstacle_count": len(obstacles),
        "obstacles": [item.to_report() for item in obstacles],
    }
    return obstacles, report


def camera_reconstruction_metrics(
    reconstructed: list[CameraObstacle],
    ground_truth: list[dict[str, Any]],
    time_s: float,
    match_threshold_m: float = 0.15,
) -> dict[str, Any]:
    """Match reconstructed obstacle centers to simulator truth for evaluation only."""
    truth_centers = []
    truth_velocities = []
    for spec in ground_truth:
        motion = str(spec.get("motion_model", "static"))
        if motion == "static":
            truth_centers.append(np.asarray(spec["position_m"], dtype=np.float64))
            truth_velocities.append(np.zeros(3, dtype=np.float64))
            continue
        center = np.asarray(spec["center_m"], dtype=np.float64)
        axis = np.asarray(spec["axis_xyz"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        if motion == "sinusoidal":
            omega = 2.0 * np.pi * float(spec["frequency_hz"])
            phase = omega * float(time_s) + float(spec["phase_rad"])
            truth_centers.append(center + axis * float(spec["amplitude_m"]) * np.sin(phase))
            truth_velocities.append(axis * float(spec["amplitude_m"]) * omega * np.cos(phase))
        elif motion == "linear_ping_pong":
            half_length = float(spec["corridor_half_length_m"])
            speed = float(spec["speed_m_s"])
            period = 4.0 * half_length / max(speed, 1.0e-12)
            cycle = (float(time_s) / period + float(spec["phase_rad"]) / (2.0 * np.pi)) % 1.0
            if cycle < 0.25:
                displacement, direction = 4.0 * cycle, 1.0
            elif cycle < 0.75:
                displacement, direction = 2.0 - 4.0 * cycle, -1.0
            else:
                displacement, direction = 4.0 * cycle - 4.0, 1.0
            truth_centers.append(center + axis * half_length * displacement)
            truth_velocities.append(axis * speed * direction)
        else:
            raise ValueError(f"unsupported ground-truth motion model: {motion}")
    truth_centers_array = np.asarray(truth_centers, dtype=np.float64).reshape(-1, 3)
    truth_velocities_array = np.asarray(truth_velocities, dtype=np.float64).reshape(-1, 3)
    estimated_centers = np.asarray([item.center_m for item in reconstructed], dtype=np.float64).reshape(-1, 3)
    matched_errors: list[float] = []
    velocity_errors: list[float] = []
    if len(truth_centers_array) and len(estimated_centers):
        costs = np.linalg.norm(truth_centers_array[:, None] - estimated_centers[None, :], axis=-1)
        truth_indices, estimated_indices = linear_sum_assignment(costs)
        for truth_index, estimated_index in zip(truth_indices, estimated_indices):
            if costs[truth_index, estimated_index] <= float(match_threshold_m):
                matched_errors.append(float(costs[truth_index, estimated_index]))
                velocity_errors.append(
                    float(
                        np.linalg.norm(
                            truth_velocities_array[truth_index]
                            - np.asarray(reconstructed[estimated_index].velocity_m_s)
                        )
                    )
                )
    matched = len(matched_errors)
    return {
        "ground_truth_count": len(truth_centers_array),
        "reconstructed_count": len(estimated_centers),
        "matched_count": matched,
        "recall": matched / len(truth_centers_array) if len(truth_centers_array) else 1.0,
        "precision": matched / len(estimated_centers) if len(estimated_centers) else 0.0,
        "mean_center_error_m": float(np.mean(matched_errors)) if matched_errors else None,
        "max_center_error_m": float(np.max(matched_errors)) if matched_errors else None,
        "mean_velocity_error_m_s": float(np.mean(velocity_errors)) if velocity_errors else None,
        "match_threshold_m": float(match_threshold_m),
    }


def camera_obstacle_clearance(
    fk_model: Any,
    trajectory: Any,
    obstacles: list[CameraObstacle],
    link_radius_m: float,
    device: str,
    observation_time_s: float = 0.10,
) -> float:
    """Return minimum link-center clearance against camera-reconstructed AABBs."""
    if not obstacles:
        return float("-inf")
    q = torch.as_tensor(trajectory.q, dtype=torch.float32, device=device)
    with torch.no_grad():
        link_positions, _ = fk_model.link_poses(q)
    time = torch.as_tensor(trajectory.time, dtype=torch.float32, device=device)
    minimum = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    for obstacle in obstacles:
        center = torch.as_tensor(obstacle.center_m, dtype=torch.float32, device=device)
        velocity = torch.as_tensor(obstacle.velocity_m_s, dtype=torch.float32, device=device)
        centers = center[None, :] + (time[:, None] - float(observation_time_s)) * velocity[None, :]
        half_extents = 0.5 * torch.as_tensor(
            obstacle.size_m,
            dtype=torch.float32,
            device=device,
        )
        offset = torch.abs(link_positions - centers[:, None, :]) - half_extents[None, None, :]
        outside = torch.linalg.norm(torch.clamp_min(offset, 0.0), dim=-1)
        inside = torch.clamp(torch.amax(offset, dim=-1), max=0.0)
        minimum = torch.minimum(minimum, (outside + inside).min())
    return float(minimum.item() - float(link_radius_m))
