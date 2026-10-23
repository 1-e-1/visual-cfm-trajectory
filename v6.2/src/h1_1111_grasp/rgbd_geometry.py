"""Multi-camera RGB-D fusion in an auxiliary-arm base frame."""

from __future__ import annotations

import numpy as np


def point_cloud_component_tokens(
    points: np.ndarray,
    max_components: int = 5,
    voxel_size_m: float = 0.05,
    maximum_extent_m: float = 0.45,
    return_point_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Approximate separated scene geometry with deterministic local AABBs.

    With ``return_point_indices=True`` a third output lists the finite-cloud row
    indices belonging to each component (used to derive target-component labels).
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tokens = np.zeros((max_components, 14), dtype=np.float32)
    mask = np.zeros(max_components, dtype=np.bool_)
    finite_mask = np.isfinite(cloud).all(axis=1)
    finite = cloud[finite_mask]
    if len(finite) == 0:
        return (tokens, mask, []) if return_point_indices else (tokens, mask)
    finite_rows = np.flatnonzero(finite_mask)

    voxel_indices = np.floor(finite / voxel_size_m).astype(np.int64)
    occupied, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    lookup = {tuple(index): row for row, index in enumerate(occupied)}
    unseen = set(range(len(occupied)))
    components: list[np.ndarray] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        voxel_rows = [seed]
        while stack:
            current = stack.pop()
            origin = occupied[current]
            for offset in offsets:
                neighbor = lookup.get(tuple(origin + offset))
                if neighbor is not None and neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    voxel_rows.append(neighbor)
        voxel_rows_array = np.asarray(voxel_rows, dtype=np.int64)
        point_rows = np.flatnonzero(np.isin(inverse, voxel_rows_array))
        if len(point_rows) >= 3:
            components.append(finite_rows[point_rows])

    while len(components) < max_components:
        splittable = [
            (float(np.ptp(finite[component], axis=0).max()), index)
            for index, component in enumerate(components)
            if len(component) >= 8
        ]
        if not splittable or max(splittable)[0] <= maximum_extent_m:
            break
        _, index = max(splittable)
        component = components.pop(index)
        component_points = finite[component]
        axis = int(np.argmax(np.ptp(component_points, axis=0)))
        order = np.argsort(component_points[:, axis], kind="stable")
        midpoint = len(order) // 2
        components.extend((component[order[:midpoint]], component[order[midpoint:]]))

    components.sort(key=len, reverse=True)
    component_rows: list[np.ndarray] = []
    for index, component in enumerate(components[:max_components]):
        component_points = finite[component]
        lower = np.quantile(component_points, 0.02, axis=0)
        upper = np.quantile(component_points, 0.98, axis=0)
        tokens[index, 0] = 0.0
        tokens[index, 1:4] = np.maximum(upper - lower, voxel_size_m)
        tokens[index, 4:7] = 0.5 * (upper + lower)
        tokens[index, 7] = 1.0
        mask[index] = True
        if return_point_indices:
            component_rows.append(component)
    if return_point_indices:
        return tokens, mask, component_rows
    return tokens, mask


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def depth_to_points_ros(
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    stride: int = 2,
    depth_range_m: tuple[float, float] = (0.03, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject image-plane depth into ROS optical coordinates (x right, y down, z forward)."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if stride < 1:
        raise ValueError("stride must be positive")
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= depth_range_m[0]) & (z <= depth_range_m[1])
    x = (columns - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack((x, y, z), axis=-1)
    return points[valid], valid


def transform_points(
    points: np.ndarray,
    source_position_w: np.ndarray,
    source_quaternion_wxyz: np.ndarray,
    target_position_w: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Transform source-frame points into a target frame from two world poses."""
    source_rotation = quaternion_wxyz_to_matrix(source_quaternion_wxyz)
    target_rotation = quaternion_wxyz_to_matrix(target_quaternion_wxyz)
    world = np.asarray(points, dtype=np.float64) @ source_rotation.T
    world += np.asarray(source_position_w, dtype=np.float64).reshape(1, 3)
    return (world - np.asarray(target_position_w, dtype=np.float64).reshape(1, 3)) @ target_rotation


def voxel_downsample(points: np.ndarray, voxel_size_m: float = 0.01) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) == 0:
        return cloud
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    keys = np.floor(cloud / float(voxel_size_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.zeros(len(sums), dtype=np.int64)
    np.add.at(sums, inverse, cloud)
    np.add.at(counts, inverse, 1)
    return sums / counts[:, None]


def fuse_depth_views(
    depths_m: np.ndarray,
    intrinsic_matrices: np.ndarray,
    camera_positions_w: np.ndarray,
    camera_quaternions_wxyz_ros: np.ndarray,
    arm_base_position_w: np.ndarray,
    arm_base_quaternion_wxyz: np.ndarray,
    stride: int = 2,
    voxel_size_m: float = 0.01,
    depth_range_m: tuple[float, float] = (0.03, 3.0),
) -> np.ndarray:
    depths = np.asarray(depths_m)
    if depths.ndim != 3:
        raise ValueError("depths_m must have shape [V, H, W]")
    view_count = depths.shape[0]
    if np.asarray(intrinsic_matrices).shape != (view_count, 3, 3):
        raise ValueError("one 3x3 intrinsic matrix is required per view")
    clouds = []
    for index in range(view_count):
        points, _ = depth_to_points_ros(
            depths[index], intrinsic_matrices[index], stride=stride, depth_range_m=depth_range_m
        )
        clouds.append(
            transform_points(
                points,
                camera_positions_w[index],
                camera_quaternions_wxyz_ros[index],
                arm_base_position_w,
                arm_base_quaternion_wxyz,
            )
        )
    return voxel_downsample(np.concatenate(clouds, axis=0), voxel_size_m=voxel_size_m)


def remove_current_arm_points(
    cloud_points: np.ndarray,
    link_segments: np.ndarray,
    removal_radius_m: float = 0.07,
) -> np.ndarray:
    """Remove only the controlled arm; H1 and the other arm remain as obstacles."""
    cloud = np.asarray(cloud_points, dtype=np.float64).reshape(-1, 3)
    minimum = np.full(len(cloud), np.inf, dtype=np.float64)
    for start, end in np.asarray(link_segments, dtype=np.float64).reshape(-1, 2, 3):
        direction = end - start
        denominator = float(direction @ direction)
        if denominator < 1.0e-12:
            distance = np.linalg.norm(cloud - start, axis=1)
        else:
            fraction = np.clip(((cloud - start) @ direction) / denominator, 0.0, 1.0)
            closest = start + fraction[:, None] * direction
            distance = np.linalg.norm(cloud - closest, axis=1)
        minimum = np.minimum(minimum, distance)
    return cloud[minimum > removal_radius_m]
