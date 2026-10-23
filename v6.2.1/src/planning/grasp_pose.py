"""Rigid transforms between the analytic TCP and the gripper grasp center."""

from __future__ import annotations

import numpy as np


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    value = value / norm
    return -value if value[0] < 0.0 else value


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion_wxyz(left)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(right)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
    )


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compose_pose(
    position_a: np.ndarray,
    quaternion_a_wxyz: np.ndarray,
    position_b: np.ndarray,
    quaternion_b_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_a = quaternion_to_matrix_wxyz(quaternion_a_wxyz)
    position = np.asarray(position_a, dtype=np.float64).reshape(3) + rotation_a @ np.asarray(
        position_b, dtype=np.float64
    ).reshape(3)
    quaternion = quaternion_multiply_wxyz(quaternion_a_wxyz, quaternion_b_wxyz)
    return position, quaternion


def invert_pose(
    position: np.ndarray, quaternion_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = normalize_quaternion_wxyz(quaternion_wxyz)
    inverse_quaternion = quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
    inverse_position = -quaternion_to_matrix_wxyz(quaternion).T @ np.asarray(
        position, dtype=np.float64
    ).reshape(3)
    return inverse_position, inverse_quaternion


def tcp_pose_to_grasp_center_pose(
    tcp_position: np.ndarray,
    tcp_quaternion_wxyz: np.ndarray,
    tcp_to_grasp_position: np.ndarray,
    tcp_to_grasp_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return compose_pose(
        tcp_position,
        tcp_quaternion_wxyz,
        tcp_to_grasp_position,
        tcp_to_grasp_quaternion_wxyz,
    )


def grasp_center_pose_to_tcp_pose(
    grasp_position: np.ndarray,
    grasp_quaternion_wxyz: np.ndarray,
    tcp_to_grasp_position: np.ndarray,
    tcp_to_grasp_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grasp_to_tcp_position, grasp_to_tcp_quaternion = invert_pose(
        tcp_to_grasp_position, tcp_to_grasp_quaternion_wxyz
    )
    return compose_pose(
        grasp_position,
        grasp_quaternion_wxyz,
        grasp_to_tcp_position,
        grasp_to_tcp_quaternion,
    )
