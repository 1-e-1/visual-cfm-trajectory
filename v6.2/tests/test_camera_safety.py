import numpy as np

from src.planning.camera_safety import (
    CameraObstacle,
    camera_reconstruction_metrics,
    look_at_rotation,
    remove_robot_points,
)


def test_reconstruction_metrics_match_static_and_moving_obstacles() -> None:
    reconstructed = [
        CameraObstacle(
            center_m=np.asarray([0.01, 0.0, 0.5]),
            size_m=np.asarray([0.1, 0.1, 0.1]),
            velocity_m_s=np.zeros(3),
            point_count=20,
        ),
        CameraObstacle(
            center_m=np.asarray([0.3, 0.0, 0.4]),
            size_m=np.asarray([0.05, 0.05, 0.05]),
            velocity_m_s=np.asarray([0.1, 0.0, 0.0]),
            point_count=12,
        ),
    ]
    truth = [
        {"motion_model": "static", "position_m": [0.0, 0.0, 0.5]},
        {
            "motion_model": "linear_ping_pong",
            "center_m": [0.3, 0.0, 0.4],
            "axis_xyz": [1.0, 0.0, 0.0],
            "corridor_half_length_m": 0.1,
            "speed_m_s": 0.1,
            "phase_rad": 0.0,
        },
    ]
    metrics = camera_reconstruction_metrics(reconstructed, truth, time_s=0.0)
    assert metrics["matched_count"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_look_at_rotation_points_optical_z_at_target() -> None:
    position = np.asarray([0.85, 0.0, 0.48])
    target = np.asarray([0.0, 0.0, 0.34])
    rotation = look_at_rotation(position, target)
    expected = target - position
    expected /= np.linalg.norm(expected)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7)
    assert np.allclose(rotation[:, 2], expected, atol=1.0e-7)


def test_remove_robot_points_keeps_distant_obstacle() -> None:
    links = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.3],
        ]
    )
    points = np.asarray(
        [
            [0.01, 0.0, 0.15],
            [0.08, 0.0, 0.28],
            [0.30, 0.20, 0.30],
        ]
    )
    filtered = remove_robot_points(points, links, radius_m=0.065)
    assert filtered.shape == (1, 3)
    assert np.allclose(filtered[0], points[-1])
