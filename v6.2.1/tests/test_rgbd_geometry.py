from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h1_1111_grasp.rgbd_geometry import (
    depth_to_points_ros,
    point_cloud_component_tokens,
    transform_points,
)


def test_depth_center_projects_forward() -> None:
    depth = np.ones((3, 3), dtype=np.float32)
    intrinsic = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    points, _ = depth_to_points_ros(depth, intrinsic, stride=1)
    assert any(np.allclose(point, [0.0, 0.0, 1.0]) for point in points)


def test_identity_rotation_and_translation() -> None:
    points = np.array([[0.0, 0.0, 1.0]])
    transformed = transform_points(
        points,
        source_position_w=np.array([1.0, 0.0, 0.0]),
        source_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_position_w=np.zeros(3),
        target_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(transformed, [[1.0, 0.0, 1.0]])


def test_component_tokens_preserve_separated_geometry() -> None:
    first = np.array([[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.00, 0.01, 0.00]])
    second = first + np.array([1.0, 0.0, 0.0])
    tokens, mask = point_cloud_component_tokens(
        np.concatenate((first, second)), voxel_size_m=0.05
    )
    assert mask.sum() == 2
    centers = np.sort(tokens[mask, 4])
    assert np.allclose(centers, [0.005, 1.005], atol=0.02)
    assert np.all(tokens[mask, 1:4] >= 0.05)
    assert np.all(tokens[mask, 7] == 1.0)
