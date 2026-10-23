from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from h1_1111_grasp.trajectory_safety import (
    environment_clearance,
    environment_clearance_batch,
    self_clearance,
    self_clearance_batch,
    trajectory_derivatives,
)


def test_capsule_pointcloud_clearance() -> None:
    segments = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    cloud = np.array([[0.5, 0.2, 0.0]])
    assert np.isclose(environment_clearance(segments, cloud, 0.05), 0.15)


def test_constant_trajectory_has_zero_derivatives() -> None:
    q = np.zeros((64, 6))
    velocity, acceleration, jerk = trajectory_derivatives(q, 2.0)
    assert np.allclose(velocity, 0.0)
    assert np.allclose(acceleration, 0.0)
    assert np.allclose(jerk, 0.0)


def test_derivatives_use_shortest_periodic_displacement() -> None:
    q = np.zeros((64, 6))
    q[:32, 0] = np.linspace(3.0, np.pi - 0.001, 32)
    q[32:, 0] = np.linspace(-np.pi + 0.001, -3.0, 32)
    velocity, acceleration, jerk = trajectory_derivatives(q, 2.0)
    assert np.max(np.abs(velocity[:, 0])) < 0.2
    assert np.isfinite(acceleration).all()
    assert np.isfinite(jerk).all()


def test_batch_self_clearance_matches_scalar() -> None:
    rng = np.random.default_rng(7)
    segments = rng.normal(size=(12, 6, 2, 3))
    expected = np.array([self_clearance(frame[None], 0.045) for frame in segments])
    actual = self_clearance_batch(segments, 0.045)
    assert np.allclose(actual, expected, atol=1.0e-10)


def test_batch_environment_clearance_matches_scalar() -> None:
    rng = np.random.default_rng(9)
    segments = rng.normal(size=(5, 4, 6, 2, 3))
    cloud = rng.normal(size=(31, 3))
    expected = np.array([environment_clearance(path, cloud, 0.05) for path in segments])
    actual = environment_clearance_batch(segments, cloud, 0.05, segment_chunk_size=7)
    assert np.allclose(actual, expected, atol=1.0e-10)
