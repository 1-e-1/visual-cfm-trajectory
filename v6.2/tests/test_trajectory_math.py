from __future__ import annotations

import numpy as np
import torch

from src.planning.flow_postprocess import (
    refine_terminal_configurations,
    refine_terminal_positions_batch,
    retime_sampled_path,
    smooth_and_time_parameterize,
)
from src.planning.metrics import trajectory_metrics
from src.planning.expert_optimizer import optimize_expert_candidates
from src.planning.trajectory_math import minimum_jerk_duration, minimum_jerk_trajectory
from src.robot.torch_kinematics import DifferentiableZBotFK


def test_minimum_jerk_endpoints_and_shapes() -> None:
    q_start = np.array([-1.0, 0.2, 0.3, -0.4, 0.5, -0.6])
    q_goal = np.array([1.0, -0.2, -0.3, 0.4, -0.5, 0.6])
    duration, feasible, _ = minimum_jerk_duration(
        q_start,
        q_goal,
        np.full(6, 2.0),
        np.full(6, 8.0),
        np.full(6, 80.0),
        0.8,
        5.0,
    )
    assert feasible
    trajectory = minimum_jerk_trajectory(q_start, q_goal, duration, points=64)
    np.testing.assert_allclose(trajectory.q[0], q_start, atol=1.0e-12)
    np.testing.assert_allclose(trajectory.q[-1], q_goal, atol=1.0e-12)
    np.testing.assert_allclose(trajectory.dq[[0, -1]], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(trajectory.ddq[[0, -1]], 0.0, atol=1.0e-12)
    assert trajectory.q.shape == (64, 6)
    metrics = trajectory_metrics(trajectory, np.ones(6))
    assert metrics["motion_cost_proxy_integral"] > 0.0


def test_expert_optimizer_preserves_exact_endpoints() -> None:
    q_start = np.zeros(6)
    q_goal = np.array([0.8, -0.4, 0.2, 0.5, -0.3, 0.1])
    candidates = optimize_expert_candidates(
        q_start=q_start,
        q_goal=q_goal,
        initial_duration=2.0,
        points=64,
        control_point_count=8,
        min_duration=0.8,
        max_duration=5.0,
        velocity_limits=np.full(6, 2.0),
        acceleration_limits=np.full(6, 8.0),
        jerk_limits=np.full(6, 80.0),
        joint_weights=np.ones(6),
        weights={
            "w_joint_length": 1.0,
            "w_motion_proxy": 0.5,
            "w_velocity": 2.0,
            "w_acceleration": 1.0,
            "w_jerk": 0.2,
            "w_duration": 0.05,
        },
        num_initializations=1,
        max_iterations=3,
        seed=42,
    )
    assert len(candidates) == 1
    np.testing.assert_array_equal(candidates[0].trajectory.q[0], q_start)
    np.testing.assert_array_equal(candidates[0].trajectory.q[-1], q_goal)


def test_terminal_ik_candidates_and_smoothing() -> None:
    fk = DifferentiableZBotFK("configs/kinematic_chain.json")
    q_start = np.zeros(6, dtype=np.float32)
    q_goal = np.array([0.2, -0.3, 0.15, 0.25, -0.1, 0.2], dtype=np.float32)
    with torch.no_grad():
        goal_position, goal_quaternion = fk(torch.from_numpy(q_goal))
    candidates = refine_terminal_configurations(
        fk_model=fk,
        q_initial=q_goal + 0.05,
        goal_position=goal_position.numpy(),
        goal_quaternion_wxyz=goal_quaternion.numpy(),
        restarts=2,
        max_iterations=30,
        position_weight=100.0,
        orientation_weight=20.0,
        regularization_weight=1.0e-5,
        seed=7,
        device="cpu",
    )
    assert len(candidates) == 2
    assert candidates[0].objective <= candidates[1].objective
    assert candidates[0].position_error_m < 1.0e-3

    raw_q = np.linspace(q_start, candidates[0].q, 64)
    trajectory, details = smooth_and_time_parameterize(
        raw_q=raw_q,
        q_start=q_start,
        q_terminal=candidates[0].q,
        predicted_duration=2.0,
        points=64,
        control_point_count=8,
        residual_weight=0.15,
        min_duration=0.8,
        max_duration=5.0,
        velocity_limits=np.full(6, 2.5),
        acceleration_limits=np.full(6, 8.0),
        jerk_limits=np.full(6, 80.0),
    )
    np.testing.assert_allclose(trajectory.q[0], q_start)
    np.testing.assert_allclose(trajectory.q[-1], candidates[0].q)
    assert details["required_duration_s"] <= trajectory.duration


def test_batched_terminal_position_ik_reaches_fk_targets() -> None:
    fk = DifferentiableZBotFK("configs/kinematic_chain.json")
    q_goal = torch.tensor(
        [[0.2, -0.3, 0.4, -0.2, 0.1, 0.3], [0.5, 0.2, -0.4, 0.1, -0.2, 0.6]],
        dtype=torch.float32,
    )
    goal_position, _ = fk(q_goal)
    results = refine_terminal_positions_batch(
        fk_model=fk,
        q_initial=np.zeros((2, 6), dtype=np.float64),
        goal_position=goal_position.detach().numpy(),
        max_iterations=40,
        regularization_weight=1.0e-5,
        device="cpu",
    )
    assert len(results) == 2
    assert max(item.position_error_m for item in results) < 1.0e-4


def test_dense_time_scaling_marks_nominal_excess_without_clipping() -> None:
    u = np.linspace(0.0, 1.0, 64)
    q = np.zeros((64, 6), dtype=np.float64)
    q[:, 0] = np.sin(2.0 * np.pi * u)
    trajectory, details = retime_sampled_path(
        q=q,
        predicted_duration=1.0,
        min_duration=0.8,
        nominal_max_duration=5.0,
        max_output_duration=30.0,
        velocity_limits=np.full(6, 0.5),
        acceleration_limits=np.full(6, 2.0),
        jerk_limits=np.full(6, 20.0),
        timing_points=1024,
    )
    assert trajectory.q.shape == (1024, 6)
    np.testing.assert_allclose(trajectory.q[[0, -1]], q[[0, -1]], atol=1.0e-7)
    assert trajectory.duration > 5.0
    assert details["duration_exceeds_nominal_max"]
    assert details["duration_valid"]
    assert not details["duration_clipped_to_max"]
    assert np.max(np.abs(trajectory.dq), axis=0)[0] <= 0.5 + 1.0e-9
    assert np.max(np.abs(trajectory.ddq), axis=0)[0] <= 2.0 + 1.0e-9
    assert np.max(np.abs(trajectory.dddq), axis=0)[0] <= 20.0 + 1.0e-9


def test_dense_time_scaling_invalidates_duration_above_output_limit() -> None:
    q = np.zeros((64, 6), dtype=np.float64)
    q[:, 0] = np.linspace(0.0, 1.0, 64)
    trajectory, details = retime_sampled_path(
        q=q,
        predicted_duration=1.0,
        min_duration=0.8,
        nominal_max_duration=5.0,
        max_output_duration=30.0,
        velocity_limits=np.full(6, 0.01),
        acceleration_limits=np.full(6, 2.0),
        jerk_limits=np.full(6, 20.0),
        timing_points=1024,
    )
    assert trajectory.duration > 30.0
    assert not details["duration_valid"]
    assert not details["duration_clipped_to_max"]
