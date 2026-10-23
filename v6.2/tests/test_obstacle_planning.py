from pathlib import Path

import numpy as np
import torch

from src.planning.obstacle_expert import _numpy_motion_position, generate_smooth_candidates
from src.planning.differentiable_safety import (
    differentiable_collision_loss,
    differentiable_self_collision_loss,
    self_clearance_per_timestep,
    safety_guided_refinement,
)
from src.robot.torch_kinematics import DifferentiableZBotFK
from src.simulation.obstacle_scene import ObstacleAwareKinematicScene


ROOT = Path(__file__).resolve().parents[1]


def test_link_pose_contract_and_gradients():
    model = DifferentiableZBotFK(ROOT / "configs" / "kinematic_chain.json")
    q = torch.zeros(2, 6, requires_grad=True)
    positions, quaternions = model.link_poses(q)
    assert positions.shape == (2, 11, 3)
    assert quaternions.shape == (2, 11, 4)
    assert torch.allclose(torch.linalg.norm(quaternions, dim=-1), torch.ones(2, 11), atol=1.0e-6)
    positions.sum().backward()
    assert q.grad is not None
    assert torch.all(torch.isfinite(q.grad))


def test_obb_sat_detects_overlap_and_separation():
    identity = np.eye(3)
    half = np.asarray([0.1, 0.1, 0.1])
    assert ObstacleAwareKinematicScene._obb_overlap(
        np.zeros(3), identity, half, np.asarray([0.15, 0.0, 0.0]), identity, half
    )
    assert not ObstacleAwareKinematicScene._obb_overlap(
        np.zeros(3), identity, half, np.asarray([0.25, 0.0, 0.0]), identity, half
    )


def test_smooth_candidates_preserve_endpoints_and_limits():
    q_start = np.zeros(6)
    q_goal = np.asarray([0.3, -0.2, 0.1, 0.2, -0.1, 0.25])
    velocity = np.asarray([2.0, 2.0, 2.0, 2.5, 2.5, 3.0])
    acceleration = np.asarray([8.0, 8.0, 8.0, 10.0, 10.0, 12.0])
    jerk = np.asarray([80.0, 80.0, 80.0, 100.0, 100.0, 120.0])
    candidates = generate_smooth_candidates(
        q_start,
        q_goal,
        points=64,
        control_points=10,
        count=8,
        min_duration=0.8,
        max_duration=5.0,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
        seed=42,
    )
    assert candidates
    for trajectory in candidates:
        assert np.array_equal(trajectory.q[0], q_start)
        assert np.array_equal(trajectory.q[-1], q_goal)
        assert np.all(np.max(np.abs(trajectory.dq), axis=0) <= velocity * 1.001)
        assert np.all(np.max(np.abs(trajectory.ddq), axis=0) <= acceleration * 1.001)
        assert np.all(np.max(np.abs(trajectory.dddq), axis=0) <= jerk * 1.001)


def test_space_debris_ping_pong_is_bounded_and_reproducible():
    spec = {
        "motion_model": "linear_ping_pong",
        "center_m": np.asarray([0.1, -0.2, 0.3]),
        "axis_xyz": np.asarray([1.0, 2.0, -1.0]),
        "corridor_half_length_m": 0.18,
        "speed_m_s": 0.24,
        "phase_rad": 1.2,
    }
    times = np.linspace(0.0, 20.0, 1001)
    positions = np.stack([_numpy_motion_position(spec, float(time_s)) for time_s in times])
    center = np.asarray(spec["center_m"])
    distances = np.linalg.norm(positions - center[None, :], axis=1)
    assert float(distances.max()) <= 0.180001
    assert np.array_equal(
        _numpy_motion_position(spec, 0.75),
        _numpy_motion_position(spec, 0.75),
    )


def test_obstacle_shape_pool_respects_training_bounds():
    import yaml

    obstacle_cfg = yaml.safe_load((ROOT / "configs" / "obstacles.yaml").read_text(encoding="utf-8"))
    variants = obstacle_cfg["shape_pool"]["variants"]
    assert len({item["id"] for item in variants}) == len(variants)
    assert {item["role"] for item in variants} == {"static", "dynamic", "debris"}
    assert all(
        max(item["size_m"]) <= obstacle_cfg["task_sampling"]["max_obstacle_diameter_m"]
        for item in variants
    )
    debris_sizes = [max(item["size_m"]) for item in variants if item["role"] == "debris"]
    assert min(debris_sizes) >= obstacle_cfg["space_debris"]["diameter_range_m"][0]
    assert max(debris_sizes) <= obstacle_cfg["space_debris"]["diameter_range_m"][1]


class _LinearPointFK(torch.nn.Module):
    def link_poses(self, q):
        positions = q[:, None, :3]
        quaternions = torch.zeros(q.shape[0], 1, 4, device=q.device, dtype=q.dtype)
        quaternions[..., 0] = 1.0
        return positions, quaternions


def test_collision_loss_and_guidance_reduce_obstacle_risk():
    fk = _LinearPointFK()
    q = torch.zeros(2, 16, 6)
    q[:, :, 0] = torch.linspace(-0.12, 0.12, 16)
    duration = torch.full((2,), 2.0)
    tokens = torch.zeros(2, 5, 14)
    tokens[:, 0, 1:4] = torch.tensor([0.08, 0.08, 0.08])
    tokens[:, 0, 4:7] = torch.tensor([0.02, 0.0, 0.0])
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 0] = True
    initial = differentiable_collision_loss(fk, q, duration, tokens, mask)
    refined, report = safety_guided_refinement(
        fk,
        q,
        duration,
        tokens,
        mask,
        q_start=q[:, 0],
        steps=10,
        learning_rate=0.08,
    )
    final = differentiable_collision_loss(fk, refined, duration, tokens, mask)
    assert float(initial) > 0.0
    assert float(final) < float(initial)
    assert report["final_safety_loss"] < report["initial_safety_loss"]
    assert torch.allclose(refined[:, 0], q[:, 0])


def test_self_collision_loss_has_finite_joint_gradients():
    fk = DifferentiableZBotFK(ROOT / "configs" / "kinematic_chain.json")
    q = torch.zeros(2, 8, 6, requires_grad=True)
    clearance = self_clearance_per_timestep(fk, q)
    loss = differentiable_self_collision_loss(fk, q)
    assert clearance.shape == (2, 8)
    assert torch.all(torch.isfinite(clearance))
    assert torch.isfinite(loss)
    loss.backward()
    assert q.grad is not None
    assert torch.all(torch.isfinite(q.grad))
