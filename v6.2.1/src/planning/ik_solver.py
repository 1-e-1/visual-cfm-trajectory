"""Multi-restart position-only IK for the periodic six-DOF auxiliary arm.

Shared by dataset generation (P2 q_ik labels) and IK-first inference.
All joint angles are wrapped to [-pi, pi); FK is the differentiable ZBot chain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.robot.torch_kinematics import DifferentiableZBotFK


def _wrap_angles(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(values, torch.Tensor):
        return torch.remainder(values + np.pi, 2.0 * np.pi) - np.pi
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_distance_rad(
    actual_wxyz: np.ndarray | torch.Tensor,
    desired_wxyz: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Shortest rotation angle, invariant to the quaternion sign."""
    if isinstance(actual_wxyz, torch.Tensor):
        actual = actual_wxyz / torch.linalg.norm(
            actual_wxyz, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        desired = torch.as_tensor(
            desired_wxyz, dtype=actual.dtype, device=actual.device
        )
        desired = desired / torch.linalg.norm(
            desired, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        dot = torch.sum(actual * desired, dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)
    actual = np.asarray(actual_wxyz, dtype=np.float64)
    desired = np.asarray(desired_wxyz, dtype=np.float64)
    actual = actual / np.maximum(np.linalg.norm(actual, axis=-1, keepdims=True), 1.0e-12)
    desired = desired / np.maximum(np.linalg.norm(desired, axis=-1, keepdims=True), 1.0e-12)
    dot = np.clip(np.abs(np.sum(actual * desired, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


@dataclass(frozen=True)
class PoseIKCandidate:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    self_clearance_m: float | None
    environment_clearance_m: float | None


def solve_pose_ik_candidates(
    fk_model: DifferentiableZBotFK,
    goal_position: np.ndarray,
    goal_quaternion_wxyz: np.ndarray,
    n_candidates: int = 8,
    restarts: int = 128,
    iterations: int = 300,
    learning_rate: float = 0.05,
    orientation_weight: float = 0.1,
    position_tolerance_m: float = 0.005,
    orientation_tolerance_rad: float = np.deg2rad(5.0),
    self_clearance_fn=None,
    environment_clearance_fn=None,
    clearance_threshold_m: float = 0.01,
    seed_configurations: np.ndarray | None = None,
    seed: int = 0,
    device: str = "cuda:0",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[PoseIKCandidate]]:
    """Find up to eight collision-filtered numerical pose-IK branches.

    The USD chain has not been proven to have a standard spherical wrist, so
    this function does not assume that exactly eight analytic solutions exist.
    Non-converged or unsafe restarts are never returned as valid solutions.
    """
    if n_candidates < 1 or restarts < 1 or iterations < 1:
        raise ValueError("n_candidates, restarts and iterations must be positive")
    goal = np.asarray(goal_position, dtype=np.float64).reshape(3)
    goal_quaternion = np.asarray(goal_quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(goal_quaternion))
    if norm < 1.0e-12:
        raise ValueError("goal_quaternion_wxyz must be non-zero")
    goal_quaternion /= norm

    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, size=(restarts, 6)).astype(np.float32)
    if seed_configurations is not None:
        seeds = np.asarray(seed_configurations, dtype=np.float32).reshape(-1, 6)
        count = min(len(seeds), restarts)
        initial[:count] = _wrap_angles(seeds[:count])
    q_free = torch.tensor(initial, dtype=torch.float32, device=device, requires_grad=True)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    goal_q_t = torch.as_tensor(goal_quaternion, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([q_free], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        position, quaternion = fk_model(q_free)
        position_loss = torch.sum((position - goal_t) ** 2, dim=-1)
        quaternion_dot = torch.sum(quaternion * goal_q_t, dim=-1).abs().clamp(max=1.0)
        orientation_loss = 1.0 - quaternion_dot.square()
        (position_loss + orientation_weight * orientation_loss).mean().backward()
        optimizer.step()
        with torch.no_grad():
            q_free.copy_(torch.remainder(q_free + np.pi, 2.0 * np.pi) - np.pi)

    q_tensor = _wrap_angles(q_free.detach())
    with torch.no_grad():
        positions, quaternions = fk_model(q_tensor)
        position_errors = torch.linalg.norm(positions - goal_t, dim=-1)
        orientation_errors = quaternion_distance_rad(quaternions, goal_q_t)
    q_values = q_tensor.cpu().numpy()
    position_values = position_errors.cpu().numpy()
    orientation_values = orientation_errors.cpu().numpy()

    diagnostics: list[PoseIKCandidate] = []
    for index, q_row in enumerate(q_values):
        if position_values[index] > position_tolerance_m:
            continue
        if orientation_values[index] > orientation_tolerance_rad:
            continue
        self_clearance = (
            float(self_clearance_fn(q_row[None]))
            if self_clearance_fn is not None
            else None
        )
        environment_clearance = (
            float(environment_clearance_fn(q_row[None]))
            if environment_clearance_fn is not None
            else None
        )
        if self_clearance is not None and self_clearance < clearance_threshold_m:
            continue
        if environment_clearance is not None and environment_clearance < clearance_threshold_m:
            continue
        candidate = PoseIKCandidate(
            q=np.asarray(_wrap_angles(q_row), dtype=np.float32),
            position_error_m=float(position_values[index]),
            orientation_error_rad=float(orientation_values[index]),
            self_clearance_m=self_clearance,
            environment_clearance_m=environment_clearance,
        )
        if any(
            np.linalg.norm(_wrap_angles(candidate.q - other.q)) <= 0.2
            for other in diagnostics
        ):
            continue
        diagnostics.append(candidate)

    diagnostics.sort(
        key=lambda item: (
            -(item.self_clearance_m if item.self_clearance_m is not None else 0.0),
            -(item.environment_clearance_m if item.environment_clearance_m is not None else 0.0),
            item.position_error_m + 0.1 * item.orientation_error_rad,
        )
    )
    diagnostics = diagnostics[:n_candidates]
    solutions = (
        np.stack([item.q for item in diagnostics]).astype(np.float32)
        if diagnostics
        else np.empty((0, 6), dtype=np.float32)
    )
    if return_diagnostics:
        return solutions, diagnostics
    return solutions


def solve_ik_candidates(
    fk_model: DifferentiableZBotFK,
    goal_position: np.ndarray,
    n_candidates: int = 4,
    restarts: int = 8,
    iterations: int = 100,
    learning_rate: float = 0.08,
    position_tolerance_m: float = 0.005,
    self_clearance_fn=None,
    clearance_threshold_m: float = 0.01,
    seed: int = 0,
    device: str = "cuda:0",
) -> np.ndarray:
    """Return up to ``n_candidates`` wrapped joint configurations whose FK
    reaches ``goal_position`` within tolerance, deduplicated and sorted by
    optional self-clearance (descending).  Falls back to the lowest-error
    restarts when fewer solutions converge."""
    goal = np.asarray(goal_position, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(seed)
    q_free = torch.tensor(
        rng.uniform(-np.pi, np.pi, size=(restarts, 6)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([q_free], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad()
        position, _ = fk_model(q_free)
        loss = torch.mean(torch.sum((position - goal_t) ** 2, dim=-1))
        loss.backward()
        optimizer.step()
    q = _wrap_angles(q_free.detach())
    with torch.no_grad():
        position, _ = fk_model(q)
        error = torch.linalg.norm(position - goal_t, dim=-1).cpu().numpy()
    q_np = q.cpu().numpy()

    converged = [q_np[i] for i in range(restarts) if error[i] <= position_tolerance_m]
    if self_clearance_fn is not None and len(converged):
        clearances = np.asarray(
            [float(self_clearance_fn(q_row[None])) for q_row in converged]
        )
        converged = [q for q, c in zip(converged, clearances) if c >= clearance_threshold_m]

    # deduplicate: keep configurations separated by more than 0.2 rad
    unique: list[np.ndarray] = []
    for q_row in converged:
        if all(np.linalg.norm(_wrap_angles(q_row) - _wrap_angles(u)) > 0.2 for u in unique):
            unique.append(q_row)
        if len(unique) >= n_candidates:
            break
    if len(unique) < n_candidates:
        fallback = sorted(range(restarts), key=lambda i: error[i])[:n_candidates - len(unique)]
        for i in fallback:
            if not any(np.allclose(_wrap_angles(q_np[i]), _wrap_angles(u), atol=0.05) for u in unique):
                unique.append(q_np[i])
            if len(unique) >= n_candidates:
                break
    return np.stack([_wrap_angles(q) for q in unique[:n_candidates]]).astype(np.float32)
