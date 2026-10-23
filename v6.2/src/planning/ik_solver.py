"""Multi-restart position-only IK for the periodic six-DOF auxiliary arm.

Shared by dataset generation (P2 q_ik labels) and IK-first inference.
All joint angles are wrapped to [-pi, pi); FK is the differentiable ZBot chain.
"""
from __future__ import annotations

import numpy as np
import torch

from src.robot.torch_kinematics import DifferentiableZBotFK


def _wrap_angles(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(values, torch.Tensor):
        return torch.remainder(values + np.pi, 2.0 * np.pi) - np.pi
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


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
