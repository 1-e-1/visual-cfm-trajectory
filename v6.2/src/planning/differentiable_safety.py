"""Differentiable conservative robot-obstacle clearance objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _bounded_linear_displacement(
    velocity: torch.Tensor,
    time_s: torch.Tensor,
    max_displacement_m: float,
) -> torch.Tensor:
    raw = velocity[:, None, :, :] * time_s[:, :, None, None]
    norm = torch.linalg.norm(raw, dim=-1, keepdim=True)
    bounded_norm = max_displacement_m * torch.tanh(norm / max_displacement_m)
    return raw * bounded_norm / norm.clamp_min(1.0e-9)


def obstacle_clearance_per_timestep(
    fk_model: torch.nn.Module,
    q_path: torch.Tensor,
    duration: torch.Tensor,
    obstacle_tokens: torch.Tensor,
    obstacle_mask: torch.Tensor,
    link_radius_m: float = 0.055,
    max_dynamic_displacement_m: float = 0.30,
) -> torch.Tensor:
    """Return conservative minimum clearance with shape ``[B, T]``."""
    batch, points, dof = q_path.shape
    if dof != 6:
        raise ValueError("q_path must have shape [B, T, 6]")
    if obstacle_tokens.shape[:2] != obstacle_mask.shape:
        raise ValueError("obstacle token and mask batch dimensions must match")
    link_positions, _ = fk_model.link_poses(q_path.reshape(batch * points, dof))
    link_positions = link_positions.reshape(batch, points, -1, 3)

    fractions = torch.linspace(0.0, 1.0, points, device=q_path.device, dtype=q_path.dtype)
    time_s = duration[:, None] * fractions[None, :]
    size = obstacle_tokens[..., 1:4].to(q_path.dtype)
    initial_position = obstacle_tokens[..., 4:7].to(q_path.dtype)
    velocity = obstacle_tokens[..., 11:14].to(q_path.dtype)
    displacement = _bounded_linear_displacement(
        velocity,
        time_s,
        max_displacement_m=max_dynamic_displacement_m,
    )
    centers = initial_position[:, None, :, :] + displacement
    half_extents = 0.5 * size

    delta = (
        torch.abs(link_positions[:, :, :, None, :] - centers[:, :, None, :, :])
        - half_extents[:, None, None, :, :]
    )
    outside = torch.linalg.norm(torch.clamp_min(delta, 0.0), dim=-1)
    inside = torch.clamp_max(torch.amax(delta, dim=-1), 0.0)
    signed_distance = outside + inside - link_radius_m
    signed_distance = signed_distance.masked_fill(
        ~obstacle_mask[:, None, None, :].bool(),
        torch.finfo(signed_distance.dtype).max,
    )
    return signed_distance.amin(dim=(2, 3))


def differentiable_collision_loss(
    fk_model: torch.nn.Module,
    q_path: torch.Tensor,
    duration: torch.Tensor,
    obstacle_tokens: torch.Tensor,
    obstacle_mask: torch.Tensor,
    safety_margin_m: float = 0.020,
    link_radius_m: float = 0.055,
    temperature_m: float = 0.010,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize clearance below a margin using a smooth one-sided barrier."""
    clearance = obstacle_clearance_per_timestep(
        fk_model,
        q_path,
        duration,
        obstacle_tokens,
        obstacle_mask,
        link_radius_m=link_radius_m,
    )
    violation = F.softplus((safety_margin_m - clearance) / temperature_m) * temperature_m
    per_sample = torch.mean((violation / (safety_margin_m + link_radius_m)) ** 2, dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def self_clearance_per_timestep(
    fk_model: torch.nn.Module,
    q_path: torch.Tensor,
    capsule_radius_m: float = 0.045,
    adjacency: int = 2,
    samples_per_segment: int = 5,
) -> torch.Tensor:
    """Approximate non-adjacent link capsule clearance with shape ``[B, T]``."""
    batch, points, dof = q_path.shape
    if dof != 6:
        raise ValueError("q_path must have shape [B, T, 6]")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    link_positions, _ = fk_model.link_poses(q_path.reshape(batch * points, dof))
    link_positions = link_positions.reshape(batch, points, -1, 3)
    base = torch.zeros(
        batch,
        points,
        1,
        3,
        device=q_path.device,
        dtype=q_path.dtype,
    )
    chain_points = torch.cat((base, link_positions), dim=2)
    starts = chain_points[:, :, :-1]
    ends = chain_points[:, :, 1:]
    segment_count = starts.shape[2]
    pairs = [
        (left, right)
        for left in range(segment_count)
        for right in range(left + adjacency + 1, segment_count)
    ]
    if not pairs:
        return torch.full(
            (batch, points),
            torch.finfo(q_path.dtype).max,
            device=q_path.device,
            dtype=q_path.dtype,
        )
    fractions = torch.linspace(
        0.0,
        1.0,
        samples_per_segment,
        device=q_path.device,
        dtype=q_path.dtype,
    )
    sampled = starts[..., None, :] + fractions[None, None, None, :, None] * (
        ends - starts
    )[..., None, :]
    left_indices = torch.tensor([pair[0] for pair in pairs], device=q_path.device)
    right_indices = torch.tensor([pair[1] for pair in pairs], device=q_path.device)
    left_samples = sampled.index_select(2, left_indices)
    right_samples = sampled.index_select(2, right_indices)
    distances = torch.linalg.norm(
        left_samples[..., :, None, :] - right_samples[..., None, :, :],
        dim=-1,
    ).amin(dim=(-1, -2))
    lengths = torch.linalg.norm(ends - starts, dim=-1)
    valid = (lengths.index_select(2, left_indices) >= 1.0e-8) & (
        lengths.index_select(2, right_indices) >= 1.0e-8
    )
    distances = distances.masked_fill(~valid, torch.finfo(distances.dtype).max)
    return distances.amin(dim=2) - 2.0 * capsule_radius_m


def differentiable_self_collision_loss(
    fk_model: torch.nn.Module,
    q_path: torch.Tensor,
    safety_margin_m: float = 0.010,
    capsule_radius_m: float = 0.045,
    temperature_m: float = 0.005,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize self-clearance below the required capsule margin."""
    clearance = self_clearance_per_timestep(
        fk_model,
        q_path,
        capsule_radius_m=capsule_radius_m,
    )
    violation = F.softplus((safety_margin_m - clearance) / temperature_m) * temperature_m
    per_sample = torch.mean(
        (violation / (safety_margin_m + 2.0 * capsule_radius_m)) ** 2,
        dim=1,
    )
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def safety_guided_refinement(
    fk_model: torch.nn.Module,
    q_paths: torch.Tensor,
    durations: torch.Tensor,
    obstacle_tokens: torch.Tensor,
    obstacle_mask: torch.Tensor,
    q_start: torch.Tensor,
    steps: int = 12,
    learning_rate: float = 0.08,
    safety_margin_m: float = 0.020,
    max_joint_adjustment_rad: float = 0.35,
    self_collision_weight: float = 2.0,
    periodic_joints: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Refine sampled paths with collision, smoothness, and residual objectives."""
    if steps <= 0:
        return q_paths, {"initial_safety_loss": 0.0, "final_safety_loss": 0.0}
    original = q_paths.detach()
    optimized = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([optimized], lr=learning_rate)
    initial_safety = None
    final_safety = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        safety = differentiable_collision_loss(
            fk_model,
            optimized,
            durations,
            obstacle_tokens,
            obstacle_mask,
            safety_margin_m=safety_margin_m,
            reduction="none",
        )
        self_safety = differentiable_self_collision_loss(
            fk_model,
            optimized,
            reduction="none",
        )
        joint_delta = torch.diff(optimized, dim=1)
        residual = optimized - original
        if periodic_joints:
            joint_delta = torch.remainder(joint_delta + torch.pi, 2.0 * torch.pi) - torch.pi
            residual = torch.remainder(residual + torch.pi, 2.0 * torch.pi) - torch.pi
        acceleration_proxy = torch.diff(joint_delta, dim=1)
        loss = (
            safety.mean()
            + self_collision_weight * self_safety.mean()
            + 0.04 * acceleration_proxy.square().mean()
            + 0.12 * residual.square().mean()
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            optimized.clamp_(
                min=original - max_joint_adjustment_rad,
                max=original + max_joint_adjustment_rad,
            )
            if periodic_joints:
                optimized.copy_(
                    torch.remainder(optimized + torch.pi, 2.0 * torch.pi) - torch.pi
                )
            optimized[:, 0] = q_start
        if initial_safety is None:
            initial_safety = float(safety.mean().detach())
        final_safety = float(safety.mean().detach())
    return optimized.detach(), {
        "initial_safety_loss": initial_safety or 0.0,
        "final_safety_loss": final_safety or 0.0,
    }
