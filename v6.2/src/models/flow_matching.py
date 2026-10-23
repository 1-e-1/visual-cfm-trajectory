"""Conditional Flow Matching network for fixed-length joint trajectories."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ConditionalFlowMatcher(nn.Module):
    def __init__(
        self,
        condition_dim: int = 13,
        trajectory_points: int = 64,
        dof: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_embedding_dim: int = 128,
        min_duration: float = 0.8,
        max_duration: float = 5.0,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.trajectory_points = trajectory_points
        self.dof = dof
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.joint_scale = math.pi
        self.input_projection = nn.Linear(dof, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, trajectory_points, hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, dof)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, noisy_trajectory: torch.Tensor, flow_time: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_trajectory.shape[1:] != (self.trajectory_points, self.dof):
            raise ValueError(f"Expected trajectory [B,{self.trajectory_points},{self.dof}]")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"Expected condition dimension {self.condition_dim}")
        normalized_condition = torch.cat((condition[:, :6] / self.joint_scale, condition[:, 6:]), dim=-1)
        condition_embedding = self.condition_encoder(normalized_condition)
        time_embedding = self.time_encoder(flow_time)
        tokens = self.input_projection(noisy_trajectory) + self.position_embedding
        tokens = tokens + condition_embedding[:, None, :] + time_embedding[:, None, :]
        velocity = self.velocity_head(self.output_norm(self.backbone(tokens)))
        duration_unit = torch.sigmoid(self.duration_head(condition_embedding).squeeze(-1))
        duration = self.min_duration + (self.max_duration - self.min_duration) * duration_unit
        return velocity, duration

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        steps: int = 32,
        solver: str = "heun",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        batch = condition.shape[0]
        trajectory = torch.randn(
            batch,
            self.trajectory_points,
            self.dof,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        trajectory[:, 0, :] = condition[:, :6] / self.joint_scale
        dt = 1.0 / steps
        duration = torch.full((batch,), self.min_duration, device=condition.device, dtype=condition.dtype)
        for step in range(steps):
            time = torch.full((batch,), step / steps, device=condition.device, dtype=condition.dtype)
            velocity, duration = self(trajectory, time, condition)
            if solver == "euler":
                trajectory = trajectory + dt * velocity
            else:
                prediction = trajectory + dt * velocity
                next_time = torch.full((batch,), (step + 1) / steps, device=condition.device, dtype=condition.dtype)
                next_velocity, duration = self(prediction, next_time, condition)
                trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
            trajectory[:, 0, :] = condition[:, :6] / self.joint_scale
        trajectory[:, 0, :] = condition[:, :6] / self.joint_scale
        return trajectory * self.joint_scale, duration
