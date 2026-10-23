"""Condition-only generative trajectory baselines for paper experiments."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.flow_matching import SinusoidalTimeEmbedding


def _periodic_condition(condition: torch.Tensor, dof: int = 6) -> torch.Tensor:
    start = condition[:, :dof]
    goal = condition[:, dof : 2 * dof]
    rest = condition[:, 2 * dof :]
    return torch.cat(
        (torch.sin(start), torch.cos(start), torch.sin(goal), torch.cos(goal), rest),
        dim=-1,
    )


class ConditionalTrajectoryBackbone(nn.Module):
    """Shared Transformer backbone with no visual or geometric input."""

    def __init__(
        self,
        condition_dim: int = 15,
        trajectory_points: int = 64,
        dof: int = 6,
        hidden_dim: int = 192,
        num_layers: int = 4,
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
        self.joint_scale = math.pi
        self.periodic_joints = True
        self.goal_condition_dof = dof
        self.min_duration = min_duration
        self.max_duration = max_duration
        encoded_condition_dim = 4 * dof + condition_dim - 2 * dof
        self.input_projection = nn.Linear(dof, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, trajectory_points, hidden_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.condition_encoder = nn.Sequential(
            nn.Linear(encoded_condition_dim, hidden_dim),
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
        self.backbone = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, dof)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_embedding = self.condition_encoder(_periodic_condition(condition, self.dof))
        tokens = (
            self.input_projection(noisy_trajectory)
            + self.position_embedding
            + condition_embedding[:, None]
            + self.time_encoder(time)[:, None]
        )
        output = self.output_head(self.output_norm(self.backbone(tokens)))
        duration_unit = torch.sigmoid(
            self.duration_head(condition_embedding).squeeze(-1)
        )
        duration = self.min_duration + (
            self.max_duration - self.min_duration
        ) * duration_unit
        return output, duration


class ConditionOnlyCFM(ConditionalTrajectoryBackbone):
    """Conditional flow matching baseline without environment observations."""

    @torch.no_grad()
    def sample_candidates(
        self,
        condition: torch.Tensor,
        candidates: int,
        steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = condition.repeat(candidates, 1)
        trajectory = 2.0 * torch.rand(
            candidates,
            self.trajectory_points,
            self.dof,
            device=condition.device,
            generator=generator,
        ) - 1.0
        start = condition[:, : self.dof] / self.joint_scale
        trajectory[:, 0] = start
        dt = 1.0 / steps
        duration = torch.full(
            (candidates,), self.min_duration, device=condition.device
        )
        for index in range(steps):
            time = torch.full(
                (candidates,), index / steps, device=condition.device
            )
            velocity, duration = self(trajectory, time, condition)
            prediction = torch.remainder(trajectory + dt * velocity + 1.0, 2.0) - 1.0
            next_time = torch.full(
                (candidates,), (index + 1) / steps, device=condition.device
            )
            next_velocity, duration = self(prediction, next_time, condition)
            trajectory = torch.remainder(
                trajectory + 0.5 * dt * (velocity + next_velocity) + 1.0,
                2.0,
            ) - 1.0
            trajectory[:, 0] = start
        return trajectory * self.joint_scale, duration


class ConditionOnlyDDIM(ConditionalTrajectoryBackbone):
    """Conditional diffusion baseline trained with DDPM and sampled with DDIM."""

    def __init__(self, *args, diffusion_steps: int = 1000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diffusion_steps = diffusion_steps
        betas = torch.linspace(1.0e-4, 0.02, diffusion_steps)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_bar", alphas_bar)

    @torch.no_grad()
    def sample_candidates(
        self,
        condition: torch.Tensor,
        candidates: int,
        steps: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = condition.repeat(candidates, 1)
        sample = torch.randn(
            candidates,
            self.trajectory_points,
            self.dof,
            device=condition.device,
            generator=generator,
        )
        start = condition[:, : self.dof] / self.joint_scale
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            max(2, steps),
            device=condition.device,
        ).round().long().unique_consecutive()
        duration = torch.full(
            (candidates,), self.min_duration, device=condition.device
        )
        for schedule_index, timestep in enumerate(schedule):
            t = torch.full(
                (candidates,),
                float(timestep) / (self.diffusion_steps - 1),
                device=condition.device,
            )
            predicted_noise, duration = self(sample, t, condition)
            alpha = self.alphas_bar[timestep]
            predicted_clean = (
                sample - torch.sqrt(1.0 - alpha) * predicted_noise
            ) / torch.sqrt(alpha)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            if schedule_index == len(schedule) - 1:
                sample = predicted_clean
            else:
                next_timestep = schedule[schedule_index + 1]
                next_alpha = self.alphas_bar[next_timestep]
                sample = (
                    torch.sqrt(next_alpha) * predicted_clean
                    + torch.sqrt(1.0 - next_alpha) * predicted_noise
                )
            sample[:, 0] = start
        return sample * self.joint_scale, duration


def build_condition_only_model(config: dict, model_type: str) -> nn.Module:
    model_cfg = config["model"]
    kwargs = dict(
        condition_dim=int(config["base_condition_dim"]),
        trajectory_points=int(config["trajectory_shape"][0]),
        dof=int(config["trajectory_shape"][1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
    )
    if model_type == "condition_cfm":
        return ConditionOnlyCFM(**kwargs)
    if model_type == "condition_ddim":
        return ConditionOnlyDDIM(
            **kwargs,
            diffusion_steps=int(config.get("diffusion_steps", 1000)),
        )
    raise ValueError(f"unsupported model_type: {model_type}")
