"""Obstacle-conditioned Flow Matching with privileged and wrist-vision encoders."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.flow_matching import SinusoidalTimeEmbedding


class ObstacleTokenEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 14,
        hidden_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        classifier: bool = False,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1) if classifier else None

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        encoded = self.transformer(self.input(tokens), src_key_padding_mask=~mask.bool())
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.output(pooled)
        if self.classifier is not None:
            logits = self.classifier(encoded).squeeze(-1)
            return {"pooled": pooled, "logits": logits, "encoded": encoded}
        return pooled


class ResidualVisionBlock(nn.Module):
    """Residual convolution block used by the high-resolution visual encoder."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(16, output_channels // 4), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, output_channels // 4), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(image) + self.skip(image))


class WristVisionEncoder(nn.Module):
    """Small-input RGB-D-semantic temporal encoder with channel-wise view fusion."""

    def __init__(
        self,
        output_dim: int = 128,
        temporal_frames: int = 2,
        camera_views: int = 1,
        backbone: str = "resnet18_small_input",
        condition_cond_dim: int = 0,
        target_type_dim: int = 0,
    ) -> None:
        super().__init__()
        if camera_views < 1:
            raise ValueError("camera_views must be positive")
        self.temporal_frames = temporal_frames
        self.camera_views = camera_views
        self.backbone_name = backbone
        self.condition_cond_dim = condition_cond_dim
        self.target_type_dim = target_type_dim
        channels = temporal_frames * camera_views * 5
        if backbone in {"resnet18_small_input", "compact"}:
            self.backbone = nn.Sequential(
                nn.Conv2d(channels, 32, 5, stride=2, padding=2),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
                nn.Conv2d(128, 192, 3, stride=2, padding=1),
                nn.GroupNorm(12, 192),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            final_channels = 192
        elif backbone in {"residual_cnn_large", "large"}:
            self.backbone = nn.Sequential(
                nn.Conv2d(channels, 64, 5, stride=2, padding=2, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                ResidualVisionBlock(64, 64),
                ResidualVisionBlock(64, 128, stride=2),
                ResidualVisionBlock(128, 128),
                ResidualVisionBlock(128, 256, stride=2),
                ResidualVisionBlock(256, 256),
                ResidualVisionBlock(256, 384, stride=2),
                ResidualVisionBlock(384, 384),
                ResidualVisionBlock(384, 512, stride=2),
                nn.AdaptiveAvgPool2d(1),
            )
            final_channels = 512
        else:
            raise ValueError(f"unsupported vision backbone: {backbone}")
        self.output = nn.Sequential(
            nn.Flatten(), nn.Linear(final_channels, output_dim), nn.LayerNorm(output_dim)
        )
        self.film = None
        self.target_center_head = None
        self.target_size_head = None
        self.target_type_head = None
        if condition_cond_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(condition_cond_dim, final_channels * 2), nn.SiLU()
            )
        if target_type_dim > 0:
            self.target_center_head = nn.Sequential(
                nn.Linear(final_channels, 64), nn.SiLU(), nn.Linear(64, 3)
            )
            self.target_size_head = nn.Sequential(
                nn.Linear(final_channels, 64), nn.SiLU(), nn.Linear(64, 3)
            )
            self.target_type_head = nn.Sequential(
                nn.Linear(final_channels, 64), nn.SiLU(), nn.Linear(64, target_type_dim)
            )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        semantic: torch.Tensor,
        condition_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if rgb.ndim == 5:
            rgb = rgb.unsqueeze(1)
            depth = depth.unsqueeze(1)
            semantic = semantic.unsqueeze(1)
        if rgb.ndim != 6:
            raise ValueError("vision tensors must have shape [B,T,H,W,C] or [B,V,T,H,W,C]")
        batch, views, temporal, height, width = rgb.shape[:5]
        if views != self.camera_views:
            raise ValueError(f"expected {self.camera_views} camera views, received {views}")
        if temporal != self.temporal_frames:
            raise ValueError(f"expected {self.temporal_frames} temporal frames, received {temporal}")
        rgb_normalized = rgb.float() / 127.5 - 1.0
        depth_normalized = torch.clamp(depth.float(), 0.0, 2.0) / 2.0
        semantic_normalized = torch.clamp(semantic.float(), 0.0, 8.0) / 8.0
        frames = torch.cat((rgb_normalized, depth_normalized, semantic_normalized), dim=-1)
        channels = frames.shape[-1]
        image = frames.permute(0, 2, 1, 5, 3, 4).reshape(
            batch,
            temporal * views * channels,
            height,
            width,
        )
        features = self.backbone(image)
        if self.film is not None:
            if condition_pos is None:
                raise ValueError("condition_pos is required when condition_cond_dim > 0")
            gamma, beta = self.film(condition_pos.float()).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            features = features * (1.0 + gamma) + beta
        latent = self.output(features)
        if self.target_center_head is None:
            return latent
        features_flat = features.flatten(1)
        aux = {
            "target_center": self.target_center_head(features_flat),
            "target_size": self.target_size_head(features_flat),
            "target_type_logits": self.target_type_head(features_flat),
        }
        return latent, aux


class ObstacleConditionalFlowMatcher(nn.Module):
    def __init__(
        self,
        condition_dim: int = 9,
        trajectory_points: int = 64,
        dof: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_embedding_dim: int = 128,
        obstacle_hidden_dim: int = 128,
        obstacle_layers: int = 3,
        obstacle_heads: int = 4,
        vision_temporal_frames: int = 2,
        vision_camera_views: int = 1,
        vision_backbone: str = "resnet18_small_input",
        periodic_joints: bool = False,
        goal_condition_dof: int = 0,
        min_duration: float = 0.8,
        max_duration: float = 5.0,
        vision_condition_dim: int = 0,
        use_target_heads: bool = False,
        target_type_dim: int = 5,
        fuse_point_cloud: bool = False,
        point_cloud_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.vision_condition_dim = vision_condition_dim
        self.fuse_point_cloud = fuse_point_cloud
        self.point_cloud_classifier = point_cloud_classifier
        self.condition_dim = condition_dim
        self.trajectory_points = trajectory_points
        self.dof = dof
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.joint_scale = math.pi
        self.periodic_joints = periodic_joints
        self.goal_condition_dof = goal_condition_dof
        self.obstacle_encoder = ObstacleTokenEncoder(
            hidden_dim=obstacle_hidden_dim,
            layers=obstacle_layers,
            heads=obstacle_heads,
            classifier=point_cloud_classifier,
        )
        self.vision_encoder = WristVisionEncoder(
            output_dim=obstacle_hidden_dim,
            temporal_frames=vision_temporal_frames,
            camera_views=vision_camera_views,
            backbone=vision_backbone,
            condition_cond_dim=vision_condition_dim,
            target_type_dim=target_type_dim if use_target_heads else 0,
        )
        self.input_projection = nn.Linear(dof, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, trajectory_points, hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if periodic_joints:
            encoded_condition_dim = 2 * dof + 2 * goal_condition_dof + (condition_dim - dof - goal_condition_dof)
        elif goal_condition_dof > 0:
            encoded_condition_dim = condition_dim
        else:
            encoded_condition_dim = condition_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(encoded_condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obstacle_projection = nn.Linear(obstacle_hidden_dim, hidden_dim)
        self.combined_projection = None
        if fuse_point_cloud:
            self.combined_projection = nn.Linear(3 * obstacle_hidden_dim, hidden_dim)
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_obstacles(
        self,
        obstacle_tokens: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if vision is not None:
            return self.vision_encoder(*vision)
        if obstacle_tokens is None or obstacle_mask is None:
            raise ValueError("Provide privileged obstacle tokens/mask or wrist vision")
        return self.obstacle_encoder(obstacle_tokens, obstacle_mask)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obstacle_latent = self.encode_obstacles(obstacle_tokens, obstacle_mask, vision)
        return self.forward_with_obstacle_latent(
            noisy_trajectory, flow_time, condition, obstacle_latent
        )

    def forward_with_obstacle_latent(
        self,
        noisy_trajectory: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        obstacle_latent: torch.Tensor,
        already_projected: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the velocity field from a precomputed static obstacle representation."""
        if self.periodic_joints:
            start_angles = condition[:, : self.dof]
            if self.goal_condition_dof > 0:
                goal_angles = condition[:, self.dof : self.dof + self.goal_condition_dof]
                rest = condition[:, self.dof + self.goal_condition_dof :]
                normalized_condition = torch.cat(
                    (
                        torch.sin(start_angles),
                        torch.cos(start_angles),
                        torch.sin(goal_angles),
                        torch.cos(goal_angles),
                        rest,
                    ),
                    dim=-1,
                )
            else:
                normalized_condition = torch.cat(
                    (torch.sin(start_angles), torch.cos(start_angles), condition[:, self.dof :]),
                    dim=-1,
                )
        elif self.goal_condition_dof > 0:
            normalized_condition = torch.cat(
                (
                    condition[:, : self.dof + self.goal_condition_dof] / self.joint_scale,
                    condition[:, self.dof + self.goal_condition_dof :],
                ),
                dim=-1,
            )
        else:
            normalized_condition = torch.cat(
                (condition[:, : self.dof] / self.joint_scale, condition[:, self.dof :]), dim=-1
            )
        condition_embedding = self.condition_encoder(normalized_condition)
        obstacle_embedding = (
            obstacle_latent
            if already_projected
            else self.obstacle_projection(obstacle_latent)
        )
        time_embedding = self.time_encoder(flow_time)
        tokens = self.input_projection(noisy_trajectory) + self.position_embedding
        tokens = tokens + (condition_embedding + obstacle_embedding + time_embedding)[:, None, :]
        velocity = self.velocity_head(self.output_norm(self.backbone(tokens)))
        duration_unit = torch.sigmoid(
            self.duration_head(torch.cat((condition_embedding, obstacle_embedding), dim=-1)).squeeze(-1)
        )
        duration = self.min_duration + (self.max_duration - self.min_duration) * duration_unit
        return velocity, duration

    def encode_v621(
        self,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        p_goal: torch.Tensor,
        cloud_tokens: torch.Tensor,
        cloud_mask: torch.Tensor,
        target_component_score: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """v6.2.1: fuse FiLM-conditioned wrist vision with the point-cloud latent.

        Returns a combined latent for the CFM backbone plus auxiliary
        predictions (target object center/size/type and per-component logits).
        When ``target_component_score`` is provided the target token is selected
        with the ground-truth component scores (privileged training phase).
        """
        vision_latent, aux = self.vision_encoder(*vision, condition_pos=p_goal)
        if self.fuse_point_cloud:
            cloud = self.obstacle_encoder(cloud_tokens, cloud_mask)
            pooled = cloud["pooled"]
            logits = cloud["logits"]
            encoded = cloud["encoded"]
            if target_component_score is not None:
                scores = target_component_score.float().masked_fill(~cloud_mask.bool(), 0.0)
                weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            else:
                safe_logits = logits.masked_fill(~cloud_mask.bool(), -1.0e9)
                weights = torch.softmax(safe_logits, dim=-1)
            target_token = (encoded * weights[..., None]).sum(dim=1)
            aux["component_logits"] = logits
        else:
            pooled = self.obstacle_encoder(cloud_tokens, cloud_mask)
            target_token = torch.zeros_like(pooled)
        combined = torch.cat((vision_latent, pooled, target_token), dim=-1)
        return self.combined_projection(combined), aux

    def forward_v621(
        self,
        noisy_trajectory: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        p_goal: torch.Tensor,
        cloud_tokens: torch.Tensor,
        cloud_mask: torch.Tensor,
        target_component_score: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        latent, aux = self.encode_v621(
            vision,
            p_goal,
            cloud_tokens,
            cloud_mask,
            target_component_score=target_component_score,
        )
        velocity, duration = self.forward_with_obstacle_latent(
            noisy_trajectory, flow_time, condition, latent, already_projected=True
        )
        return velocity, duration, aux

    @torch.no_grad()
    def sample_v621(
        self,
        condition: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        semantic: torch.Tensor,
        p_goal: torch.Tensor,
        cloud_tokens: torch.Tensor,
        cloud_mask: torch.Tensor,
        steps: int = 32,
        generator: torch.Generator | None = None,
        target_component_score: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = condition.shape[0]
        if self.periodic_joints:
            trajectory = 2.0 * torch.rand(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            ) - 1.0
        else:
            trajectory = torch.randn(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        start = condition[:, : self.dof] / self.joint_scale
        if self.periodic_joints:
            start = torch.remainder(start + 1.0, 2.0) - 1.0
        trajectory[:, 0] = start
        dt = 1.0 / steps
        duration = torch.full((batch,), self.min_duration, device=condition.device)
        latent, aux = self.encode_v621(
            (rgb, depth, semantic),
            p_goal,
            cloud_tokens,
            cloud_mask,
            target_component_score=target_component_score,
        )
        for step in range(steps):
            time = torch.full((batch,), step / steps, device=condition.device)
            velocity, duration = self.forward_with_obstacle_latent(
                trajectory, time, condition, latent, already_projected=True
            )
            prediction = trajectory + dt * velocity
            if self.periodic_joints:
                prediction = torch.remainder(prediction + 1.0, 2.0) - 1.0
            next_time = torch.full((batch,), (step + 1) / steps, device=condition.device)
            next_velocity, duration = self.forward_with_obstacle_latent(
                prediction, next_time, condition, latent, already_projected=True
            )
            trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
            if self.periodic_joints:
                trajectory = torch.remainder(trajectory + 1.0, 2.0) - 1.0
            trajectory[:, 0] = start
        return trajectory * self.joint_scale, duration, aux

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor,
        obstacle_mask: torch.Tensor,
        steps: int = 32,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = condition.shape[0]
        if self.periodic_joints:
            trajectory = 2.0 * torch.rand(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            ) - 1.0
        else:
            trajectory = torch.randn(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        start = condition[:, : self.dof] / self.joint_scale
        if self.periodic_joints:
            start = torch.remainder(start + 1.0, 2.0) - 1.0
        trajectory[:, 0] = start
        dt = 1.0 / steps
        duration = torch.full((batch,), self.min_duration, device=condition.device)
        obstacle_latent = self.obstacle_encoder(obstacle_tokens, obstacle_mask)
        for step in range(steps):
            time = torch.full((batch,), step / steps, device=condition.device)
            velocity, duration = self.forward_with_obstacle_latent(
                trajectory, time, condition, obstacle_latent
            )
            prediction = trajectory + dt * velocity
            if self.periodic_joints:
                prediction = torch.remainder(prediction + 1.0, 2.0) - 1.0
            next_time = torch.full((batch,), (step + 1) / steps, device=condition.device)
            next_velocity, duration = self.forward_with_obstacle_latent(
                prediction, next_time, condition, obstacle_latent
            )
            trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
            if self.periodic_joints:
                trajectory = torch.remainder(trajectory + 1.0, 2.0) - 1.0
            trajectory[:, 0] = start
        return trajectory * self.joint_scale, duration

    @torch.no_grad()
    def sample_visual(
        self,
        condition: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        semantic: torch.Tensor,
        steps: int = 32,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = condition.shape[0]
        if self.periodic_joints:
            trajectory = 2.0 * torch.rand(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            ) - 1.0
        else:
            trajectory = torch.randn(
                batch,
                self.trajectory_points,
                self.dof,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        start = condition[:, : self.dof] / self.joint_scale
        if self.periodic_joints:
            start = torch.remainder(start + 1.0, 2.0) - 1.0
        trajectory[:, 0] = start
        dt = 1.0 / steps
        duration = torch.full((batch,), self.min_duration, device=condition.device)
        obstacle_latent = self.vision_encoder(rgb, depth, semantic)
        for step in range(steps):
            time = torch.full((batch,), step / steps, device=condition.device)
            velocity, duration = self.forward_with_obstacle_latent(
                trajectory, time, condition, obstacle_latent
            )
            prediction = trajectory + dt * velocity
            if self.periodic_joints:
                prediction = torch.remainder(prediction + 1.0, 2.0) - 1.0
            next_time = torch.full((batch,), (step + 1) / steps, device=condition.device)
            next_velocity, duration = self.forward_with_obstacle_latent(
                prediction, next_time, condition, obstacle_latent
            )
            trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
            if self.periodic_joints:
                trajectory = torch.remainder(trajectory + 1.0, 2.0) - 1.0
            trajectory[:, 0] = start
        return trajectory * self.joint_scale, duration
