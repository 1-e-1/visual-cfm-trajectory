"""Flow Matching and differentiable kinematic auxiliary losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def augment_condition_qik(
    condition: torch.Tensor,
    qik_candidates: torch.Tensor | None,
    fk_model: nn.Module,
    goal_condition_dof: int,
    position_tolerance_m: float = 0.005,
    orientation_tolerance_rad: float = 0.0872664626,
    qik_candidate_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly replace the condition's q_ik block (P2 train/eval consistency).

    Chooses uniformly among the q_ik candidates whose FK reaches the condition
    goal position within tolerance; q_ik[0] is always a valid fallback. Returns
    the (possibly cloned) augmented condition.
    """
    if qik_candidates is None or goal_condition_dof <= 0:
        return condition
    if qik_candidates.shape[0] != condition.shape[0] or qik_candidates.ndim != 3:
        return condition
    device = condition.device
    batch = condition.shape[0]
    goal_position = condition[:, -3:]
    with torch.no_grad():
        positions, quaternions = fk_model(qik_candidates.reshape(-1, 6))
        _, target_quaternion = fk_model(condition[:, 6 : 6 + goal_condition_dof])
    positions = positions.reshape(batch, -1, 3)
    quaternions = quaternions.reshape(batch, -1, 4)
    errors = torch.linalg.norm(positions - goal_position[:, None, :], dim=-1)
    quaternion_dot = torch.sum(
        quaternions * target_quaternion[:, None, :], dim=-1
    ).abs().clamp(max=1.0)
    orientation_errors = 2.0 * torch.acos(quaternion_dot)
    valid = (errors <= float(position_tolerance_m)) & (
        orientation_errors <= float(orientation_tolerance_rad)
    )
    if qik_candidate_mask is not None:
        valid &= qik_candidate_mask.to(device=device, dtype=torch.bool)
    valid[:, 0] = True
    picks = torch.zeros(batch, dtype=torch.long, device=device)
    if generator is None or generator.device.type != torch.device(device).type:
        generator = torch.Generator(device=device)
    for index in range(batch):
        options = torch.nonzero(valid[index]).squeeze(-1)
        pick = torch.randint(0, options.numel(), (1,), generator=generator, device=device)
        picks[index] = options[pick]
    augmented = condition.clone()
    augmented[:, 6 : 6 + goal_condition_dof] = qik_candidates[torch.arange(batch, device=device), picks]
    return augmented

from src.planning.differentiable_safety import (
    differentiable_collision_loss,
    differentiable_self_collision_loss,
)


class FlowMatchingLoss(nn.Module):
    def __init__(
        self,
        fk_model: nn.Module,
        weights: dict[str, float],
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
        jerk_limits: torch.Tensor,
        trajectory_points: int = 64,
    ) -> None:
        super().__init__()
        self.fk_model = fk_model
        self.weights = weights
        self.trajectory_points = trajectory_points
        self.register_buffer("velocity_limits", velocity_limits.float())
        self.register_buffer("acceleration_limits", acceleration_limits.float())
        self.register_buffer("jerk_limits", jerk_limits.float())

    def forward(
        self,
        model: nn.Module,
        expert_trajectory: torch.Tensor,
        condition: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = expert_trajectory.shape[0]
        joint_scale = float(getattr(model, "joint_scale", torch.pi))
        normalized_expert = expert_trajectory / joint_scale
        noise = torch.randn_like(normalized_expert)
        flow_time = torch.rand(batch, device=expert_trajectory.device, dtype=expert_trajectory.dtype)
        time_view = flow_time[:, None, None]
        interpolated = (1.0 - time_view) * noise + time_view * normalized_expert
        target_velocity = normalized_expert - noise
        predicted_velocity, predicted_duration = model(interpolated, flow_time, condition)
        reconstructed = interpolated + (1.0 - time_view) * predicted_velocity

        flow_loss = torch.mean((predicted_velocity - target_velocity) ** 2)
        start_loss = torch.mean((reconstructed[:, 0, :] - condition[:, :6] / joint_scale) ** 2)
        projected = torch.cat(
            (condition[:, None, :6], reconstructed[:, 1:, :] * joint_scale),
            dim=1,
        )
        final_position, final_quaternion = self.fk_model(projected[:, -1, :])
        goal_position = condition[:, 6:9]
        goal_quaternion = condition[:, 9:13]
        goal_quaternion = goal_quaternion / torch.linalg.norm(goal_quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
        position_loss = torch.mean((final_position - goal_position) ** 2)
        quaternion_dot = torch.sum(final_quaternion * goal_quaternion, dim=-1).abs().clamp(max=1.0)
        orientation_loss = torch.mean(1.0 - quaternion_dot**2)

        dt = predicted_duration.clamp_min(1.0e-3) / (self.trajectory_points - 1)
        velocity = torch.diff(projected, dim=1) / dt[:, None, None]
        acceleration = torch.diff(velocity, dim=1) / dt[:, None, None]
        jerk = torch.diff(acceleration, dim=1) / dt[:, None, None]
        normalized_velocity = velocity / self.velocity_limits[None, None, :]
        normalized_acceleration = acceleration / self.acceleration_limits[None, None, :]
        normalized_jerk = jerk / self.jerk_limits[None, None, :]
        velocity_loss = F.smooth_l1_loss(normalized_velocity, torch.zeros_like(normalized_velocity))
        acceleration_loss = F.smooth_l1_loss(normalized_acceleration, torch.zeros_like(normalized_acceleration))
        jerk_loss = F.smooth_l1_loss(normalized_jerk, torch.zeros_like(normalized_jerk))
        duration_loss = torch.mean((predicted_duration - target_duration) ** 2)

        losses = {
            "flow": flow_loss,
            "start_consistency": start_loss,
            "final_tcp_position": position_loss,
            "final_tcp_orientation": orientation_loss,
            "velocity_smoothness": velocity_loss,
            "acceleration_smoothness": acceleration_loss,
            "jerk_regularization": jerk_loss,
            "duration": duration_loss,
        }
        total = sum(self.weights[name] * value for name, value in losses.items())
        losses["total"] = total
        return total, losses


class ObstacleFlowMatchingLoss(nn.Module):
    """Position-only CFM objective for obstacle-conditioned trajectories."""

    def __init__(
        self,
        fk_model: nn.Module,
        weights: dict[str, float],
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
        jerk_limits: torch.Tensor,
        trajectory_points: int = 64,
        periodic_joints: bool = False,
    ) -> None:
        super().__init__()
        self.fk_model = fk_model
        self.weights = weights
        self.trajectory_points = trajectory_points
        self.periodic_joints = periodic_joints
        self.register_buffer("velocity_limits", velocity_limits.float())
        self.register_buffer("acceleration_limits", acceleration_limits.float())
        self.register_buffer("jerk_limits", jerk_limits.float())

    def forward(
        self,
        model: nn.Module,
        expert_trajectory: torch.Tensor,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor,
        obstacle_mask: torch.Tensor,
        target_duration: torch.Tensor,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        expert_trajectories: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if expert_trajectories is not None:
            if expert_trajectories.ndim != 4 or expert_trajectories.shape[-2:] != (self.trajectory_points, 6):
                raise ValueError(f"expert_trajectories must have shape [B,N,{self.trajectory_points},6]")
            labels = torch.randint(
                0,
                expert_trajectories.shape[1],
                (expert_trajectories.shape[0],),
                device=expert_trajectories.device,
            )
            expert_trajectory = expert_trajectories[torch.arange(expert_trajectories.shape[0]), labels]
        joint_scale = float(getattr(model, "joint_scale", torch.pi))
        goal_condition_dof = int(getattr(model, "goal_condition_dof", 0))
        dof = int(getattr(model, "dof", 6))
        if goal_condition_dof > 0:
            # v6.2: bend the expert flow target so its terminal equals the condition's q_ik.
            # Keeps the flow objective consistent with the terminal anchor (P2).
            q_ik_norm = torch.remainder(
                condition[:, dof : dof + goal_condition_dof] / joint_scale + 1.0, 2.0
            ) - 1.0
            terminal_delta = torch.remainder(
                q_ik_norm - expert_trajectory[:, -1] + 1.0, 2.0
            ) - 1.0
            ramp_points = 8
            progress = torch.linspace(0.0, 1.0, ramp_points, device=expert_trajectory.device)
            ramp = (progress**2) * (3.0 - 2.0 * progress)  # smoothstep
            expert_trajectory[:, -ramp_points:] = torch.remainder(
                expert_trajectory[:, -ramp_points:]
                + ramp[None, :, None] * terminal_delta[:, None, :]
                + 1.0,
                2.0,
            ) - 1.0
        batch = expert_trajectory.shape[0]
        normalized_expert = expert_trajectory / joint_scale
        if self.periodic_joints:
            normalized_expert = torch.remainder(normalized_expert + 1.0, 2.0) - 1.0
            noise = 2.0 * torch.rand_like(normalized_expert) - 1.0
        else:
            noise = torch.randn_like(normalized_expert)
        flow_time = torch.rand(batch, device=expert_trajectory.device, dtype=expert_trajectory.dtype)
        time_view = flow_time[:, None, None]
        if self.periodic_joints:
            target_velocity = torch.remainder(normalized_expert - noise + 1.0, 2.0) - 1.0
            interpolated = torch.remainder(noise + time_view * target_velocity + 1.0, 2.0) - 1.0
        else:
            interpolated = (1.0 - time_view) * noise + time_view * normalized_expert
            target_velocity = normalized_expert - noise
        predicted_velocity, predicted_duration = self._call_model(
            model,
            interpolated,
            flow_time,
            condition,
            obstacle_tokens,
            obstacle_mask,
            vision,
        )
        reconstructed = interpolated + (1.0 - time_view) * predicted_velocity
        if self.periodic_joints:
            reconstructed = torch.remainder(reconstructed + 1.0, 2.0) - 1.0
        projected = torch.cat((condition[:, None, :6], reconstructed[:, 1:] * joint_scale), dim=1)
        if self.periodic_joints:
            projected = torch.remainder(projected + torch.pi, 2.0 * torch.pi) - torch.pi
        goal_condition_dof = int(getattr(model, "goal_condition_dof", 0))
        if goal_condition_dof > 0:
            # v6.1 terminal anchor (P2): the output trajectory ends exactly at q_ik.
            terminal = torch.remainder(condition[:, 6 : 6 + goal_condition_dof] + torch.pi, 2.0 * torch.pi) - torch.pi
            projected[:, -1] = terminal

        flow_loss = torch.mean((predicted_velocity - target_velocity) ** 2)
        start_error = reconstructed[:, 0] - condition[:, :6] / joint_scale
        if self.periodic_joints:
            start_error = torch.remainder(start_error + 1.0, 2.0) - 1.0
        start_loss = torch.mean(start_error**2)
        final_position, _ = self.fk_model(projected[:, -1])
        position_loss = torch.mean((final_position - condition[:, -3:]) ** 2)
        reconstructed_terminal = reconstructed[:, -1] * joint_scale
        dof = int(getattr(model, "dof", 6))
        if self.periodic_joints:
            reconstructed_terminal = torch.remainder(
                reconstructed_terminal + torch.pi, 2.0 * torch.pi
            ) - torch.pi
        _, predicted_terminal_quaternion = self.fk_model(reconstructed_terminal)
        if goal_condition_dof > 0:
            with torch.no_grad():
                _, goal_terminal_quaternion = self.fk_model(
                    condition[:, dof : dof + goal_condition_dof]
                )
            quaternion_dot = torch.sum(
                predicted_terminal_quaternion * goal_terminal_quaternion, dim=-1
            ).abs().clamp(max=1.0)
            orientation_loss = torch.mean(1.0 - quaternion_dot.square())
        else:
            orientation_loss = torch.zeros(
                (), device=predicted_velocity.device, dtype=predicted_velocity.dtype
            )
        dt = predicted_duration.clamp_min(1.0e-3) / (self.trajectory_points - 1)
        joint_delta = torch.diff(projected, dim=1)
        if self.periodic_joints:
            joint_delta = torch.remainder(joint_delta + torch.pi, 2.0 * torch.pi) - torch.pi
        velocity = joint_delta / dt[:, None, None]
        acceleration = torch.diff(velocity, dim=1) / dt[:, None, None]
        jerk = torch.diff(acceleration, dim=1) / dt[:, None, None]
        velocity_loss = F.smooth_l1_loss(
            velocity / self.velocity_limits[None, None, :],
            torch.zeros_like(velocity),
        )
        acceleration_loss = F.smooth_l1_loss(
            acceleration / self.acceleration_limits[None, None, :],
            torch.zeros_like(acceleration),
        )
        jerk_loss = F.smooth_l1_loss(
            jerk / self.jerk_limits[None, None, :],
            torch.zeros_like(jerk),
        )
        if goal_condition_dof > 0:
            goal_joint_error = reconstructed[:, -1] - condition[:, dof : dof + goal_condition_dof] / joint_scale
            if self.periodic_joints:
                goal_joint_error = torch.remainder(goal_joint_error + 1.0, 2.0) - 1.0
            goal_joint_loss = torch.mean(goal_joint_error**2)
        else:
            goal_joint_loss = torch.zeros((), device=predicted_velocity.device, dtype=predicted_velocity.dtype)
        duration_loss = torch.mean((predicted_duration - target_duration) ** 2)
        collision_loss = differentiable_collision_loss(
            self.fk_model,
            projected,
            predicted_duration,
            obstacle_tokens,
            obstacle_mask,
        )
        self_collision_loss = differentiable_self_collision_loss(
            self.fk_model,
            projected,
        )
        losses = {
            "flow": flow_loss,
            "start_consistency": start_loss,
            "final_tcp_position": position_loss,
            "final_tcp_orientation": orientation_loss,
            "velocity_smoothness": velocity_loss,
            "acceleration_smoothness": acceleration_loss,
            "jerk_regularization": jerk_loss,
            "goal_joint_consistency": goal_joint_loss,
            "duration": duration_loss,
            "collision_clearance": collision_loss,
            "self_collision_clearance": self_collision_loss,
        }
        total = sum(self.weights.get(name, 0.0) * value for name, value in losses.items())
        losses["total"] = total
        return total, losses

    def _call_model(
        self,
        model: nn.Module,
        interpolated: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor,
        obstacle_mask: torch.Tensor,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vision is None:
            return model(
                interpolated,
                flow_time,
                condition,
                obstacle_tokens,
                obstacle_mask,
            )
        return model(
            interpolated,
            flow_time,
            condition,
            vision=vision,
        )


class VisualCFMV621Loss(ObstacleFlowMatchingLoss):
    """v6.2.1 flow objective plus target-object aux losses.

    The base flow/kinematic losses are reused verbatim; the model call is routed
    to ``forward_v621`` (vision + point cloud + p_goal) and the auxiliary
    predictions (target component logits, object center/size/type) are added.
    """

    def __init__(
        self,
        fk_model: nn.Module,
        weights: dict[str, float],
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
        jerk_limits: torch.Tensor,
        trajectory_points: int = 64,
        periodic_joints: bool = False,
        aux_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            fk_model=fk_model,
            weights=weights,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            jerk_limits=jerk_limits,
            trajectory_points=trajectory_points,
            periodic_joints=periodic_joints,
        )
        self.aux_weights = aux_weights or {
            "target_component": 2.0,
            "object_center": 2.0,
            "object_size": 1.0,
            "object_type": 1.0,
        }
        self._last_aux: dict[str, torch.Tensor] | None = None
        self.privileged_target = True
        self._target_score: torch.Tensor | None = None

    def _call_model(
        self,
        model: nn.Module,
        interpolated: torch.Tensor,
        flow_time: torch.Tensor,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor,
        obstacle_mask: torch.Tensor,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vision is None:
            raise ValueError("VisualCFMV621Loss requires vision inputs")
        p_goal = condition[:, -3:]
        score = self._target_score if self.privileged_target else None
        velocity, duration, aux = model.forward_v621(
            interpolated,
            flow_time,
            condition,
            vision,
            p_goal,
            obstacle_tokens,
            obstacle_mask,
            target_component_score=score,
        )
        self._last_aux = aux
        return velocity, duration

    def forward(
        self,
        model: nn.Module,
        expert_trajectory: torch.Tensor,
        condition: torch.Tensor,
        obstacle_tokens: torch.Tensor,
        obstacle_mask: torch.Tensor,
        target_duration: torch.Tensor,
        vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        expert_trajectories: torch.Tensor | None = None,
        target_component_score: torch.Tensor | None = None,
        object_center: torch.Tensor | None = None,
        object_size: torch.Tensor | None = None,
        object_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._target_score = target_component_score
        total, losses = super().forward(
            model,
            expert_trajectory,
            condition,
            obstacle_tokens,
            obstacle_mask,
            target_duration,
            vision=vision,
            expert_trajectories=expert_trajectories,
        )
        self._target_score = None
        aux = self._last_aux
        if aux is None:
            return total, losses
        device = condition.device
        mask = obstacle_mask.bool()
        if target_component_score is not None:
            logits = aux["component_logits"][mask]
            score = target_component_score[mask].to(logits.dtype)
            component_loss = F.binary_cross_entropy_with_logits(logits, score)
            losses["target_component"] = component_loss
            total = total + self.aux_weights["target_component"] * component_loss
        if object_center is not None and "target_center" in aux:
            center_loss = F.mse_loss(aux["target_center"], object_center.float())
            losses["object_center"] = center_loss
            total = total + self.aux_weights["object_center"] * center_loss
        if object_size is not None and "target_size" in aux:
            size_loss = F.mse_loss(aux["target_size"], object_size.float())
            losses["object_size"] = size_loss
            total = total + self.aux_weights["object_size"] * size_loss
        if object_type is not None and "target_type_logits" in aux:
            type_loss = F.cross_entropy(
                aux["target_type_logits"], object_type.long().squeeze(-1)
            )
            losses["object_type"] = type_loss
            total = total + self.aux_weights["object_type"] * type_loss
        losses["total"] = total
        return total, losses
