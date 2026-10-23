"""Custom MDP terms for 1111-scene payload-aware H1 balance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_dual_aux_arms(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    default_positions: list[float],
    position_noise: float,
    lower_limits: list[float],
    upper_limits: list[float],
) -> None:
    ids = _ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    default = torch.tensor(default_positions, device=env.device, dtype=asset.data.joint_pos.dtype)
    position = default.unsqueeze(0).repeat(len(ids), 1)
    position += (2.0 * torch.rand_like(position) - 1.0) * position_noise
    lower = torch.tensor(lower_limits, device=env.device, dtype=position.dtype)
    upper = torch.tensor(upper_limits, device=env.device, dtype=position.dtype)
    position = torch.clamp(position, lower, upper)
    velocity = torch.zeros_like(position)
    asset.write_joint_state_to_sim(position, velocity, joint_ids=joint_ids, env_ids=ids)
    asset.set_joint_position_target(position, joint_ids=joint_ids, env_ids=ids)


def nudge_dual_aux_arm_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_delta: float,
    lower_limits: list[float],
    upper_limits: list[float],
) -> None:
    ids = _ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    current = asset.data.joint_pos_target[ids][:, joint_ids]
    delta = (2.0 * torch.rand_like(current) - 1.0) * max_delta
    target = current + delta
    lower = torch.tensor(lower_limits, device=env.device, dtype=target.dtype)
    upper = torch.tensor(upper_limits, device=env.device, dtype=target.dtype)
    target = torch.clamp(target, lower, upper)
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=ids)


def base_xy_relative(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square((height - target_height) / std))


def base_xy_position_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.exp(-torch.sum(torch.square(base_xy_relative(env, asset_cfg) / std), dim=1))


def base_xy_position_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return torch.sum(torch.square(base_xy_relative(env, asset_cfg)), dim=1)


def base_xy_drift_exceeded(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.linalg.vector_norm(base_xy_relative(env, asset_cfg), dim=1) > maximum_distance


def aux_arm_joint_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def aux_arm_joint_acceleration_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

