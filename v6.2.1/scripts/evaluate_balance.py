"""Evaluate dual-arm H1 balance with deterministic, task-level metrics."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

import tensordict  # noqa: F401
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--stage", choices=("payload", "manipulation", "robust"), default="robust")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from h1_1111_grasp.constants import DUAL_ARM_JOINT_PATTERN
from h1_1111_grasp.env_cfg import (
    H11111BalanceEnvCfg,
    H11111ManipulationEnvCfg,
    H11111PPORunnerCfg,
    H11111RobustEnvCfg,
)


def _statistics(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.float().flatten()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
        f"{prefix}_max": float(values.max().item()),
    }


def main() -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    torch.manual_seed(args.seed)
    env_cls = {
        "payload": H11111BalanceEnvCfg,
        "manipulation": H11111ManipulationEnvCfg,
        "robust": H11111RobustEnvCfg,
    }[args.stage]
    env_cfg = env_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args.steps * env_cfg.sim.dt * env_cfg.decimation + 1.0

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        H11111PPORunnerCfg(), metadata.version("rsl-rl-lib")
    )
    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw_env.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(
        str(args.checkpoint), metadata.version("rsl-rl-lib")
    )
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
    )
    policy = runner.get_inference_policy(device=raw_env.device)

    robot = raw_env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_PATTERN, preserve_order=True)
    arm_cfg.resolve(raw_env.scene)
    obs, _ = env.reset()
    ever_failed = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
    termination_counts = {name: 0 for name in raw_env.termination_manager.active_terms}
    height_samples: list[torch.Tensor] = []
    tilt_samples: list[torch.Tensor] = []
    xy_samples: list[torch.Tensor] = []
    arm_error_samples: list[torch.Tensor] = []
    arm_speed_samples: list[torch.Tensor] = []
    arm_acceleration_samples: list[torch.Tensor] = []
    action_rate_samples: list[torch.Tensor] = []
    previous_action = torch.zeros((args.num_envs, 19), device=raw_env.device)

    with torch.inference_mode():
        for _ in range(args.steps):
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            step_failure = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
            for name in termination_counts:
                term = raw_env.termination_manager.get_term(name).bool()
                termination_counts[name] += int(term.sum().item())
                if name != "time_out":
                    step_failure |= term
            ever_failed |= step_failure

            relative_height = robot.data.root_pos_w[:, 2] - raw_env.scene.env_origins[:, 2]
            projected_gravity = robot.data.projected_gravity_b
            tilt = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
            xy = torch.linalg.vector_norm(
                robot.data.root_pos_w[:, :2] - raw_env.scene.env_origins[:, :2], dim=1
            )
            arm_position = robot.data.joint_pos[:, arm_cfg.joint_ids]
            arm_target = robot.data.joint_pos_target[:, arm_cfg.joint_ids]
            height_samples.append(relative_height.clone())
            tilt_samples.append(tilt.clone())
            xy_samples.append(xy.clone())
            arm_error_samples.append(torch.abs(arm_position - arm_target).clone())
            arm_speed_samples.append(torch.abs(robot.data.joint_vel[:, arm_cfg.joint_ids]).clone())
            arm_acceleration_samples.append(torch.abs(robot.data.joint_acc[:, arm_cfg.joint_ids]).clone())
            action_rate_samples.append(torch.abs(action - previous_action).clone())
            previous_action = action

    height = torch.cat(height_samples)
    tilt = torch.cat(tilt_samples)
    xy = torch.cat(xy_samples)
    arm_error = torch.cat(arm_error_samples)
    arm_speed = torch.cat(arm_speed_samples)
    arm_acceleration = torch.cat(arm_acceleration_samples)
    action_rate = torch.cat(action_rate_samples)
    non_timeout_failures = sum(
        count for name, count in termination_counts.items() if name != "time_out"
    )
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "stage": args.stage,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "duration_per_env_s": args.steps * env_cfg.sim.dt * env_cfg.decimation,
        "observation_shape": [args.num_envs, 95],
        "action_dim": 19,
        "failure_free_env_rate": float((~ever_failed).float().mean().item()),
        "non_timeout_failure_events": non_timeout_failures,
        "termination_counts_by_term": termination_counts,
        **_statistics(height, "base_height_m"),
        **_statistics(tilt, "base_tilt_rad"),
        **_statistics(xy, "base_xy_displacement_m"),
        **_statistics(arm_error, "aux_joint_abs_tracking_error_rad"),
        **_statistics(arm_speed, "aux_joint_abs_speed_rad_s"),
        **_statistics(arm_acceleration, "aux_joint_abs_acceleration_rad_s2"),
        **_statistics(action_rate, "policy_action_abs_step_change"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
