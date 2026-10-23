"""Replay ranked visual candidates in PhysX while the H1 balance policy runs."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
import tensordict  # noqa: F401
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate_bank", type=Path, required=True)
parser.add_argument("--balance_checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_tasks", type=int, default=0)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--hold_steps", type=int, default=10)
parser.add_argument("--contact_impulse_threshold", type=float, default=1.0e-4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from h1_1111_grasp.env_cfg import H11111BalanceEnvCfg, H11111PPORunnerCfg


CONTROL_DT = 0.02


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _wilson(successes: int, tasks: int) -> list[float]:
    if tasks == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / tasks
    denominator = 1.0 + z * z / tasks
    center = (p + z * z / (2.0 * tasks)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / tasks + z * z / (4.0 * tasks * tasks)) / denominator
    return [center - half, center + half]


def _interpolate_path(q: np.ndarray, duration: float, time_s: float) -> np.ndarray:
    continuous = np.unwrap(np.asarray(q, dtype=np.float64), axis=0, period=2.0 * np.pi)
    sample_time = np.linspace(0.0, float(duration), len(continuous))
    return np.asarray(
        [np.interp(time_s, sample_time, continuous[:, joint]) for joint in range(6)],
        dtype=np.float32,
    )


def main() -> None:
    if not args.candidate_bank.is_file():
        raise FileNotFoundError(args.candidate_bank)
    if not args.balance_checkpoint.is_file():
        raise FileNotFoundError(args.balance_checkpoint)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = np.load(args.candidate_bank, allow_pickle=False)
    task_count = len(data["q"])
    if args.max_tasks:
        task_count = min(task_count, args.max_tasks)

    env_cfg = H11111BalanceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 3600.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.move_dual_aux_arms = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.low_height = None
    env_cfg.terminations.excessive_tilt = None
    env_cfg.terminations.base_xy_drift = None

    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        H11111PPORunnerCfg(), metadata.version("rsl-rl-lib")
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw_env.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(
        str(args.balance_checkpoint), metadata.version("rsl-rl-lib")
    )
    runner.load(
        checkpoint,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
    )
    policy = runner.get_inference_policy(device=raw_env.device)
    robot = raw_env.scene["robot"]

    active_side = ""
    contact_events: list[dict[str, object]] = []

    def on_contacts(headers, contact_data) -> None:
        marker = f"/ZBot6S_{active_side}/"
        if not active_side:
            return
        for header in headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            if marker not in actor0 and marker not in actor1:
                continue
            if "/Robotiq2F85/" in actor0 and "/Robotiq2F85/" in actor1:
                continue
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            impulse = sum(float(contact_data[index].impulse) for index in range(start, stop))
            if impulse < args.contact_impulse_threshold:
                continue
            contact_events.append(
                {
                    "actor0": actor0,
                    "actor1": actor1,
                    "collider0": str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    "collider1": str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                    "impulse": impulse,
                }
            )

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
    records = []
    for task_index in range(task_count):
        active_side = str(data["side"][task_index])
        suffix = "_0" if active_side == "left" else ""
        arm_ids, arm_names = robot.find_joints(
            [f"^joint{joint}{suffix}$" for joint in range(1, 7)], preserve_order=True
        )
        if len(arm_ids) != 6:
            raise RuntimeError(f"Failed to resolve {active_side} arm joints: {arm_names}")
        candidate_records = []
        selected_rank = None
        for rank in range(data["q"].shape[1]):
            obs, _ = env.reset()
            q_path = np.asarray(data["q"][task_index, rank], dtype=np.float32)
            duration = float(data["duration"][task_index, rank])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            joint_pos[:, arm_ids] = torch.as_tensor(
                q_path[0], dtype=joint_pos.dtype, device=raw_env.device
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            contact_events.clear()
            max_tilt = 0.0
            min_height = float("inf")
            max_xy = 0.0

            def step_with_target(target: np.ndarray) -> None:
                nonlocal obs, max_tilt, min_height, max_xy
                robot.set_joint_position_target(
                    torch.as_tensor(target, dtype=torch.float32, device=raw_env.device)[None],
                    joint_ids=arm_ids,
                )
                with torch.inference_mode():
                    h1_action = policy(obs)
                obs, _, _, _ = env.step(h1_action)
                tilt = float(
                    torch.acos(
                        torch.clamp(-robot.data.projected_gravity_b[0, 2], -1.0, 1.0)
                    )
                )
                height = float(robot.data.root_pos_w[0, 2] - raw_env.scene.env_origins[0, 2])
                xy = float(
                    torch.linalg.vector_norm(
                        robot.data.root_pos_w[0, :2] - raw_env.scene.env_origins[0, :2]
                    )
                )
                max_tilt = max(max_tilt, tilt)
                min_height = min(min_height, height)
                max_xy = max(max_xy, xy)

            for _ in range(args.settle_steps):
                step_with_target(q_path[0])
            playback_steps = max(1, int(math.ceil(duration / CONTROL_DT)))
            emergency_stop = False
            for step in range(playback_steps + args.hold_steps):
                time_s = min(step * CONTROL_DT, duration)
                step_with_target(_interpolate_path(q_path, duration, time_s))
                if contact_events:
                    emergency_stop = True
                    break
            actual = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
            final_error = float(
                np.max(np.abs(_wrap_angles(actual - _wrap_angles(q_path[-1]))))
            )
            fell = min_height < 0.65 or max_tilt > 0.85 or max_xy > 0.30
            physx_safe = not emergency_stop and not fell and final_error <= 0.15
            candidate_records.append(
                {
                    "rank": rank,
                    "visual_geometric_success": bool(
                        data["geometric_success"][task_index, rank]
                    ),
                    "physx_safe": physx_safe,
                    "emergency_stop": emergency_stop,
                    "fell": fell,
                    "max_tilt_rad": max_tilt,
                    "min_base_height_m": min_height,
                    "max_base_xy_m": max_xy,
                    "final_joint_error_rad": final_error,
                    "contact_events": list(contact_events),
                }
            )
            if physx_safe:
                selected_rank = rank
                break
        records.append(
            {
                "task_index": task_index,
                "sample_id": str(data["sample_id"][task_index]),
                "side": active_side,
                "selected_rank": selected_rank,
                "physx_success": selected_rank is not None,
                "candidates": candidate_records,
            }
        )
        print(
            f"[PHYSX] {task_index + 1}/{task_count} side={active_side} "
            f"selected_rank={selected_rank}",
            flush=True,
        )

    successes = sum(record["physx_success"] for record in records)
    top1_successes = sum(
        record["candidates"][0]["physx_safe"] for record in records
    )
    collision_rejections = sum(
        candidate["emergency_stop"]
        for record in records
        for candidate in record["candidates"]
    )
    falls = sum(
        candidate["fell"] for record in records for candidate in record["candidates"]
    )
    report = {
        "candidate_bank": str(args.candidate_bank),
        "balance_checkpoint": str(args.balance_checkpoint),
        "seed": args.seed,
        "tasks": task_count,
        "candidate_bank_size": int(data["q"].shape[1]),
        "control_frequency_hz": 1.0 / CONTROL_DT,
        "top1_physx_successes": top1_successes,
        "top1_physx_success_rate": top1_successes / task_count,
        "multi_candidate_physx_successes": successes,
        "multi_candidate_physx_success_rate": successes / task_count,
        "multi_candidate_physx_success_wilson_95": _wilson(successes, task_count),
        "collision_rejections": collision_rejections,
        "fall_events": falls,
        "contact_policy": (
            "Contact-report pairs involving the active auxiliary arm; internal gripper "
            "mechanism contacts excluded; fixed-joint reaction forces are not collision labels."
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    subscription = None
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
