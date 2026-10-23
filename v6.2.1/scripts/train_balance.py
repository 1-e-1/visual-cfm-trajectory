"""Fine-tune H1 balance for the corrected 1111 dual-arm equipment."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import sys
import time

import tensordict  # noqa: F401
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("payload", "manipulation", "robust"), default="payload")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--run_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from h1_1111_grasp.constants import OUTPUT_DIR
from h1_1111_grasp.env_cfg import (
    H11111BalanceEnvCfg,
    H11111ManipulationEnvCfg,
    H11111PPORunnerCfg,
    H11111RobustEnvCfg,
)


def main() -> None:
    torch.manual_seed(args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    env_cls = {
        "payload": H11111BalanceEnvCfg,
        "manipulation": H11111ManipulationEnvCfg,
        "robust": H11111RobustEnvCfg,
    }[args.stage]
    env_cfg = env_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = H11111PPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    run_name = args.run_name or f"{args.stage}_{args.num_envs}x{args.max_iterations}"
    log_dir = OUTPUT_DIR / "balance" / f"{time.strftime('%Y%m%d_%H%M%S')}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.unwrapped.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(
        str(args.checkpoint), metadata.version("rsl-rl-lib")
    )
    runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    runner.save(str(log_dir / "model_final.pt"))
    env.close()
    print(f"[OK] training_output={log_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

