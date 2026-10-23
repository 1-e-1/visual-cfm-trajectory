"""Train condition-only CFM or DDIM trajectory baselines on frozen v6.2.2 data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shortest_delta(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 1.0, 2.0) - 1.0


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    model_type: str,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    condition = batch["condition"].to(device)
    expert_bank = batch.get("q_experts")
    if expert_bank is not None:
        expert_bank = expert_bank.to(device)
        labels = torch.randint(
            expert_bank.shape[1],
            (expert_bank.shape[0],),
            device=device,
            generator=generator,
        )
        trajectory = expert_bank[torch.arange(expert_bank.shape[0], device=device), labels]
    else:
        trajectory = batch["trajectory"].to(device)
    target = shortest_delta(trajectory / model.joint_scale)
    duration_target = batch["duration"].to(device)
    batch_size = target.shape[0]

    if model_type == "condition_cfm":
        noise = 2.0 * torch.rand(
            target.shape,
            device=device,
            dtype=target.dtype,
            generator=generator,
        ) - 1.0
        time = torch.rand(
            batch_size, device=device, dtype=target.dtype, generator=generator
        )
        target_velocity = shortest_delta(target - noise)
        noisy = shortest_delta(noise + time[:, None, None] * target_velocity)
        prediction, duration = model(noisy, time, condition)
        primary = F.mse_loss(prediction, target_velocity)
        reconstructed = shortest_delta(
            noisy + (1.0 - time[:, None, None]) * prediction
        )
    else:
        timestep = torch.randint(
            model.diffusion_steps,
            (batch_size,),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            target.shape,
            device=device,
            dtype=target.dtype,
            generator=generator,
        )
        alpha = model.alphas_bar[timestep][:, None, None]
        noisy = torch.sqrt(alpha) * target + torch.sqrt(1.0 - alpha) * noise
        time = timestep.float() / (model.diffusion_steps - 1)
        prediction, duration = model(noisy, time, condition)
        primary = F.mse_loss(prediction, noise)
        reconstructed = (
            noisy - torch.sqrt(1.0 - alpha) * prediction
        ) / torch.sqrt(alpha).clamp_min(1.0e-4)
        reconstructed = reconstructed.clamp(-1.0, 1.0)

    start_target = condition[:, :6] / model.joint_scale
    goal_target = condition[:, 6:12] / model.joint_scale
    start_loss = torch.mean(shortest_delta(reconstructed[:, 0] - start_target) ** 2)
    goal_loss = torch.mean(shortest_delta(reconstructed[:, -1] - goal_target) ** 2)
    projected = reconstructed * model.joint_scale
    delta = shortest_delta(torch.diff(projected, dim=1) / model.joint_scale) * model.joint_scale
    velocity_loss = torch.mean(delta**2)
    acceleration_loss = torch.mean(torch.diff(delta, dim=1) ** 2)
    jerk_loss = torch.mean(torch.diff(delta, n=2, dim=1) ** 2)
    duration_loss = F.mse_loss(duration, duration_target)
    total = (
        10.0 * primary
        + 5.0 * start_loss
        + 10.0 * goal_loss
        + 0.02 * velocity_loss
        + 0.02 * acceleration_loss
        + 0.01 * jerk_loss
        + 0.5 * duration_loss
    )
    values = {
        "total": float(total.detach()),
        "primary": float(primary.detach()),
        "goal": float(goal_loss.detach()),
        "duration": float(duration_loss.detach()),
    }
    return total, values


def main() -> None:
    from src.datasets.npz_dataset import VisualCFMV621Dataset
    from src.models.conditional_trajectory_baselines import build_condition_only_model
    from src.utils.config import load_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("condition_cfm", "condition_ddim"), required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "visual_v622_pose" / "left")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "visual_cfm_v622_pose.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_yaml(args.config)
    model = build_condition_only_model(config, args.model_type).to(args.device)
    train = VisualCFMV621Dataset(sorted(args.data_dir.glob("train_shard_*.npz")))
    validation = VisualCFMV621Dataset(sorted(args.data_dir.glob("validation_shard_*.npz")))
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if args.train_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(train))[: max(1, round(len(train) * args.train_fraction))]
        train = Subset(train, sorted(indices.tolist()))
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_generator = torch.Generator(device=args.device).manual_seed(args.seed + 17)
    history = []
    best_validation = float("inf")
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        record: dict[str, float | int] = {"epoch": epoch}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            if split == "validation":
                validation_generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + 900_000
                )
            else:
                validation_generator = loss_generator
            model.train(split == "train")
            totals = {"total": 0.0, "primary": 0.0, "goal": 0.0, "duration": 0.0}
            count = 0
            for batch in loader:
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    loss, values = compute_loss(
                        model, batch, args.model_type, args.device, validation_generator
                    )
                if split == "train":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                batch_count = batch["condition"].shape[0]
                count += batch_count
                for key in totals:
                    totals[key] += values[key] * batch_count
            for key, value in totals.items():
                record[f"{split}_{key}"] = value / count
        history.append(record)
        marker = ""
        if float(record["validation_total"]) < best_validation:
            best_validation = float(record["validation_total"])
            torch.save(
                {
                    "model_type": args.model_type,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_validation_loss": best_validation,
                    "train_fraction": args.train_fraction,
                    "seed": args.seed,
                },
                args.output,
            )
            marker = " *"
        if epoch == 1 or epoch % 10 == 0 or marker:
            print(
                f"[{args.model_type}] ep={epoch}/{args.epochs} "
                f"train={record['train_total']:.5f} val={record['validation_total']:.5f}{marker}",
                flush=True,
            )
    report = {
        "model_type": args.model_type,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_items": len(train),
        "validation_items": len(validation),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_loss": best_validation,
        "elapsed_s": perf_counter() - started,
        "checkpoint": str(args.output),
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
