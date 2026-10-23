"""Train the privileged obstacle-conditioned position-planning CFM."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import ObstacleFlowMatchingLoss, augment_condition_qik


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_epoch(
    model,
    criterion,
    loader,
    device,
    optimizer=None,
    amp_dtype=None,
    max_grad_norm=1.0,
    fk_model=None,
    qik_augment=False,
    generator=None,
):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        condition = batch["condition"].to(device, non_blocking=True)
        tokens = batch["obstacle_tokens"].to(device, non_blocking=True)
        mask = batch["obstacle_mask"].to(device, non_blocking=True)
        trajectory = batch["trajectory"].to(device, non_blocking=True)
        duration = batch["duration"].to(device, non_blocking=True)
        if training and qik_augment and fk_model is not None:
            qik_candidates = batch.get("q_ik_candidates", None)
            if qik_candidates is not None:
                qik_candidates = qik_candidates.to(device, non_blocking=True)
            condition = augment_condition_qik(
                condition,
                qik_candidates,
                fk_model,
                int(getattr(model, "goal_condition_dof", 0) or 0),
                generator=generator,
            )
        q_experts = batch.get("q_experts", None)
        if q_experts is not None:
            q_experts = q_experts.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, metrics = criterion(
                model,
                trajectory,
                condition,
                tokens,
                mask,
                duration,
                expert_trajectories=q_experts,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        batch_size = condition.shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        count += batch_size
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    from src.datasets.npz_dataset import ObstacleTrajectoryShardDataset
    from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher
    from src.robot.torch_kinematics import DifferentiableZBotFK
    from src.utils.config import load_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "obstacle_v1")
    parser.add_argument(
        "--flow-config", type=Path, default=PROJECT_ROOT / "configs" / "obstacle_flow_matching.yaml"
    )
    parser.add_argument("--robot-config", type=Path, default=PROJECT_ROOT / "configs" / "robot.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="Load model weights only and start a new fine-tuning run from epoch 1.",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--milestone-every", type=int, default=500)
    parser.add_argument("--cosine-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--loss-weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one loss weight from obstacle_flow_matching.yaml. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_flow",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "obstacle_flow",
    )
    args = parser.parse_args()
    if args.save_every < 1 or args.milestone_every < 1:
        parser.error("Checkpoint intervals must be positive")
    if args.resume_checkpoint is not None and args.initialize_from is not None:
        parser.error("--resume-checkpoint and --initialize-from are mutually exclusive")

    flow_cfg = load_yaml(args.flow_config)
    loss_weight_overrides: dict[str, float] = {}
    for item in args.loss_weight:
        if "=" not in item:
            parser.error(f"--loss-weight must use NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in flow_cfg["loss_weights"]:
            parser.error(f"Unknown loss weight {name!r}; valid keys: {sorted(flow_cfg['loss_weights'])}")
        try:
            parsed_value = float(value)
        except ValueError:
            parser.error(f"Invalid float value for {name!r}: {value!r}")
        flow_cfg["loss_weights"][name] = parsed_value
        loss_weight_overrides[name] = parsed_value
    robot_cfg = load_yaml(args.robot_config)
    seed = int(flow_cfg["seed"])
    set_seed(seed)
    model_cfg = flow_cfg["model"]
    obstacle_encoder_cfg = flow_cfg["obstacle_encoder"]
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model = ObstacleConditionalFlowMatcher(
        condition_dim=int(flow_cfg["base_condition_dim"]),
        trajectory_points=int(flow_cfg["trajectory_shape"][0]),
        dof=int(flow_cfg["trajectory_shape"][1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
        obstacle_hidden_dim=int(obstacle_encoder_cfg["hidden_dim"]),
        obstacle_layers=int(obstacle_encoder_cfg["transformer_layers"]),
        obstacle_heads=int(obstacle_encoder_cfg["transformer_heads"]),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
        goal_condition_dof=int(flow_cfg.get("goal_condition_dof", 0)),
    ).to(args.device)
    if args.initialize_from is not None:
        checkpoint = torch.load(args.initialize_from, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"[INITIALIZE] checkpoint={args.initialize_from} source_epoch={checkpoint.get('epoch')}",
            flush=True,
        )
    fk_model = DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(args.device)
    limits = robot_cfg["planning_soft_limits"]
    loss_weights = {
        name: float(flow_cfg["loss_weights"][name])
        for name in (
            "flow",
            "start_consistency",
            "goal_joint_consistency",
            "final_tcp_position",
            "velocity_smoothness",
            "acceleration_smoothness",
            "jerk_regularization",
            "duration",
            "collision_clearance",
            "self_collision_clearance",
        )
        if name in flow_cfg["loss_weights"]
    }
    criterion = ObstacleFlowMatchingLoss(
        fk_model=fk_model,
        weights=loss_weights,
        velocity_limits=torch.tensor(limits["velocity"]),
        acceleration_limits=torch.tensor(limits["acceleration"]),
        jerk_limits=torch.tensor(limits["jerk"]),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
    ).to(args.device)
    train_dataset = ObstacleTrajectoryShardDataset(sorted(args.data_dir.glob("train_shard_*.npz")))
    validation_dataset = ObstacleTrajectoryShardDataset(sorted(args.data_dir.glob("validation_shard_*.npz")))
    if not train_dataset or not validation_dataset:
        raise RuntimeError(
            f"Training and validation splits must be non-empty: "
            f"train={len(train_dataset)}, validation={len(validation_dataset)}"
        )
    batch_size = args.batch_size or int(flow_cfg["training"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(flow_cfg["training"]["learning_rate"]),
        weight_decay=float(flow_cfg["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(flow_cfg["training"]["epochs_per_stage"][0])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(flow_cfg["training"]["learning_rate"]) * 0.01,
        )
        if args.cosine_lr
        else None
    )
    amp_dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir / "tensorboard")
    csv_path = log_dir / "training.csv"
    best_loss = float("inf")
    start_epoch = 1
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        start_epoch = int(checkpoint["epoch"]) + 1
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("loader_generator_state") is not None:
            generator.set_state(checkpoint["loader_generator_state"].cpu())
        print(
            f"[RESUME] checkpoint={args.resume_checkpoint} start_epoch={start_epoch} "
            f"best_validation={best_loss:.6f}",
            flush=True,
        )
    if start_epoch > epochs:
        raise RuntimeError(f"Checkpoint epoch {start_epoch - 1} already reaches requested epochs={epochs}")
    started = perf_counter()
    metric_names = [
        "flow",
        "start_consistency",
        "final_tcp_position",
        "velocity_smoothness",
        "acceleration_smoothness",
        "jerk_regularization",
        "duration",
        "collision_clearance",
        "goal_joint_consistency",
        "self_collision_clearance",
        "total",
    ]
    append_csv = start_epoch > 1 and csv_path.is_file()
    with csv_path.open("a" if append_csv else "w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(
            stream,
            fieldnames=["epoch"]
            + [f"train_{name}" for name in metric_names]
            + [f"validation_{name}" for name in metric_names],
        )
        if not append_csv:
            writer_csv.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = mean_epoch(
                model,
                criterion,
                train_loader,
                args.device,
                optimizer=optimizer,
                amp_dtype=amp_dtype,
                max_grad_norm=float(flow_cfg["training"]["max_grad_norm"]),
                fk_model=fk_model,
                qik_augment=bool(flow_cfg.get("training", {}).get("qik_augmentation", False)),
                generator=generator,
            )
            with torch.no_grad():
                validation_metrics = mean_epoch(
                    model,
                    criterion,
                    validation_loader,
                    args.device,
                    amp_dtype=amp_dtype,
                )
            writer_csv.writerow(
                {"epoch": epoch}
                | {f"train_{name}": train_metrics[name] for name in metric_names}
                | {f"validation_{name}": validation_metrics[name] for name in metric_names}
            )
            stream.flush()
            for name in metric_names:
                writer.add_scalar(f"train/{name}", train_metrics[name], epoch)
                writer.add_scalar(f"validation/{name}", validation_metrics[name], epoch)
            if scheduler is not None:
                scheduler.step()
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_validation_loss": min(best_loss, validation_metrics["total"]),
                "config": flow_cfg,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "loader_generator_state": generator.get_state(),
            }
            if validation_metrics["total"] < best_loss:
                best_loss = validation_metrics["total"]
                torch.save(payload, args.output_dir / "best.pt")
            if epoch % args.save_every == 0 or epoch == epochs:
                torch.save(payload, args.output_dir / "latest.pt")
            if epoch % args.milestone_every == 0 or epoch == epochs:
                torch.save(payload, args.output_dir / f"epoch_{epoch:04d}.pt")
            print(
                f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
                f"validation={validation_metrics['total']:.6f}",
                flush=True,
            )
    writer.close()
    summary = {
        "status": "success",
        "seed": seed,
        "device": args.device,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "batch_size": batch_size,
        "cosine_lr": args.cosine_lr,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_loss": best_loss,
        "final_train_metrics": train_metrics,
        "final_validation_metrics": validation_metrics,
        "elapsed_s": perf_counter() - started,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "planning_mode": "position_only",
        "loss_weight_overrides": loss_weight_overrides,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
