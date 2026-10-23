"""Train the v6.2.1 Visual CFM (grasp-place, target-object encoding).

Progressive target-association stages (internal checkpoint names in parentheses):
  1. Ground-truth-guided target representation learning (privileged_aux).
  2. Prediction-driven target association learning (visual_distill).
  3. End-to-end joint fine-tuning at a lower learning rate (visual_finetune).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(flow_cfg: dict, device: str):
    from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher

    model_cfg = flow_cfg["model"]
    encoder_cfg = flow_cfg["obstacle_encoder"]
    vision_cfg = flow_cfg["vision_encoder"]
    return ObstacleConditionalFlowMatcher(
        condition_dim=int(flow_cfg["base_condition_dim"]),
        trajectory_points=int(flow_cfg["trajectory_shape"][0]),
        dof=int(flow_cfg["trajectory_shape"][1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
        obstacle_hidden_dim=int(encoder_cfg["hidden_dim"]),
        obstacle_layers=int(encoder_cfg["transformer_layers"]),
        obstacle_heads=int(encoder_cfg["transformer_heads"]),
        vision_temporal_frames=int(vision_cfg["temporal_frames"]),
        vision_camera_views=int(vision_cfg["camera_views"]),
        vision_backbone=str(vision_cfg["backbone"]),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
        goal_condition_dof=int(flow_cfg.get("goal_condition_dof", 0)),
        vision_condition_dim=int(vision_cfg.get("condition_cond_dim", 0)),
        use_target_heads=bool(vision_cfg.get("use_target_heads", False)),
        target_type_dim=int(vision_cfg.get("target_type_dim", 5)),
        fuse_point_cloud=bool(vision_cfg.get("fuse_point_cloud", False)),
        point_cloud_classifier=bool(vision_cfg.get("point_cloud_classifier", False)),
    ).to(device)


def load_v621_initial_state(
    model: torch.nn.Module,
    checkpoint: dict,
    reinitialize_vision_backbone: bool = False,
) -> dict:
    """Load a v6.2 checkpoint; v6.2.1-only keys are randomly initialized."""
    state = dict(checkpoint["model_state_dict"])
    if reinitialize_vision_backbone:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("vision_encoder.backbone.")
            and not key.startswith("vision_encoder.output.")
            and not key.startswith("vision_encoder.film.")
            and not key.startswith("vision_encoder.target_")
        }
    first_conv_key = "vision_encoder.backbone.0.weight"
    if first_conv_key in state:
        source = state[first_conv_key]
        target = model.state_dict()[first_conv_key]
        if source.shape[1] != target.shape[1]:
            if source.shape[1] % target.shape[1] != 0:
                raise ValueError(
                    f"cannot adapt first vision convolution {tuple(source.shape)} "
                    f"to {tuple(target.shape)}"
                )
            source_views = source.shape[1] // target.shape[1]
            temporal_frames = 2
            channels_per_view = target.shape[1] // temporal_frames
            selected = []
            for temporal in range(temporal_frames):
                start = temporal * source_views * channels_per_view
                selected.append(source[:, start : start + channels_per_view])
            state[first_conv_key] = torch.cat(selected, dim=1)
    result = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "vision_encoder.backbone.",
        "vision_encoder.output.",
        "obstacle_encoder.classifier.",
        "vision_encoder.film.",
        "vision_encoder.target_center_head.",
        "vision_encoder.target_size_head.",
        "vision_encoder.target_type_head.",
        "combined_projection.",
    )
    disallowed_missing = [
        key
        for key in result.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or result.unexpected_keys:
        raise ValueError(
            f"incompatible v6.2 checkpoint: missing={disallowed_missing}, "
            f"unexpected={result.unexpected_keys}"
        )
    return {
        "loaded_keys": len(state),
        "new_v621_keys": list(result.missing_keys),
        "unexpected": result.unexpected_keys,
    }


def main() -> None:
    from src.datasets.npz_dataset import VisualCFMV621Dataset
    from src.robot.torch_kinematics import DifferentiableZBotFK
    from src.training.losses import VisualCFMV621Loss, augment_condition_qik
    from src.utils.config import load_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "visual_v621")
    parser.add_argument(
        "--flow-config", type=Path, default=PROJECT_ROOT / "configs" / "visual_cfm_v621.yaml"
    )
    parser.add_argument("--robot-config", type=Path, default=PROJECT_ROOT / "configs" / "robot.yaml")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "visual_cfm_v62_left.pt",
        help="v6.2 visual checkpoint for warm start (may be missing)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override all phase epochs")
    parser.add_argument(
        "--stage-epochs",
        default="",
        help="comma-separated privileged,predicted-target,finetune epochs",
    )
    parser.add_argument(
        "--loss-ablation",
        choices=(
            "none",
            "no_terminal",
            "no_environment_collision",
            "no_self_collision",
            "no_dynamics",
            "no_auxiliary",
        ),
        default="none",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="deterministic fraction of physical training tasks; keeps grasp/place pairs",
    )
    parser.add_argument(
        "--vision-view-index",
        type=int,
        default=None,
        help="train on one camera view; config camera_views must equal 1",
    )
    parser.add_argument("--zero-geometry-input", action="store_true")
    parser.add_argument("--disable-film-condition", action="store_true")
    parser.add_argument("--single-expert", action="store_true")
    parser.add_argument("--reinitialize-vision-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "checkpoints" / "visual_cfm_v621_left.pt")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs" / "logs" / "visual_cfm_v621_training.json")
    parser.add_argument("--save-stage-checkpoints", action="store_true")
    args = parser.parse_args()

    flow_cfg = load_yaml(args.flow_config)
    if args.loss_ablation == "no_terminal":
        for name in ("goal_joint_consistency", "final_tcp_position", "final_tcp_orientation"):
            flow_cfg["loss_weights"][name] = 0.0
    elif args.loss_ablation == "no_environment_collision":
        flow_cfg["loss_weights"]["collision_clearance"] = 0.0
    elif args.loss_ablation == "no_self_collision":
        flow_cfg["loss_weights"]["self_collision_clearance"] = 0.0
    elif args.loss_ablation == "no_dynamics":
        for name in ("velocity_smoothness", "acceleration_smoothness", "jerk_regularization"):
            flow_cfg["loss_weights"][name] = 0.0
    elif args.loss_ablation == "no_auxiliary":
        flow_cfg["aux_weights"] = {
            name: 0.0 for name in flow_cfg.get("aux_weights", {})
        }
    robot_cfg = load_yaml(args.robot_config)
    set_seed(int(flow_cfg["seed"]))
    model = build_model(flow_cfg, args.device)
    model.disable_film_condition = args.disable_film_condition
    initialization_report: dict | None = None
    if args.init_checkpoint.exists():
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        initialization_report = load_v621_initial_state(
            model,
            checkpoint,
            reinitialize_vision_backbone=args.reinitialize_vision_backbone,
        )
        print(f"[INIT] loaded {args.init_checkpoint.name}: {initialization_report}", flush=True)

    train_data = VisualCFMV621Dataset(
        sorted((args.data_dir / "left").glob("train_shard_*.npz")),
        vision_view_index=args.vision_view_index,
    )
    validation_data = VisualCFMV621Dataset(
        sorted((args.data_dir / "left").glob("validation_shard_*.npz")),
        vision_view_index=args.vision_view_index,
    )
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if args.train_fraction < 1.0:
        physical_tasks = len(train_data) // len(train_data.stages)
        rng = np.random.default_rng(int(flow_cfg["seed"]))
        selected_tasks = rng.permutation(physical_tasks)[
            : max(1, round(physical_tasks * args.train_fraction))
        ]
        selected_indices = sorted(
            task * len(train_data.stages) + stage
            for task in selected_tasks
            for stage in range(len(train_data.stages))
        )
        train_data = Subset(train_data, selected_indices)
    batch_size = args.batch_size or int(flow_cfg["training"]["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    print(
        f"[DATA] train={len(train_data)} validation={len(validation_data)} "
        f"batch={batch_size}",
        flush=True,
    )

    limits = robot_cfg["planning_soft_limits"]
    loss_names = (
        "flow",
        "start_consistency",
        "goal_joint_consistency",
        "final_tcp_position",
        "final_tcp_orientation",
        "velocity_smoothness",
        "acceleration_smoothness",
        "jerk_regularization",
        "duration",
        "collision_clearance",
        "self_collision_clearance",
    )
    loss_weights = {
        name: float(flow_cfg["loss_weights"].get(name, 0.0)) for name in loss_names
    }
    criterion = VisualCFMV621Loss(
        fk_model=DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(args.device),
        weights=loss_weights,
        velocity_limits=torch.tensor(limits["velocity"]),
        acceleration_limits=torch.tensor(limits["acceleration"]),
        jerk_limits=torch.tensor(limits["jerk"]),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
        aux_weights={name: float(value) for name, value in flow_cfg.get("aux_weights", {}).items()},
    ).to(args.device)

    training_cfg = flow_cfg["training"]
    if args.stage_epochs:
        stage_epochs = [int(value) for value in args.stage_epochs.split(",")]
        if len(stage_epochs) != 3 or any(value < 0 for value in stage_epochs):
            raise ValueError("stage_epochs must contain three non-negative integers")
    else:
        stage_epochs = [int(args.epochs)] * 3 if args.epochs else list(training_cfg["epochs_per_stage"])
    base_lr = float(training_cfg["learning_rate"])
    finetune_lr = float(training_cfg.get("finetune_learning_rate", base_lr * 0.1))
    history: list[dict] = []
    best_validation = float("inf")
    stage_checkpoints: dict[str, str] = {}
    start_time = perf_counter()

    def run_phase(
        phase_name: str,
        epochs: int,
        learning_rate: float,
        privileged_target: bool,
    ) -> None:
        nonlocal best_validation
        stage_best_validation = float("inf")
        criterion.privileged_target = privileged_target
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=float(training_cfg["weight_decay"]),
        )
        for epoch in range(1, epochs + 1):
            epoch_values: dict[str, float] = {"phase": phase_name, "epoch": epoch}
            for split, loader in (("train", train_loader), ("validation", validation_loader)):
                cpu_rng_state = None
                cuda_rng_states = None
                if split == "validation":
                    cpu_rng_state = torch.random.get_rng_state()
                    cuda_rng_states = torch.cuda.get_rng_state_all()
                    validation_seed = int(flow_cfg["seed"]) + 900_000
                    torch.manual_seed(validation_seed)
                    torch.cuda.manual_seed_all(validation_seed)
                model.train(split == "train")
                model.obstacle_encoder.train(split == "train")
                total = 0.0
                aux_total = 0.0
                count = 0
                for batch in loader:
                    condition = batch["condition"].to(args.device)
                    tokens = batch["obstacle_tokens"].to(args.device)
                    if args.zero_geometry_input:
                        tokens = torch.zeros_like(tokens)
                    mask = batch["obstacle_mask"].to(args.device)
                    trajectory = batch["trajectory"].to(args.device)
                    duration = batch["duration"].to(args.device)
                    rgb = batch["rgb"].to(args.device)
                    depth = batch["depth"].to(args.device)
                    semantic = batch["semantic"].to(args.device)
                    if split == "train":
                        rgb = torch.clamp(
                            rgb.float() + torch.randn_like(rgb.float()) * 2.0, 0.0, 255.0
                        )
                        depth = depth.float() + torch.randn_like(depth.float()) * 0.003
                        if bool(training_cfg.get("qik_augmentation", False)):
                            condition = augment_condition_qik(
                                condition,
                                batch["q_ik_candidates"].to(args.device),
                                criterion.fk_model,
                                int(getattr(model, "goal_condition_dof", 0) or 0),
                                qik_candidate_mask=(
                                    batch["q_ik_candidate_mask"].to(args.device)
                                    if "q_ik_candidate_mask" in batch
                                    else None
                                ),
                            )
                        optimizer.zero_grad(set_to_none=True)
                    kwargs = {
                        "target_component_score": batch.get("target_component_score"),
                        "object_center": batch.get("object_center_foot0"),
                        "object_size": batch.get("object_size"),
                        "object_type": batch.get("object_type"),
                    }
                    for key, value in kwargs.items():
                        if value is not None:
                            kwargs[key] = value.to(args.device)
                    with torch.set_grad_enabled(split == "train"):
                        loss, metrics = criterion(
                            model,
                            trajectory,
                            condition,
                            tokens,
                            mask,
                            duration,
                            vision=(rgb, depth, semantic),
                            expert_trajectories=(
                                batch["q_experts"].to(args.device)
                                if "q_experts" in batch and not args.single_expert
                                else None
                            ),
                            **kwargs,
                        )
                    if split == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad],
                            float(training_cfg.get("max_grad_norm", 1.0)),
                        )
                        optimizer.step()
                    batch_count = condition.shape[0]
                    total += float(loss.detach()) * batch_count
                    aux_total += float(metrics.get("target_component", torch.zeros((), device=args.device)).detach()) * batch_count
                    count += batch_count
                epoch_values[split] = total / count
                epoch_values[f"{split}_aux"] = aux_total / count
                if split == "validation":
                    torch.random.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            history.append(dict(epoch_values))
            marker = ""
            if epoch_values["validation"] < best_validation:
                best_validation = epoch_values["validation"]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": phase_name,
                        "model_state_dict": model.state_dict(),
                        "best_validation_loss": best_validation,
                        "config": flow_cfg,
                        "initialized_from": str(args.init_checkpoint),
                        "initialization_report": initialization_report,
                    },
                    args.output,
                )
                marker = " *"
            if args.save_stage_checkpoints and epoch_values["validation"] < stage_best_validation:
                stage_best_validation = epoch_values["validation"]
                stage_path = args.output.with_name(
                    f"{args.output.stem}_{phase_name}{args.output.suffix}"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": phase_name,
                        "model_state_dict": model.state_dict(),
                        "stage_best_validation_loss": stage_best_validation,
                        "config": flow_cfg,
                        "initialized_from": str(args.init_checkpoint),
                        "initialization_report": initialization_report,
                    },
                    stage_path,
                )
                stage_checkpoints[phase_name] = str(stage_path)
            print(
                f"[{phase_name}] ep {epoch}/{epochs} train={epoch_values['train']:.4f} "
                f"val={epoch_values['validation']:.4f} aux={epoch_values['train_aux']:.4f}{marker}",
                flush=True,
            )

    stage_names = ("privileged_aux", "visual_distill", "visual_finetune")
    lrs = (base_lr, base_lr, finetune_lr)
    privileged = (True, False, False)
    for stage_name, epochs, learning_rate, is_privileged in zip(
        stage_names, stage_epochs, lrs, privileged
    ):
        if epochs > 0:
            run_phase(stage_name, int(epochs), learning_rate, is_privileged)

    final_checkpoint = None
    if args.save_stage_checkpoints:
        final_path = args.output.with_name(f"{args.output.stem}_final{args.output.suffix}")
        torch.save(
            {
                "epoch": int(stage_epochs[-1]),
                "stage": "visual_finetune_final",
                "model_state_dict": model.state_dict(),
                "config": flow_cfg,
                "initialized_from": str(args.init_checkpoint),
                "initialization_report": initialization_report,
            },
            final_path,
        )
        final_checkpoint = str(final_path)

    report = {
        "data_dir": str(args.data_dir),
        "flow_config": str(args.flow_config),
        "init_checkpoint": str(args.init_checkpoint),
        "initialization_report": initialization_report,
        "batch_size": batch_size,
        "train_fraction": args.train_fraction,
        "vision_view_index": args.vision_view_index,
        "loss_ablation": args.loss_ablation,
        "zero_geometry_input": args.zero_geometry_input,
        "disable_film_condition": args.disable_film_condition,
        "single_expert": args.single_expert,
        "reinitialize_vision_backbone": args.reinitialize_vision_backbone,
        "train_items": len(train_data),
        "validation_items": len(validation_data),
        "stages": stage_names,
        "epochs_per_stage": [int(value) for value in stage_epochs],
        "history": history,
        "best_validation_loss": best_validation,
        "stage_checkpoints": stage_checkpoints,
        "final_checkpoint": final_checkpoint,
        "elapsed_s": perf_counter() - start_time,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
