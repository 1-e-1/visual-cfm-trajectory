"""Distill privileged obstacle state into the wrist RGB-D-semantic encoder."""

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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(flow_cfg: dict, device: str):
    from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher

    model_cfg = flow_cfg["model"]
    encoder_cfg = flow_cfg["obstacle_encoder"]
    vision_cfg = flow_cfg.get("vision_encoder", {})
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
        vision_temporal_frames=int(vision_cfg.get("temporal_frames", 2)),
        vision_camera_views=int(vision_cfg.get("camera_views", 1)),
        vision_backbone=str(vision_cfg.get("backbone", "resnet18_small_input")),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
        goal_condition_dof=int(flow_cfg.get("goal_condition_dof", 0)),
    ).to(device)


def load_initial_state(
    model: torch.nn.Module,
    checkpoint: dict,
    target_view_order: list[str] | None = None,
) -> dict:
    """Load compatible weights and adapt compact visual inputs when possible."""
    state = dict(checkpoint["model_state_dict"])
    key = "vision_encoder.backbone.0.weight"
    source_weight = state[key]
    target_weight = model.state_dict()[key]
    expanded = source_weight.shape != target_weight.shape
    view_mapping = None
    periodic_condition_adapted = False
    condition_key = "condition_encoder.0.weight"
    source_condition = state[condition_key]
    target_condition = model.state_dict()[condition_key]
    if source_condition.shape != target_condition.shape:
        expected_periodic_shape = (
            source_condition.shape[0],
            source_condition.shape[1] + model.dof,
        )
        if tuple(target_condition.shape) != expected_periodic_shape:
            raise ValueError(
                f"cannot adapt condition encoder {tuple(source_condition.shape)} "
                f"to {tuple(target_condition.shape)}"
            )
        adapted = torch.zeros_like(target_condition)
        adapted[:, : model.dof] = source_condition[:, : model.dof] / torch.pi
        adapted[:, 2 * model.dof :] = source_condition[:, model.dof :]
        state[condition_key] = adapted
        periodic_condition_adapted = True
    if expanded:
        source_view_order = list(
            checkpoint.get("config", {}).get("vision_encoder", {}).get("view_order", [])
        )
        target_view_order = list(target_view_order or [])
        temporal_frames = model.vision_encoder.temporal_frames
        modality_channels = 5
        source_channels_per_time = source_weight.shape[1] // temporal_frames
        target_channels_per_time = target_weight.shape[1] // temporal_frames
        can_map_named_views = (
            source_weight.shape[0] == target_weight.shape[0]
            and source_weight.shape[2:] == target_weight.shape[2:]
            and source_channels_per_time % modality_channels == 0
            and target_channels_per_time % modality_channels == 0
            and len(source_view_order) == source_channels_per_time // modality_channels
            and len(target_view_order) == target_channels_per_time // modality_channels
            and set(source_view_order).issubset(target_view_order)
        )
        if can_map_named_views:
            source_views = len(source_view_order)
            target_views = len(target_view_order)
            source_by_view = source_weight.reshape(
                source_weight.shape[0],
                temporal_frames,
                source_views,
                modality_channels,
                *source_weight.shape[2:],
            )
            target_by_view = torch.zeros_like(target_weight).reshape(
                target_weight.shape[0],
                temporal_frames,
                target_views,
                modality_channels,
                *target_weight.shape[2:],
            )
            for source_index, view_name in enumerate(source_view_order):
                target_index = target_view_order.index(view_name)
                target_by_view[:, :, target_index] = source_by_view[:, :, source_index]
            state[key] = target_by_view.reshape_as(target_weight)
            view_mapping = {
                "source": source_view_order,
                "target": target_view_order,
                "new_zero_initialized_views": [
                    view for view in target_view_order if view not in source_view_order
                ],
            }
        else:
            can_expand_views = (
                source_weight.shape[0] == target_weight.shape[0]
                and source_weight.shape[2:] == target_weight.shape[2:]
                and target_weight.shape[1] % source_weight.shape[1] == 0
            )
            if can_expand_views:
                view_factor = target_weight.shape[1] // source_weight.shape[1]
                modalities = source_weight.shape[1] // temporal_frames
                source_weight = source_weight.reshape(
                    source_weight.shape[0], temporal_frames, modalities, *source_weight.shape[2:]
                )
                state[key] = (
                    source_weight.unsqueeze(2)
                    .repeat(1, 1, view_factor, 1, 1, 1)
                    .reshape_as(target_weight)
                    / view_factor
                )
            else:
                state = {
                    name: value
                    for name, value in state.items()
                    if not name.startswith("vision_encoder.")
                }
    incompatible = model.load_state_dict(state, strict=False)
    non_visual_missing = [name for name in incompatible.missing_keys if not name.startswith("vision_encoder.")]
    if non_visual_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"incompatible non-visual checkpoint keys: missing={non_visual_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "input_convolution_expanded": expanded and key in state,
        "periodic_condition_encoder_expanded": periodic_condition_adapted,
        "vision_encoder_reinitialized": key not in state,
        "missing_visual_keys": [
            name for name in incompatible.missing_keys if name.startswith("vision_encoder.")
        ],
        "periodic_condition_encoder_adapted": periodic_condition_adapted,
        "vision_view_mapping": view_mapping,
    }


def validate_vision_dataset(data_dir: Path, flow_cfg: dict, vision_view_index: int | None = None) -> dict:
    train_paths = sorted(data_dir.glob("train_shard_*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No training shards under {data_dir}")
    expected_views = int(flow_cfg.get("vision_encoder", {}).get("camera_views", 1))
    expected_order = list(flow_cfg.get("vision_encoder", {}).get("view_order", ["wrist"]))
    with np.load(train_paths[0], allow_pickle=False) as sample:
        if not {"rgb", "depth", "semantic"}.issubset(sample.files):
            raise ValueError(f"Missing visual tensors in {train_paths[0]}")
        rgb_shape = tuple(sample["rgb"].shape)
        actual_views = int(rgb_shape[1]) if len(rgb_shape) == 6 else 1
        actual_order = (
            [str(item) for item in sample["camera_view_names"].tolist()]
            if "camera_view_names" in sample.files
            else ["wrist"]
        )
    if vision_view_index is not None:
        if not 0 <= vision_view_index < actual_views:
            raise ValueError(f"vision view index {vision_view_index} is outside {actual_views} dataset views")
        selected_order = [actual_order[vision_view_index]]
        if expected_views != 1 or selected_order != expected_order:
            raise ValueError(f"model expects {expected_order}, selected dataset view is {selected_order}")
    else:
        if actual_views != expected_views:
            raise ValueError(f"expected {expected_views} camera views, dataset contains {actual_views}")
        if actual_order != expected_order:
            raise ValueError(f"expected camera order {expected_order}, dataset contains {actual_order}")
        selected_order = actual_order
    return {
        "rgb_sample_shape": list(rgb_shape[2:] if vision_view_index is not None else rgb_shape[1:]),
        "camera_views": expected_views,
        "view_order": selected_order,
        "source_camera_views": actual_views,
        "source_view_order": actual_order,
        "vision_view_index": vision_view_index,
    }


def main() -> None:
    from src.datasets.npz_dataset import ObstacleTrajectoryShardDataset
    from src.robot.torch_kinematics import DifferentiableZBotFK
    from src.training.losses import ObstacleFlowMatchingLoss, augment_condition_qik
    from src.utils.config import load_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "obstacle_v1_vision")
    parser.add_argument(
        "--privileged-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_flow" / "best.pt",
    )
    parser.add_argument(
        "--initialize-visual-checkpoint",
        type=Path,
        default=None,
        help="Initialize from an existing Visual CFM checkpoint; its input convolution is expanded as needed.",
    )
    parser.add_argument(
        "--flow-config", type=Path, default=PROJECT_ROOT / "configs" / "obstacle_flow_matching.yaml"
    )
    parser.add_argument("--robot-config", type=Path, default=PROJECT_ROOT / "configs" / "robot.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=2.0e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--vision-view-index",
        type=int,
        default=None,
        help="Select one view from a multi-view dataset before feeding the visual encoder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_flow" / "visual_distilled.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "obstacle_vision_training.json",
    )
    args = parser.parse_args()
    flow_cfg = load_yaml(args.flow_config)
    robot_cfg = load_yaml(args.robot_config)
    seed = int(flow_cfg["seed"])
    set_seed(seed)
    dataset_contract = validate_vision_dataset(args.data_dir, flow_cfg, args.vision_view_index)
    model = build_model(flow_cfg, args.device)
    initialization_path = args.initialize_visual_checkpoint or args.privileged_checkpoint
    checkpoint = torch.load(initialization_path, map_location=args.device, weights_only=False)
    initialization_report = load_initial_state(
        model,
        checkpoint,
        target_view_order=list(flow_cfg.get("vision_encoder", {}).get("view_order", [])),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.vision_encoder.parameters():
        parameter.requires_grad_(True)
    model.obstacle_encoder.eval()
    train_data = ObstacleTrajectoryShardDataset(
        sorted(args.data_dir.glob("train_shard_*.npz")), vision_view_index=args.vision_view_index
    )
    validation_data = ObstacleTrajectoryShardDataset(
        sorted(args.data_dir.glob("validation_shard_*.npz")),
        vision_view_index=args.vision_view_index,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.vision_encoder.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    best_validation = float("inf")
    history = []
    started = perf_counter()
    distilled_output = args.output.with_name(f"{args.output.stem}_distilled{args.output.suffix}")
    for epoch in range(1, args.epochs + 1):
        epoch_values = {}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            model.vision_encoder.train(split == "train")
            total = 0.0
            count = 0
            for batch in loader:
                tokens = batch["obstacle_tokens"].to(args.device)
                mask = batch["obstacle_mask"].to(args.device)
                rgb = batch["rgb"].to(args.device)
                depth = batch["depth"].to(args.device)
                semantic = batch["semantic"].to(args.device)
                if split == "train":
                    rgb = torch.clamp(
                        rgb.float() + torch.randn_like(rgb.float()) * 2.0,
                        0.0,
                        255.0,
                    )
                    depth = depth.float() + torch.randn_like(depth.float()) * 0.003
                    optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher = model.obstacle_encoder(tokens, mask)
                student = model.vision_encoder(rgb, depth, semantic)
                loss = F.mse_loss(student, teacher)
                if split == "train":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.vision_encoder.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.detach()) * tokens.shape[0]
                count += tokens.shape[0]
            epoch_values[split] = total / count
        history.append({"epoch": epoch, **epoch_values})
        if epoch_values["validation"] < best_validation:
            best_validation = epoch_values["validation"]
            distilled_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_validation_loss": best_validation,
                    "config": flow_cfg,
                    "stage": "visual_latent_distillation",
                    "initialized_from": str(initialization_path),
                    **initialization_report,
                },
                distilled_output,
            )
        print(
            f"vision_epoch={epoch:03d} train={epoch_values['train']:.6f} "
            f"validation={epoch_values['validation']:.6f}",
            flush=True,
        )

    distilled = torch.load(distilled_output, map_location=args.device, weights_only=False)
    model.load_state_dict(distilled["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.obstacle_encoder.parameters():
        parameter.requires_grad_(False)
    freeze_duration_head = bool(
        flow_cfg.get("training", {}).get("freeze_duration_head", False)
    )
    if freeze_duration_head:
        for parameter in model.duration_head.parameters():
            parameter.requires_grad_(False)
    limits = robot_cfg["planning_soft_limits"]
    loss_names = (
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
    loss_weights = {
        name: float(flow_cfg["loss_weights"][name])
        for name in loss_names
        if name in flow_cfg["loss_weights"]
    }
    if freeze_duration_head:
        # A frozen head must not back-propagate its auxiliary loss into the
        # shared condition representation.
        loss_weights["duration"] = 0.0
    criterion = ObstacleFlowMatchingLoss(
        fk_model=DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(args.device),
        weights=loss_weights,
        velocity_limits=torch.tensor(limits["velocity"]),
        acceleration_limits=torch.tensor(limits["acceleration"]),
        jerk_limits=torch.tensor(limits["jerk"]),
        periodic_joints=bool(flow_cfg.get("periodic_joints", False)),
    ).to(args.device)
    finetune_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_lr,
        weight_decay=1.0e-4,
    )
    best_finetune = float("inf")
    finetune_history = []
    for epoch in range(1, args.finetune_epochs + 1):
        epoch_values = {}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(split == "train")
            model.obstacle_encoder.eval()
            total = 0.0
            collision_total = 0.0
            count = 0
            for batch in loader:
                condition = batch["condition"].to(args.device)
                tokens = batch["obstacle_tokens"].to(args.device)
                mask = batch["obstacle_mask"].to(args.device)
                trajectory = batch["trajectory"].to(args.device)
                duration = batch["duration"].to(args.device)
                rgb = batch["rgb"].to(args.device)
                depth = batch["depth"].to(args.device)
                semantic = batch["semantic"].to(args.device)
                if split == "train":
                    rgb = torch.clamp(
                        rgb.float() + torch.randn_like(rgb.float()) * 2.0,
                        0.0,
                        255.0,
                    )
                    depth = depth.float() + torch.randn_like(depth.float()) * 0.003
                    if bool(flow_cfg.get("training", {}).get("qik_augmentation", False)):
                        qik_candidates = batch.get("q_ik_candidates", None)
                        if qik_candidates is not None:
                            qik_candidates = qik_candidates.to(args.device)
                        condition = augment_condition_qik(
                            condition,
                            qik_candidates,
                            criterion.fk_model,
                            int(getattr(model, "goal_condition_dof", 0) or 0),
                        )
                    finetune_optimizer.zero_grad(set_to_none=True)
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
                            batch.get("q_experts", None).to(args.device)
                            if batch.get("q_experts", None) is not None else None
                        ),
                    )
                if split == "train":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        1.0,
                    )
                    finetune_optimizer.step()
                batch_count = condition.shape[0]
                total += float(loss.detach()) * batch_count
                collision_total += float(metrics["collision_clearance"].detach()) * batch_count
                count += batch_count
            epoch_values[split] = total / count
            epoch_values[f"{split}_collision"] = collision_total / count
        finetune_history.append({"epoch": epoch, **epoch_values})
        if epoch_values["validation"] < best_finetune:
            best_finetune = epoch_values["validation"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_validation_loss": best_finetune,
                    "config": flow_cfg,
                    "stage": "visual_cfm_finetune",
                    "distilled_checkpoint": str(distilled_output),
                    "initialized_from": str(initialization_path),
                    **initialization_report,
                },
                args.output,
            )
        print(
            f"visual_finetune_epoch={epoch:03d} train={epoch_values['train']:.6f} "
            f"validation={epoch_values['validation']:.6f} "
            f"collision={epoch_values['validation_collision']:.6f}",
            flush=True,
        )
    summary = {
        "status": "success",
        "stage": "visual_cfm_finetune",
        "epochs": args.epochs,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "best_validation_loss": best_validation,
        "best_finetune_validation_loss": best_finetune,
        "elapsed_s": perf_counter() - started,
        "checkpoint": str(args.output),
        "distilled_checkpoint": str(distilled_output),
        "initialized_from": str(initialization_path),
        **initialization_report,
        "vision_backbone": str(flow_cfg.get("vision_encoder", {}).get("backbone", "resnet18_small_input")),
        "camera_views": int(flow_cfg.get("vision_encoder", {}).get("camera_views", 1)),
        "batch_size": args.batch_size,
        "finetune_epochs": args.finetune_epochs,
        "finetune_learning_rate": args.finetune_lr,
        "duration_head_frozen": freeze_duration_head,
        "dataset_contract": dataset_contract,
        "history": history,
        "finetune_history": finetune_history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

