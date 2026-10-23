"""Audit model architecture, checkpoint provenance and training history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_visual_cfm_v621 import build_model  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/visual_cfm_v622_pose.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/checkpoints/visual_cfm_v622_pose.pt",
    )
    parser.add_argument(
        "--training_report",
        type=Path,
        default=ROOT / "outputs/logs/visual_cfm_v622_pose_training.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_v622/model_audit.json",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    parameters = list(model.parameters())
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    history = training_report["history"]
    best_by_stage = {}
    for stage in training_report["stages"]:
        rows = [row for row in history if row["phase"] == stage]
        best = min(rows, key=lambda row: row["validation"])
        best_by_stage[stage] = {
            "epoch": int(best["epoch"]),
            "validation_loss": float(best["validation"]),
            "validation_aux": float(best["validation_aux"]),
        }
    payload = {
        "config": str(args.config.relative_to(ROOT)),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch_within_stage": checkpoint.get("epoch"),
        "checkpoint_best_validation_loss": checkpoint.get("best_validation_loss"),
        "initialized_from": checkpoint.get("initialized_from"),
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "trajectory_input_shape": ["B", 64, 6],
        "condition_shape": ["B", 15],
        "condition_layout": config["condition_layout"],
        "vision_shapes_per_sample": {
            "rgb": [5, 2, 128, 128, 3],
            "depth": [5, 2, 128, 128, 1],
            "semantic": [5, 2, 128, 128, 1],
        },
        "geometry_token_shape_per_sample": [8, 14],
        "velocity_field_output_shape": ["B", 64, 6],
        "duration_head_output_shape": ["B"],
        "model": config["model"],
        "vision_encoder": config["vision_encoder"],
        "obstacle_encoder": config["obstacle_encoder"],
        "loss_weights": config["loss_weights"],
        "aux_weights": config["aux_weights"],
        "inference": config["inference"],
        "training_best_by_stage": best_by_stage,
        "training_elapsed_s": training_report["elapsed_s"],
        "audit_warning": (
            "The globally selected checkpoint is not a post-finetune checkpoint when "
            "checkpoint_stage differs from visual_finetune."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
