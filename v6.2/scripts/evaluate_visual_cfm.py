"""Evaluate 1111 Visual CFM with visual-geometry multi-candidate selection."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from h1_1111_grasp.trajectory_safety import (
    environment_clearance_batch,
    link_segments,
    self_clearance_batch,
)
from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher
from src.planning.differentiable_safety import (
    obstacle_clearance_per_timestep,
    safety_guided_refinement,
)
from src.planning.ik_solver import solve_ik_candidates
from src.planning.flow_postprocess import (
    geometry_preserving_terminal_projection,
    refine_terminal_positions_batch,
    retime_sampled_path,
    smooth_and_time_parameterize,
)
from src.robot.torch_kinematics import DifferentiableZBotFK
from src.utils.config import load_yaml


def build_model(config: dict, device: str) -> ObstacleConditionalFlowMatcher:
    model_cfg = config["model"]
    obstacle_cfg = config["obstacle_encoder"]
    vision_cfg = config["vision_encoder"]
    return ObstacleConditionalFlowMatcher(
        condition_dim=int(config["base_condition_dim"]),
        trajectory_points=int(config["trajectory_shape"][0]),
        dof=int(config["trajectory_shape"][1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        time_embedding_dim=int(model_cfg["time_embedding_dim"]),
        obstacle_hidden_dim=int(obstacle_cfg["hidden_dim"]),
        obstacle_layers=int(obstacle_cfg["transformer_layers"]),
        obstacle_heads=int(obstacle_cfg["transformer_heads"]),
        vision_temporal_frames=int(vision_cfg["temporal_frames"]),
        vision_camera_views=int(vision_cfg["camera_views"]),
        vision_backbone=str(vision_cfg["backbone"]),
        periodic_joints=bool(config.get("periodic_joints", False)),
        goal_condition_dof=int(config.get("goal_condition_dof", 0)),
    ).to(device)


@torch.no_grad()
def sample_from_visual_latent(
    model: ObstacleConditionalFlowMatcher,
    condition: torch.Tensor,
    latent: torch.Tensor,
    candidates: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory_batches = []
    duration_batches = []
    for start in range(0, candidates, 16):
        batch = min(16, candidates - start)
        repeated_condition = condition.repeat(batch, 1)
        repeated_latent = latent.repeat(batch, 1)
        if model.periodic_joints:
            trajectory = 2.0 * torch.rand(
                batch,
                model.trajectory_points,
                model.dof,
                device=condition.device,
                generator=generator,
            ) - 1.0
        else:
            trajectory = torch.randn(
                batch,
                model.trajectory_points,
                model.dof,
                device=condition.device,
                generator=generator,
            )
        trajectory[:, 0] = repeated_condition[:, : model.dof] / model.joint_scale
        time_step = 1.0 / steps
        duration = torch.full((batch,), model.min_duration, device=condition.device)
        for step in range(steps):
            flow_time = torch.full((batch,), step / steps, device=condition.device)
            velocity, duration = model.forward_with_obstacle_latent(
                trajectory, flow_time, repeated_condition, repeated_latent
            )
            prediction = trajectory + time_step * velocity
            if model.periodic_joints:
                prediction = torch.remainder(prediction + 1.0, 2.0) - 1.0
            next_time = torch.full((batch,), (step + 1) / steps, device=condition.device)
            next_velocity, duration = model.forward_with_obstacle_latent(
                prediction, next_time, repeated_condition, repeated_latent
            )
            trajectory = trajectory + 0.5 * time_step * (velocity + next_velocity)
            if model.periodic_joints:
                trajectory = torch.remainder(trajectory + 1.0, 2.0) - 1.0
            trajectory[:, 0] = repeated_condition[:, : model.dof] / model.joint_scale
        if int(getattr(model, "goal_condition_dof", 0) or 0) > 0:
            # v6.1 terminal anchor (P2): snap the last waypoint to q_ik.
            q_ik = torch.remainder(
                repeated_condition[:, model.dof : model.dof + int(getattr(model, "goal_condition_dof", 0))]
                / model.joint_scale
                + 1.0,
                2.0,
            ) - 1.0
            trajectory[:, -1] = q_ik
        trajectory_batches.append(trajectory * model.joint_scale)
        duration_batches.append(duration)
    return torch.cat(trajectory_batches), torch.cat(duration_batches)


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z / denominator * sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records: list[dict], key: str) -> dict:
    successes = [record for record in records if record[key]["success"]]
    result = {
        "tasks": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / max(len(records), 1),
        "success_rate_wilson_95": wilson(len(successes), len(records)),
    }
    for side in ("left", "right"):
        side_rows = [record for record in records if record["side"] == side]
        side_success = sum(record[key]["success"] for record in side_rows)
        result[f"{side}_success_rate"] = side_success / max(len(side_rows), 1)
    if successes:
        for metric in (
            "raw_terminal_error_m",
            "terminal_error_m",
            "environment_clearance_m",
            "self_clearance_m",
            "velocity_rms_rad_s",
            "acceleration_rms_rad_s2",
            "jerk_rms_rad_s3",
        ):
            values = np.asarray([record[key][metric] for record in successes])
            result[f"mean_{metric}"] = float(values.mean())
            result[f"p95_{metric}"] = float(np.quantile(values, 0.95))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--view_indices",
        default="",
        help="Comma-separated source-view indices supplied to the visual encoder.",
    )
    parser.add_argument("--zero_views", default="")
    parser.add_argument("--rgb_noise_std", type=float, default=0.0)
    parser.add_argument("--depth_noise_std", type=float, default=0.0)
    parser.add_argument("--position_tolerance_m", type=float, default=0.03)
    parser.add_argument("--required_clearance_m", type=float, default=0.01)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--safety_refine_steps", type=int, default=0)
    parser.add_argument("--path_residual_weight", type=float, default=0.15)
    parser.add_argument(
        "--postprocess_mode",
        choices=("reference_spline", "geometry_preserving"),
        default="reference_spline",
    )
    parser.add_argument("--geometry_control_points", type=int, default=16)
    parser.add_argument("--timing_points", type=int, default=64)
    parser.add_argument("--unclipped_duration", action="store_true")
    parser.add_argument("--nominal_max_duration_s", type=float, default=5.0)
    parser.add_argument("--max_output_duration_s", type=float, default=30.0)
    parser.add_argument("--save_selected_npz", type=Path, default=None)
    parser.add_argument("--save_candidate_bank_npz", type=Path, default=None)
    parser.add_argument("--candidate_bank_size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_yaml(args.config)
    model = build_model(config, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    fk_model = DifferentiableZBotFK(WORKSPACE_ROOT / "configs" / "kinematic_chain.json").to(
        args.device
    )
    robot_config = load_yaml(WORKSPACE_ROOT / "configs" / "robot.yaml")
    limits = robot_config["planning_soft_limits"]
    view_indices = [int(item) for item in args.view_indices.split(",") if item.strip()]
    zero_views = {int(item) for item in args.zero_views.split(",") if item.strip()}
    paths = sorted(args.data_dir.glob("test_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no test shards under {args.data_dir}")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            # Accessing data[key] inside the row loop makes NumPy decompress the
            # complete NPZ member once per sample.  Materialize each member once
            # and let row dictionaries retain inexpensive array views.
            shard = {
                key: np.ascontiguousarray(data[key])
                for key in data.files
                if key != "camera_view_names"
            }
        rows.extend(
            {key: value[row] for key, value in shard.items()}
            for row in range(len(shard["q"]))
        )
    if args.max_tasks:
        rows = rows[: args.max_tasks]
    source_camera_views = int(rows[0]["rgb"].shape[0])
    expected_camera_views = int(config["vision_encoder"]["camera_views"])
    selected_camera_views = len(view_indices) if view_indices else source_camera_views
    if selected_camera_views != expected_camera_views:
        raise ValueError(
            f"model expects {expected_camera_views} views, received {selected_camera_views}; "
            "set --view_indices when evaluating a subset"
        )

    records = []
    selected_trajectories = []
    candidate_banks = []
    started = perf_counter()
    for task_index, row in enumerate(rows):
        condition = torch.as_tensor(row["condition"], dtype=torch.float32, device=args.device)[None]
        rgb = torch.as_tensor(row["rgb"], device=args.device)[None].clone()
        depth = torch.as_tensor(row["depth"], device=args.device)[None].clone()
        semantic = torch.as_tensor(row["semantic"], device=args.device)[None].clone()
        if view_indices:
            rgb = rgb[:, view_indices]
            depth = depth[:, view_indices]
            semantic = semantic[:, view_indices]
        for view in zero_views:
            rgb[:, view].zero_()
            depth[:, view].zero_()
            semantic[:, view].zero_()
        generator = torch.Generator(device=args.device).manual_seed(args.seed + task_index)
        if args.rgb_noise_std:
            rgb = (rgb.float() + torch.randn(rgb.shape, device=args.device, generator=generator) * args.rgb_noise_std).clamp(0, 255)
        if args.depth_noise_std:
            depth = (depth.float() + torch.randn(depth.shape, device=args.device, generator=generator) * args.depth_noise_std).clamp_min(0)
        with torch.no_grad():
            latent = model.vision_encoder(rgb, depth, semantic)
        ik_solutions = None
        if getattr(model, "goal_condition_dof", 0) > 0 and bool(
            config.get("inference", {}).get("ik_first", False)
        ):
            goal_position_np = condition[:, -3:].cpu().numpy()[0]
            ik_solutions = solve_ik_candidates(
                fk_model,
                goal_position_np,
                n_candidates=int(config.get("inference", {}).get("ik_candidates", 4)),
                restarts=8,
                iterations=100,
                position_tolerance_m=0.005,
                seed=int(args.seed) + task_index,
                device=args.device,
            )
            if len(ik_solutions) == 0:
                ik_solutions = condition[:, 6:12].cpu().numpy().reshape(1, 6)
        with torch.no_grad():
            if ik_solutions is not None:
                candidate_batches = []
                duration_batches = []
                for q_ik in ik_solutions:
                    ik_condition = torch.cat(
                        (
                            condition[:, :6],
                            torch.as_tensor(q_ik, dtype=torch.float32, device=args.device)[None],
                            condition[:, -3:],
                        ),
                        dim=-1,
                    )
                    batch_paths, batch_durations = sample_from_visual_latent(
                        model, ik_condition, latent, args.candidates, args.ode_steps, generator
                    )
                    candidate_batches.append(batch_paths)
                    duration_batches.append(batch_durations)
                paths_tensor = torch.cat(candidate_batches, dim=0)
                durations = torch.cat(duration_batches, dim=0)
            else:
                paths_tensor, durations = sample_from_visual_latent(
                    model, condition, latent, args.candidates, args.ode_steps, generator
                )
            candidate_count = int(paths_tensor.shape[0])
            raw_terminal_positions, _ = fk_model(paths_tensor[:, -1])
            raw_terminal_errors = torch.linalg.norm(
                raw_terminal_positions - condition[:, -3:], dim=1
            ).cpu().numpy()
        paths_np = paths_tensor.cpu().numpy()
        durations_np = durations.cpu().numpy()
        del rgb, depth, semantic, latent
        torch.cuda.empty_cache()
        tokens = torch.as_tensor(
            row["obstacle_tokens"], dtype=torch.float32, device=args.device
        )[None].repeat(candidate_count, 1, 1)
        masks = torch.as_tensor(
            row["obstacle_mask"], dtype=torch.bool, device=args.device
        )[None].repeat(candidate_count, 1)
        if args.safety_refine_steps:
            refined_batches = []
            for start in range(0, candidate_count, 16):
                end = min(start + 16, candidate_count)
                refined, _ = safety_guided_refinement(
                    fk_model,
                    paths_tensor[start:end],
                    durations[start:end],
                    tokens[start:end],
                    masks[start:end],
                    q_start=condition[:, :6].repeat(end - start, 1),
                    steps=args.safety_refine_steps,
                    periodic_joints=True,
                )
                refined_batches.append(refined)
            paths_tensor = torch.cat(refined_batches, dim=0)
            paths_np = paths_tensor.cpu().numpy()
        evaluation_paths_np = paths_np
        duration_clipped = np.zeros(candidate_count, dtype=np.bool_)
        duration_valid = np.ones(candidate_count, dtype=np.bool_)
        duration_exceeds_nominal = np.zeros(candidate_count, dtype=np.bool_)
        if args.postprocess:
            ik_results = refine_terminal_positions_batch(
                fk_model=fk_model,
                q_initial=paths_np[:, -1],
                goal_position=row["condition"][-3:],
                max_iterations=40,
                regularization_weight=1.0e-5,
                device=args.device,
            )
            processed_paths = []
            dense_paths = []
            processed_durations = []
            for index, ik_result in enumerate(ik_results):
                raw_unwrapped = np.unwrap(paths_np[index], axis=0, period=2.0 * np.pi)
                raw_unwrapped += (row["q_start"] - raw_unwrapped[0])[None]
                terminal = raw_unwrapped[-1] + (
                    (ik_result.q - raw_unwrapped[-1] + np.pi) % (2.0 * np.pi) - np.pi
                )
                if args.postprocess_mode == "geometry_preserving":
                    trajectory, timing = geometry_preserving_terminal_projection(
                        raw_q=raw_unwrapped,
                        q_start=row["q_start"],
                        q_terminal=terminal,
                        predicted_duration=float(durations_np[index]),
                        min_duration=0.8,
                        max_duration=(
                            args.max_output_duration_s
                            if args.unclipped_duration
                            else args.nominal_max_duration_s
                        ),
                        velocity_limits=np.asarray(limits["velocity"]),
                        acceleration_limits=np.asarray(limits["acceleration"]),
                        jerk_limits=np.asarray(limits["jerk"]),
                        control_point_count=args.geometry_control_points,
                    )
                else:
                    trajectory, timing = smooth_and_time_parameterize(
                        raw_q=raw_unwrapped,
                        q_start=row["q_start"],
                        q_terminal=terminal,
                        predicted_duration=float(durations_np[index]),
                        points=64,
                        control_point_count=12,
                        residual_weight=args.path_residual_weight,
                        min_duration=0.8,
                        max_duration=(
                            args.max_output_duration_s
                            if args.unclipped_duration
                            else args.nominal_max_duration_s
                        ),
                        velocity_limits=np.asarray(limits["velocity"]),
                        acceleration_limits=np.asarray(limits["acceleration"]),
                        jerk_limits=np.asarray(limits["jerk"]),
                    )
                processed_paths.append(trajectory.q)
                if args.unclipped_duration or args.timing_points != len(trajectory.q):
                    dense_trajectory, timing = retime_sampled_path(
                        q=trajectory.q,
                        predicted_duration=float(durations_np[index]),
                        min_duration=0.8,
                        nominal_max_duration=args.nominal_max_duration_s,
                        max_output_duration=args.max_output_duration_s,
                        velocity_limits=np.asarray(limits["velocity"]),
                        acceleration_limits=np.asarray(limits["acceleration"]),
                        jerk_limits=np.asarray(limits["jerk"]),
                        timing_points=args.timing_points,
                    )
                    dense_paths.append(dense_trajectory.q)
                    processed_durations.append(dense_trajectory.duration)
                    duration_valid[index] = bool(timing["duration_valid"])
                    duration_exceeds_nominal[index] = bool(
                        timing["duration_exceeds_nominal_max"]
                    )
                else:
                    dense_paths.append(trajectory.q)
                    processed_durations.append(trajectory.duration)
                    duration_clipped[index] = bool(timing["duration_clipped_to_max"])
                    duration_valid[index] = not duration_clipped[index]
            paths_np = np.asarray(processed_paths, dtype=np.float32)
            evaluation_paths_np = np.asarray(dense_paths, dtype=np.float32)
            durations_np = np.asarray(processed_durations, dtype=np.float32)
            paths_tensor = torch.as_tensor(paths_np, dtype=torch.float32, device=args.device)
            durations = torch.as_tensor(durations_np, dtype=torch.float32, device=args.device)
        evaluation_points = evaluation_paths_np.shape[1]
        collision_screen_points = paths_np.shape[1]
        segments = link_segments(
            fk_model, paths_np.reshape(-1, 6), device=args.device
        ).reshape(candidate_count, collision_screen_points, -1, 2, 3)
        with torch.no_grad():
            terminal_positions, _ = fk_model(paths_tensor[:, -1])
            terminal_errors = torch.linalg.norm(
                terminal_positions - condition[:, -3:], dim=1
            ).cpu().numpy()
            if "scene_points" not in row:
                environment = obstacle_clearance_per_timestep(
                    fk_model,
                    paths_tensor,
                    durations,
                    tokens,
                    masks,
                    link_radius_m=0.05,
                ).amin(dim=1).cpu().numpy()
        if "scene_points" in row:
            cloud = np.asarray(row["scene_points"])[
                np.asarray(row["scene_point_mask"], dtype=np.bool_)
            ]
            environment = environment_clearance_batch(
                segments[:, :, 1:], cloud, capsule_radius_m=0.05
            )
        self_distances = self_clearance_batch(
            segments.reshape(-1, segments.shape[2], 2, 3), 0.045
        ).reshape(candidate_count, collision_screen_points).min(axis=1)
        metrics = []
        for index in range(candidate_count):
            unwrapped = np.unwrap(
                evaluation_paths_np[index], axis=0, period=2.0 * np.pi
            )
            time = np.linspace(
                0.0, float(durations_np[index]), evaluation_points
            )
            velocity = np.gradient(unwrapped, time, axis=0, edge_order=2)
            acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
            jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
            dynamics_ok = (
                np.max(np.abs(velocity), axis=0)
                <= np.asarray(limits["velocity"]) * (1.0 + 1.0e-6)
            ).all() and (
                np.max(np.abs(acceleration), axis=0)
                <= np.asarray(limits["acceleration"]) * (1.0 + 1.0e-6)
            ).all() and (
                np.max(np.abs(jerk), axis=0)
                <= np.asarray(limits["jerk"]) * (1.0 + 1.0e-6)
            ).all()
            success = bool(
                terminal_errors[index] <= args.position_tolerance_m
                and environment[index] >= args.required_clearance_m
                and self_distances[index] >= args.required_clearance_m
                and dynamics_ok
                and duration_valid[index]
            )
            metrics.append(
                {
                    "index": index,
                    "success": success,
                    "raw_terminal_error_m": float(raw_terminal_errors[index]),
                    "terminal_error_m": float(terminal_errors[index]),
                    "environment_clearance_m": float(environment[index]),
                    "self_clearance_m": float(self_distances[index]),
                    "velocity_rms_rad_s": float(np.sqrt(np.mean(velocity**2))),
                    "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))),
                    "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk**2))),
                    "duration_s": float(durations_np[index]),
                    "duration_exceeds_nominal_max": bool(
                        duration_exceeds_nominal[index]
                    ),
                    "terminal_ok": bool(terminal_errors[index] <= args.position_tolerance_m),
                    "environment_ok": bool(environment[index] >= args.required_clearance_m),
                    "self_clearance_ok": bool(
                        self_distances[index] >= args.required_clearance_m
                    ),
                    "dynamics_ok": bool(dynamics_ok),
                    "duration_ok": bool(duration_valid[index]),
                }
            )
        valid = [item for item in metrics if item["success"]]
        selected = min(
            valid,
            key=lambda item: item["terminal_error_m"]
            + 0.002 * item["acceleration_rms_rad_s2"]
            + 0.0002 * item["jerk_rms_rad_s3"],
            default=min(metrics, key=lambda item: item["terminal_error_m"]),
        )
        ranked = sorted(
            metrics,
            key=lambda item: (
                not item["success"],
                item["terminal_error_m"]
                + 40.0 * max(
                    0.0,
                    args.required_clearance_m
                    - min(item["environment_clearance_m"], item["self_clearance_m"]),
                )
                + 0.002 * item["acceleration_rms_rad_s2"]
                + 0.0002 * item["jerk_rms_rad_s3"],
            ),
        )
        bank_indices = [item["index"] for item in ranked[: args.candidate_bank_size]]
        records.append(
            {
                "task_index": task_index,
                "sample_id": str(row["sample_id"]),
                "side": str(row["side"]),
                "single": metrics[0],
                "multi": selected,
                "any_clearance_safe_candidate": any(
                    item["environment_clearance_m"] >= args.required_clearance_m
                    and item["self_clearance_m"] >= args.required_clearance_m
                    for item in metrics
                ),
                "candidate_constraint_pass_rates": {
                    key: float(np.mean([item[key] for item in metrics]))
                    for key in (
                        "terminal_ok",
                        "environment_ok",
                        "self_clearance_ok",
                        "dynamics_ok",
                        "duration_ok",
                        "success",
                    )
                },
                "task_has_passing_candidate": {
                    key: any(item[key] for item in metrics)
                    for key in (
                        "terminal_ok",
                        "environment_ok",
                        "self_clearance_ok",
                        "dynamics_ok",
                        "duration_ok",
                        "success",
                    )
                },
            }
        )
        selected_trajectories.append(
            {
                "sample_id": np.asarray(str(row["sample_id"])),
                "side": np.asarray(str(row["side"])),
                "q_start": np.asarray(row["q_start"], dtype=np.float32),
                "goal_position": np.asarray(row["condition"][-3:], dtype=np.float32),
                "q": np.asarray(paths_np[selected["index"]], dtype=np.float32),
                "q_dense": np.asarray(
                    evaluation_paths_np[selected["index"]], dtype=np.float32
                ),
                "duration": np.float32(durations_np[selected["index"]]),
                "geometric_success": np.bool_(selected["success"]),
            }
        )
        candidate_banks.append(
            {
                "sample_id": np.asarray(str(row["sample_id"])),
                "side": np.asarray(str(row["side"])),
                "q_start": np.asarray(row["q_start"], dtype=np.float32),
                "goal_position": np.asarray(row["condition"][-3:], dtype=np.float32),
                "q": np.asarray(paths_np[bank_indices], dtype=np.float32),
                "q_dense": np.asarray(
                    evaluation_paths_np[bank_indices], dtype=np.float32
                ),
                "duration": np.asarray(durations_np[bank_indices], dtype=np.float32),
                "geometric_success": np.asarray(
                    [metrics[index]["success"] for index in bank_indices], dtype=np.bool_
                ),
                "environment_clearance_m": np.asarray(
                    [metrics[index]["environment_clearance_m"] for index in bank_indices],
                    dtype=np.float32,
                ),
                "self_clearance_m": np.asarray(
                    [metrics[index]["self_clearance_m"] for index in bank_indices],
                    dtype=np.float32,
                ),
            }
        )
        print(
            f"[EVAL] {task_index + 1}/{len(rows)} single={metrics[0]['success']} "
            f"multi={selected['success']} error={selected['terminal_error_m']:.4f}",
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "tasks": len(records),
        "candidates": args.candidates,
        "ode_steps": args.ode_steps,
        "input_view_indices": view_indices or list(range(source_camera_views)),
        "zero_views": sorted(zero_views),
        "rgb_noise_std": args.rgb_noise_std,
        "depth_noise_std": args.depth_noise_std,
        "postprocess": args.postprocess,
        "safety_refine_steps": args.safety_refine_steps,
        "path_residual_weight": args.path_residual_weight,
        "postprocess_mode": args.postprocess_mode,
        "geometry_control_points": args.geometry_control_points,
        "timing_points": args.timing_points,
        "candidate_collision_screen_points": collision_screen_points,
        "unclipped_duration": args.unclipped_duration,
        "nominal_max_duration_s": args.nominal_max_duration_s,
        "max_output_duration_s": args.max_output_duration_s,
        "selection_geometry": (
            f"{source_camera_views}-camera RGB-D reconstructed "
            "point cloud; no USD obstacle truth"
            if rows and "scene_points" in rows[0]
            else f"{source_camera_views}-camera RGB-D reconstruction "
            "tokens; no USD obstacle truth"
        ),
        "single_candidate": summarize(records, "single"),
        "multi_candidate": summarize(records, "multi"),
        "any_clearance_safe_candidate_rate": float(
            np.mean([record["any_clearance_safe_candidate"] for record in records])
        ),
        "candidate_constraint_pass_rates": {
            key: float(
                np.mean(
                    [record["candidate_constraint_pass_rates"][key] for record in records]
                )
            )
            for key in (
                "terminal_ok",
                "environment_ok",
                "self_clearance_ok",
                "dynamics_ok",
                "duration_ok",
                "success",
            )
        },
        "task_has_passing_candidate_rate": {
            key: float(
                np.mean([record["task_has_passing_candidate"][key] for record in records])
            )
            for key in (
                "terminal_ok",
                "environment_ok",
                "self_clearance_ok",
                "dynamics_ok",
                "duration_ok",
                "success",
            )
        },
        "elapsed_s": perf_counter() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_selected_npz:
        args.save_selected_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_selected_npz,
            **{
                key: np.stack([item[key] for item in selected_trajectories])
                for key in selected_trajectories[0]
            },
        )
    if args.save_candidate_bank_npz:
        args.save_candidate_bank_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_candidate_bank_npz,
            **{
                key: np.stack([item[key] for item in candidate_banks])
                for key in candidate_banks[0]
            },
        )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
