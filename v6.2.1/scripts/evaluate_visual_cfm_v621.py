"""Evaluate 1111 v6.2.1 Visual CFM: two-stage grasp -> hold -> place.

Each test task is split into two sequential stages:

  * grasp: condition_grasp = [q_start(6), q_hold(6), p_goal(3)] -> q_grasp (64,6)
  * place : condition_place  = [q_hold(6), q_place_ik(6), p_place(3)] -> q_place (64,6)

Both stages use the shared 5-camera vision and the point-cloud component
tokens plus the per-stage p_goal as FiLM conditioning.  The terminal anchor of
each stage is the exact FK anchor q_ik (FK error 0 by construction); postprocess
snaps the sampled terminal to the anchor and re-times the path.

Reported metrics per stage: raw terminal error (no correction), corrected
terminal error, environment/self clearance, dynamics, duration and success.
Two-stage success requires both stages to pass.
"""

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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from h1_1111_grasp.trajectory_safety import (
    environment_clearance_batch,
    link_segments,
    self_clearance_batch,
)
from src.models.obstacle_flow_matching import ObstacleConditionalFlowMatcher
from src.models.conditional_trajectory_baselines import build_condition_only_model
from src.planning.ik_solver import quaternion_distance_rad
from src.planning.differentiable_safety import safety_guided_refinement
from src.planning.flow_postprocess import (
    geometry_preserving_terminal_projection,
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
        vision_condition_dim=int(vision_cfg.get("condition_cond_dim", 0)),
        use_target_heads=bool(vision_cfg.get("use_target_heads", False)),
        target_type_dim=int(vision_cfg.get("target_type_dim", 5)),
        fuse_point_cloud=bool(vision_cfg.get("fuse_point_cloud", False)),
        point_cloud_classifier=bool(vision_cfg.get("point_cloud_classifier", False)),
    ).to(device)


@torch.no_grad()
def sample_stage_candidates(
    model: torch.nn.Module,
    condition: torch.Tensor,
    vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    p_goal: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    candidates: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Encode the stage once, then sample ``candidates`` noise draws."""
    if hasattr(model, "sample_candidates"):
        torch.cuda.synchronize() if condition.is_cuda else None
        sample_started = perf_counter()
        paths, durations = model.sample_candidates(
            condition, candidates, steps, generator
        )
        torch.cuda.synchronize() if condition.is_cuda else None
        return paths, durations, {
            "_timing_ms": {
                "vision_encoding_ms": float("nan"),
                "cfm_sampling_ms": 1000.0 * (perf_counter() - sample_started),
            }
        }
    torch.cuda.synchronize() if condition.is_cuda else None
    encoding_started = perf_counter()
    latent, aux = model.encode_v621(vision, p_goal, tokens, mask)
    torch.cuda.synchronize() if condition.is_cuda else None
    vision_encoding_ms = 1000.0 * (perf_counter() - encoding_started)
    sampling_started = perf_counter()
    trajectory_batches = []
    duration_batches = []
    batch = min(16, candidates)
    for start in range(0, candidates, batch):
        count = min(batch, candidates - start)
        repeated_condition = condition.repeat(count, 1)
        repeated_latent = latent.repeat(count, 1)
        if model.periodic_joints:
            trajectory = 2.0 * torch.rand(
                count,
                model.trajectory_points,
                model.dof,
                device=condition.device,
                generator=generator,
            ) - 1.0
        else:
            trajectory = torch.randn(
                count,
                model.trajectory_points,
                model.dof,
                device=condition.device,
                generator=generator,
            )
        start_config = repeated_condition[:, : model.dof] / model.joint_scale
        if model.periodic_joints:
            start_config = torch.remainder(start_config + 1.0, 2.0) - 1.0
        trajectory[:, 0] = start_config
        dt = 1.0 / steps
        duration = torch.full((count,), model.min_duration, device=condition.device)
        for step in range(steps):
            flow_time = torch.full((count,), step / steps, device=condition.device)
            velocity, duration = model.forward_with_obstacle_latent(
                trajectory, flow_time, repeated_condition, repeated_latent,
                already_projected=True,
            )
            prediction = trajectory + dt * velocity
            if model.periodic_joints:
                prediction = torch.remainder(prediction + 1.0, 2.0) - 1.0
            next_time = torch.full((count,), (step + 1) / steps, device=condition.device)
            next_velocity, duration = model.forward_with_obstacle_latent(
                prediction, next_time, repeated_condition, repeated_latent,
                already_projected=True,
            )
            trajectory = trajectory + 0.5 * dt * (velocity + next_velocity)
            if model.periodic_joints:
                trajectory = torch.remainder(trajectory + 1.0, 2.0) - 1.0
            trajectory[:, 0] = start_config
        trajectory_batches.append(trajectory * model.joint_scale)
        duration_batches.append(duration)
    torch.cuda.synchronize() if condition.is_cuda else None
    aux = dict(aux)
    aux["_timing_ms"] = {
        "vision_encoding_ms": vision_encoding_ms,
        "cfm_sampling_ms": 1000.0 * (perf_counter() - sampling_started),
    }
    return torch.cat(trajectory_batches), torch.cat(duration_batches), aux


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


def summarize_stage(records: list[dict], key: str) -> dict:
    successes = [record for record in records if record[key]["success"]]
    result = {
        "tasks": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / max(len(records), 1),
        "success_rate_wilson_95": wilson(len(successes), len(records)),
    }
    if successes:
        for metric in (
            "raw_terminal_error_m",
            "raw_orientation_error_rad",
            "terminal_error_m",
            "orientation_error_rad",
            "environment_clearance_m",
            "self_clearance_m",
            "velocity_rms_rad_s",
            "acceleration_rms_rad_s2",
            "jerk_rms_rad_s3",
        ):
            values = np.asarray([record[key][metric] for record in successes])
            result[f"mean_{metric}"] = float(values.mean())
            result[f"p95_{metric}"] = float(np.quantile(values, 0.95))
    all_raw = np.asarray([record[key]["raw_terminal_error_m"] for record in records])
    result["mean_raw_terminal_error_all"] = float(all_raw.mean())
    result["p95_raw_terminal_error_all"] = float(np.quantile(all_raw, 0.95))
    result["median_raw_terminal_error_all"] = float(np.median(all_raw))
    all_raw_orientation = np.asarray(
        [record[key]["raw_orientation_error_rad"] for record in records]
    )
    result["mean_raw_orientation_error_all_rad"] = float(
        all_raw_orientation.mean()
    )
    result["p95_raw_orientation_error_all_rad"] = float(
        np.quantile(all_raw_orientation, 0.95)
    )
    return result


def evaluate_stage(
    model: torch.nn.Module,
    fk_model: DifferentiableZBotFK,
    row: dict,
    stage: str,
    vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    limits: dict,
    args: argparse.Namespace,
    device: str,
    generator: torch.Generator,
) -> dict:
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    stage_started = perf_counter()
    condition = torch.as_tensor(row[f"condition_{stage}"], dtype=torch.float32, device=device)[None]
    p_goal = condition[:, -3:]
    q_start = np.asarray(row["q_start"] if stage == "grasp" else row["q_hold"], dtype=np.float32)
    anchor_q = np.asarray(row[f"q_ik_candidates_{stage}"][0], dtype=np.float32)
    goal_position = row[f"condition_{stage}"][-3:]
    goal_quaternion = np.asarray(
        row.get(f"goal_quaternion_{stage}", np.asarray([1.0, 0.0, 0.0, 0.0])),
        dtype=np.float32,
    )
    with torch.no_grad():
        paths_tensor, durations, aux = sample_stage_candidates(
            model,
            condition,
            vision,
            p_goal,
            tokens,
            masks,
            args.candidates,
            args.ode_steps,
            generator,
        )
    sampling_timing = aux.pop("_timing_ms", {})
    with torch.no_grad():
        raw_terminal_positions, raw_terminal_quaternions = fk_model(paths_tensor[:, -1])
        raw_terminal_errors = torch.linalg.norm(
            raw_terminal_positions - condition[:, -3:], dim=1
        ).cpu().numpy()
        raw_orientation_errors = np.asarray(
            quaternion_distance_rad(
                raw_terminal_quaternions.cpu().numpy(), goal_quaternion[None]
            )
        )
    paths_np = paths_tensor.cpu().numpy()
    durations_np = durations.cpu().numpy()
    candidate_count = int(paths_tensor.shape[0])

    if args.safety_refine_steps:
        refined_batches = []
        repeated_tokens = tokens.repeat(candidate_count, 1, 1)
        repeated_masks = masks.repeat(candidate_count, 1)
        for start in range(0, candidate_count, 16):
            end = min(start + 16, candidate_count)
            refined, _ = safety_guided_refinement(
                fk_model,
                paths_tensor[start:end],
                durations[start:end],
                repeated_tokens[start:end],
                repeated_masks[start:end],
                q_start=condition[:, :6].repeat(end - start, 1),
                steps=args.safety_refine_steps,
                periodic_joints=True,
            )
            refined_batches.append(refined)
        paths_tensor = torch.cat(refined_batches, dim=0)
        paths_np = paths_tensor.cpu().numpy()

    evaluation_paths_np = paths_np
    raw_paths_for_debug = np.asarray(paths_np, dtype=np.float32).copy()
    reference_paths_for_debug: list[np.ndarray] = []
    residual_controls_for_debug: list[np.ndarray] = []
    reconstructed_controls_for_debug: list[np.ndarray] = []
    duration_valid = np.ones(candidate_count, dtype=np.bool_)
    duration_exceeds_nominal = np.zeros(candidate_count, dtype=np.bool_)
    reconstruction_ms = 0.0
    densification_ms = 0.0
    if args.postprocess:
        processed_paths = []
        dense_paths = []
        processed_durations = []
        for index in range(candidate_count):
            raw_unwrapped = np.unwrap(paths_np[index], axis=0, period=2.0 * np.pi)
            raw_unwrapped += (q_start - raw_unwrapped[0])[None]
            terminal = raw_unwrapped[-1] + (
                (anchor_q - raw_unwrapped[-1] + np.pi) % (2.0 * np.pi) - np.pi
            )
            control_indices = np.linspace(0, len(raw_unwrapped) - 1, 12).round().astype(int)
            knot_u = np.linspace(0.0, 1.0, 12)
            raw_controls = raw_unwrapped[control_indices]
            raw_chord = raw_unwrapped[0][None] + knot_u[:, None] * (
                raw_unwrapped[-1] - raw_unwrapped[0]
            )[None]
            raw_residual_controls = raw_controls - raw_chord
            minimum_jerk_knot = 10.0 * knot_u**3 - 15.0 * knot_u**4 + 6.0 * knot_u**5
            reference_controls = q_start[None] + minimum_jerk_knot[:, None] * (
                terminal - q_start
            )[None]
            reconstructed_controls = (
                reference_controls + args.path_residual_weight * raw_residual_controls
            )
            reconstructed_controls[0] = q_start
            reconstructed_controls[-1] = terminal
            debug_u = np.linspace(0.0, 1.0, args.timing_points)
            debug_phase = 10.0 * debug_u**3 - 15.0 * debug_u**4 + 6.0 * debug_u**5
            reference_paths_for_debug.append(
                q_start[None] + debug_phase[:, None] * (terminal - q_start)[None]
            )
            residual_controls_for_debug.append(raw_residual_controls)
            reconstructed_controls_for_debug.append(reconstructed_controls)
            postprocess_started = perf_counter()
            if args.postprocess_mode == "geometry_preserving":
                trajectory, timing = geometry_preserving_terminal_projection(
                    raw_q=raw_unwrapped,
                    q_start=q_start,
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
                    q_start=q_start,
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
            reconstruction_ms += 1000.0 * (perf_counter() - postprocess_started)
            if args.unclipped_duration:
                densification_started = perf_counter()
                trajectory, dense_timing = retime_sampled_path(
                    q=trajectory.q,
                    predicted_duration=float(trajectory.duration),
                    min_duration=0.8,
                    nominal_max_duration=args.nominal_max_duration_s,
                    max_output_duration=args.max_output_duration_s,
                    velocity_limits=np.asarray(limits["velocity"]),
                    acceleration_limits=np.asarray(limits["acceleration"]),
                    jerk_limits=np.asarray(limits["jerk"]),
                    timing_points=args.timing_points,
                )
                densification_ms += 1000.0 * (
                    perf_counter() - densification_started
                )
                timing = {**timing, **dense_timing}
            processed_paths.append(trajectory.q)
            dense_paths.append(trajectory.q)
            processed_durations.append(trajectory.duration)
            duration_valid[index] = bool(
                timing.get(
                    "duration_valid", not bool(timing["duration_clipped_to_max"])
                )
            )
            duration_exceeds_nominal[index] = bool(
                timing.get("duration_exceeds_nominal_max", False)
            )
        paths_np = np.asarray(processed_paths, dtype=np.float32)
        evaluation_paths_np = np.asarray(dense_paths, dtype=np.float32)
        durations_np = np.asarray(processed_durations, dtype=np.float32)
        paths_tensor = torch.as_tensor(paths_np, dtype=torch.float32, device=device)

    # Candidate dumps expose aligned diagnostic arrays even when trajectory
    # post-processing is disabled, so debug export never depends on CLI flag order.
    if not reference_paths_for_debug:
        knot_u = np.linspace(0.0, 1.0, 12)
        minimum_jerk_knot = 10.0 * knot_u**3 - 15.0 * knot_u**4 + 6.0 * knot_u**5
        debug_u = np.linspace(0.0, 1.0, args.timing_points)
        debug_phase = 10.0 * debug_u**3 - 15.0 * debug_u**4 + 6.0 * debug_u**5
        for index in range(candidate_count):
            raw_unwrapped = np.unwrap(paths_np[index], axis=0, period=2.0 * np.pi)
            raw_unwrapped += (q_start - raw_unwrapped[0])[None]
            terminal = raw_unwrapped[-1] + (
                (anchor_q - raw_unwrapped[-1] + np.pi) % (2.0 * np.pi) - np.pi
            )
            control_indices = np.linspace(0, len(raw_unwrapped) - 1, 12).round().astype(int)
            raw_controls = raw_unwrapped[control_indices]
            raw_chord = raw_unwrapped[0][None] + knot_u[:, None] * (
                raw_unwrapped[-1] - raw_unwrapped[0]
            )[None]
            residual = raw_controls - raw_chord
            reference_controls = q_start[None] + minimum_jerk_knot[:, None] * (
                terminal - q_start
            )[None]
            reference_paths_for_debug.append(
                q_start[None] + debug_phase[:, None] * (terminal - q_start)[None]
            )
            residual_controls_for_debug.append(residual)
            reconstructed_controls_for_debug.append(reference_controls + residual)
        durations = torch.as_tensor(durations_np, dtype=torch.float32, device=device)

    evaluation_points = evaluation_paths_np.shape[1]
    collision_screen_points = paths_np.shape[1]
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    collision_started = perf_counter()
    segments = link_segments(
        fk_model, paths_np.reshape(-1, 6), device=device
    ).reshape(candidate_count, collision_screen_points, -1, 2, 3)
    with torch.no_grad():
        terminal_positions, terminal_quaternions = fk_model(paths_tensor[:, -1])
        terminal_errors = torch.linalg.norm(
            terminal_positions - condition[:, -3:], dim=1
        ).cpu().numpy()
        orientation_errors = np.asarray(
            quaternion_distance_rad(
                terminal_quaternions.cpu().numpy(), goal_quaternion[None]
            )
        )
        tcp_positions, _ = fk_model(
            paths_tensor.reshape(-1, paths_tensor.shape[-1])
        )
        tcp_positions_np = (
            tcp_positions.reshape(candidate_count, collision_screen_points, 3)
            .cpu()
            .numpy()
        )
    cloud = np.asarray(row["scene_points"])[
        np.asarray(row["scene_point_mask"], dtype=np.bool_)
    ]
    if stage == "place" and args.exclude_carried_target_from_cloud:
        target_center = np.asarray(row["object_center_foot0"], dtype=np.float32).reshape(3)
        target_size = float(np.max(np.asarray(row["object_size"], dtype=np.float32)))
        target_padding = 0.5 * target_size + args.carried_target_exclusion_padding_m
        outside_target = np.any(np.abs(cloud - target_center[None]) > target_padding, axis=1)
        cloud = cloud[outside_target]
    if args.scene_point_limit and len(cloud) > args.scene_point_limit:
        point_indices = np.linspace(
            0, len(cloud) - 1, args.scene_point_limit, dtype=np.int64
        )
        cloud = cloud[point_indices]
    environment = environment_clearance_batch(
        segments[:, :, 1:], cloud, capsule_radius_m=0.05
    )
    self_distances = self_clearance_batch(
        segments.reshape(-1, segments.shape[2], 2, 3), args.self_capsule_radius_m
    ).reshape(candidate_count, collision_screen_points).min(axis=1)
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    collision_screening_ms = 1000.0 * (perf_counter() - collision_started)

    ranking_started = perf_counter()
    metrics = []
    for index in range(candidate_count):
        unwrapped = np.unwrap(evaluation_paths_np[index], axis=0, period=2.0 * np.pi)
        time = np.linspace(0.0, float(durations_np[index]), evaluation_points)
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
            and orientation_errors[index] <= args.orientation_tolerance_rad
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
                "raw_orientation_error_rad": float(raw_orientation_errors[index]),
                "terminal_error_m": float(terminal_errors[index]),
                "orientation_error_rad": float(orientation_errors[index]),
                "environment_clearance_m": float(environment[index]),
                "self_clearance_m": float(self_distances[index]),
                "velocity_rms_rad_s": float(np.sqrt(np.mean(velocity**2))),
                "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))),
                "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk**2))),
                "joint_path_length_rad": float(
                    np.sum(np.linalg.norm(np.diff(unwrapped, axis=0), axis=1))
                ),
                "duration_s": float(durations_np[index]),
                "duration_exceeds_nominal_max": bool(
                    duration_exceeds_nominal[index]
                ),
                "terminal_ok": bool(terminal_errors[index] <= args.position_tolerance_m),
                "orientation_ok": bool(
                    orientation_errors[index] <= args.orientation_tolerance_rad
                ),
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
        + 0.1 * item["orientation_error_rad"]
        + 0.002 * item["acceleration_rms_rad_s2"]
        + 0.0002 * item["jerk_rms_rad_s3"],
        default=min(
            metrics,
            key=lambda item: item["terminal_error_m"]
            + 0.1 * item["orientation_error_rad"],
        ),
    )
    ranked = sorted(
        metrics,
        key=lambda item: (
            not item["success"],
            item["terminal_error_m"]
            + 0.1 * item["orientation_error_rad"]
            + 40.0 * max(
                0.0,
                args.required_clearance_m
                - min(item["environment_clearance_m"], item["self_clearance_m"]),
            )
            + 0.002 * item["acceleration_rms_rad_s2"]
            + 0.0002 * item["jerk_rms_rad_s3"],
        ),
    )
    candidate_ranking_ms = 1000.0 * (perf_counter() - ranking_started)
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    total_stage_ms = 1000.0 * (perf_counter() - stage_started)
    return {
        "stage": stage,
        "single": metrics[0],
        "multi": selected,
        "ranked_indices": [item["index"] for item in ranked[: args.candidate_bank_size]],
        "constraint_pass_rates": {
            key: float(np.mean([item[key] for item in metrics]))
            for key in (
                "terminal_ok",
                "orientation_ok",
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
                "orientation_ok",
                "environment_ok",
                "self_clearance_ok",
                "dynamics_ok",
                "duration_ok",
                "success",
            )
        },
        "paths_np": paths_np,
        "tcp_positions_np": tcp_positions_np,
        "durations_np": durations_np,
        "raw_paths_for_debug": raw_paths_for_debug,
        "reference_paths_for_debug": np.asarray(reference_paths_for_debug, dtype=np.float32),
        "residual_controls_for_debug": np.asarray(residual_controls_for_debug, dtype=np.float32),
        "reconstructed_controls_for_debug": np.asarray(
            reconstructed_controls_for_debug, dtype=np.float32
        ),
        "metrics": metrics,
        "timing_breakdown_ms": {
            "vision_encoding_ms": float(
                sampling_timing.get("vision_encoding_ms", float("nan"))
            ),
            "cfm_sampling_ms": float(
                sampling_timing.get("cfm_sampling_ms", float("nan"))
            ),
            "terminal_projection_residual_reconstruction_ms": reconstruction_ms,
            "densification_time_scaling_ms": densification_ms,
            "collision_screening_ms": collision_screening_ms,
            "candidate_ranking_ms": candidate_ranking_ms,
            "total_stage_ms": total_stage_ms,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model_type",
        choices=("visual_cfm", "condition_cfm", "condition_ddim"),
        default="visual_cfm",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument(
        "--task_start_index",
        type=int,
        default=0,
        help="Global dataset row at which to resume evaluation.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view_indices", default="")
    parser.add_argument("--rgb_noise_std", type=float, default=0.0)
    parser.add_argument("--depth_noise_std", type=float, default=0.0)
    parser.add_argument("--camera_blackout_indices", default="")
    parser.add_argument(
        "--semantic_mode", choices=("intact", "zero"), default="intact"
    )
    parser.add_argument("--zero_geometry_tokens", action="store_true")
    parser.add_argument("--disable_film_condition", action="store_true")
    parser.add_argument(
        "--scene_point_limit",
        type=int,
        default=0,
        help="deterministically subsample online collision-screen point cloud",
    )
    parser.add_argument("--independent_stage_generators", action="store_true")
    parser.add_argument("--position_tolerance_m", type=float, default=0.03)
    parser.add_argument(
        "--orientation_tolerance_rad", type=float, default=0.0872664626
    )
    parser.add_argument("--required_clearance_m", type=float, default=0.01)
    parser.add_argument("--self_capsule_radius_m", type=float, default=0.045)
    parser.add_argument("--exclude_carried_target_from_cloud", action="store_true")
    parser.add_argument("--carried_target_exclusion_padding_m", type=float, default=0.015)
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
    parser.add_argument(
        "--candidate_dump_dir",
        type=Path,
        default=None,
        help="Optional directory for one complete candidate-bank NPZ per task.",
    )
    parser.add_argument("--candidate_bank_size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if args.model_type == "visual_cfm":
        model = build_model(config, args.device)
    else:
        model = build_condition_only_model(config, args.model_type).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.disable_film_condition = args.disable_film_condition
    model.eval()
    fk_model = DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(
        args.device
    )
    robot_config = load_yaml(PROJECT_ROOT / "configs" / "robot.yaml")
    limits = robot_config["planning_soft_limits"]
    view_indices = [int(item) for item in args.view_indices.split(",") if item.strip()]
    blackout_indices = [
        int(item) for item in args.camera_blackout_indices.split(",") if item.strip()
    ]
    paths = sorted(args.data_dir.glob("test_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no test shards under {args.data_dir}")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            shard = {
                key: np.ascontiguousarray(data[key])
                for key in data.files
                if key != "camera_view_names"
            }
        rows.extend(
            {key: value[row] for key, value in shard.items()}
            for row in range(len(shard["condition_grasp"]))
        )
    if args.task_start_index < 0 or args.task_start_index >= len(rows):
        raise ValueError(
            f"task_start_index={args.task_start_index} outside [0, {len(rows) - 1}]"
        )
    task_offset = args.task_start_index
    rows = rows[task_offset:]
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
    started = perf_counter()
    for local_task_index, row in enumerate(rows):
        task_index = task_offset + local_task_index
        rgb = torch.as_tensor(row["rgb"], device=args.device)[None].clone()
        depth = torch.as_tensor(row["depth"], device=args.device)[None].clone()
        semantic = torch.as_tensor(row["semantic"], device=args.device)[None].clone()
        if view_indices:
            rgb = rgb[:, view_indices]
            depth = depth[:, view_indices]
            semantic = semantic[:, view_indices]
        perturbation_generator = torch.Generator(device=args.device).manual_seed(
            args.seed + 1_000_000 + task_index
        )
        if args.rgb_noise_std > 0.0:
            rgb = torch.clamp(
                rgb.float()
                + torch.randn(
                    rgb.shape,
                    device=rgb.device,
                    dtype=torch.float32,
                    generator=perturbation_generator,
                )
                * args.rgb_noise_std,
                0.0,
                255.0,
            )
        if args.depth_noise_std > 0.0:
            depth = torch.clamp(
                depth.float()
                + torch.randn(
                    depth.shape,
                    device=depth.device,
                    dtype=torch.float32,
                    generator=perturbation_generator,
                )
                * args.depth_noise_std,
                0.0,
                2.0,
            )
        if args.semantic_mode == "zero":
            semantic.zero_()
        for view_index in blackout_indices:
            if view_index < 0 or view_index >= rgb.shape[1]:
                raise IndexError(
                    f"camera blackout index {view_index} outside [0, {rgb.shape[1]})"
                )
            rgb[:, view_index].zero_()
            depth[:, view_index].zero_()
            semantic[:, view_index].zero_()
        vision = (rgb, depth, semantic)
        tokens = torch.as_tensor(
            row["obstacle_tokens"], dtype=torch.float32, device=args.device
        )[None]
        masks = torch.as_tensor(
            row["obstacle_mask"], dtype=torch.bool, device=args.device
        )[None]
        if args.zero_geometry_tokens:
            tokens.zero_()
        if args.independent_stage_generators:
            grasp_generator = torch.Generator(device=args.device).manual_seed(
                args.seed + 2 * task_index
            )
            place_generator = torch.Generator(device=args.device).manual_seed(
                args.seed + 2 * task_index + 1
            )
        else:
            grasp_generator = torch.Generator(device=args.device).manual_seed(
                args.seed + task_index
            )
            place_generator = grasp_generator

        target_label = np.asarray(row["target_component_label"], dtype=np.bool_)
        target_component = int(np.flatnonzero(target_label)[0]) if target_label.any() else -1
        if args.model_type == "visual_cfm":
            torch.cuda.synchronize() if str(args.device).startswith("cuda") else None
            component_started = perf_counter()
            with torch.no_grad():
                _, aux = model.encode_v621(
                    vision,
                    torch.as_tensor(row["condition_grasp"][-3:], dtype=torch.float32, device=args.device)[None],
                    tokens,
                    masks,
                )
            torch.cuda.synchronize() if str(args.device).startswith("cuda") else None
            component_encoding_ms = 1000.0 * (
                perf_counter() - component_started
            )
            component_logits = aux["component_logits"][0].cpu().numpy()
            mask_np = row["obstacle_mask"]
            visible = np.flatnonzero(mask_np)
            predicted_component = int(visible[np.argmax(component_logits[visible])])
            component_ok = predicted_component == target_component
            if "target_center" in aux:
                center_err = float(
                    torch.linalg.norm(
                        aux["target_center"][0].cpu()
                        - torch.as_tensor(row["object_center_foot0"])
                    )
                )
            else:
                center_err = float("nan")
            if "target_type_logits" in aux:
                type_ok = bool(
                    int(aux["target_type_logits"][0].argmax().item())
                    == int(np.asarray(row["object_type"]).reshape(-1)[0])
                )
            else:
                type_ok = None
        else:
            predicted_component = -1
            component_ok = False
            center_err = float("nan")
            type_ok = None
            component_encoding_ms = 0.0

        grasp_stage = evaluate_stage(
            model, fk_model, row, "grasp", vision, tokens, masks, limits, args,
            args.device, grasp_generator,
        )
        place_stage = evaluate_stage(
            model, fk_model, row, "place", vision, tokens, masks, limits, args,
            args.device, place_generator,
        )
        del rgb, depth, semantic
        torch.cuda.empty_cache()

        two_stage_success = bool(
            grasp_stage["multi"]["success"] and place_stage["multi"]["success"]
        )
        records.append(
            {
                "task_index": task_index,
                "sample_id": str(row["sample_id"]),
                "side": str(row["side"]),
                "target_component_ok": component_ok,
                "target_component_predicted": predicted_component,
                "target_component_truth": target_component,
                "target_center_error_m": center_err,
                "target_type_ok": type_ok,
                "grasp": grasp_stage["multi"],
                "place": place_stage["multi"],
                "two_stage_success": two_stage_success,
                "grasp_pass_rates": grasp_stage["constraint_pass_rates"],
                "place_pass_rates": place_stage["constraint_pass_rates"],
                "grasp_has_passing_candidate": grasp_stage["task_has_passing_candidate"],
                "place_has_passing_candidate": place_stage["task_has_passing_candidate"],
                "timing_breakdown_ms": {
                    "component_encoding_ms": component_encoding_ms,
                    "grasp": grasp_stage["timing_breakdown_ms"],
                    "place": place_stage["timing_breakdown_ms"],
                    "total_planning_ms": (
                        component_encoding_ms
                        + grasp_stage["timing_breakdown_ms"]["total_stage_ms"]
                        + place_stage["timing_breakdown_ms"]["total_stage_ms"]
                    ),
                },
            }
        )
        selected_trajectories.append(
            {
                "sample_id": np.asarray(str(row["sample_id"])),
                "side": np.asarray(str(row["side"])),
                "grasp_q_start": np.asarray(row["q_start"], dtype=np.float32),
                "grasp_goal_position": np.asarray(row["condition_grasp"][-3:], dtype=np.float32),
                "grasp_goal_quaternion": np.asarray(
                    row.get("goal_quaternion_grasp", [1.0, 0.0, 0.0, 0.0]),
                    dtype=np.float32,
                ),
                "grasp_q": np.asarray(grasp_stage["paths_np"][grasp_stage["multi"]["index"]], dtype=np.float32),
                "grasp_duration": np.float32(grasp_stage["durations_np"][grasp_stage["multi"]["index"]]),
                "grasp_success": np.bool_(grasp_stage["multi"]["success"]),
                "place_q_start": np.asarray(row["q_hold"], dtype=np.float32),
                "place_goal_position": np.asarray(row["condition_place"][-3:], dtype=np.float32),
                "place_goal_quaternion": np.asarray(
                    row.get("goal_quaternion_place", [1.0, 0.0, 0.0, 0.0]),
                    dtype=np.float32,
                ),
                "place_q": np.asarray(place_stage["paths_np"][place_stage["multi"]["index"]], dtype=np.float32),
                "place_duration": np.float32(place_stage["durations_np"][place_stage["multi"]["index"]]),
                "place_success": np.bool_(place_stage["multi"]["success"]),
                "two_stage_success": np.bool_(two_stage_success),
            }
        )
        if args.candidate_dump_dir is not None:
            args.candidate_dump_dir.mkdir(parents=True, exist_ok=True)

            def metric_array(
                stage_result: dict, key: str, dtype=np.float32
            ) -> np.ndarray:
                return np.asarray(
                    [item[key] for item in stage_result["metrics"]], dtype=dtype
                )

            dump_path = args.candidate_dump_dir / f"task_{task_index:04d}.npz"
            np.savez_compressed(
                dump_path,
                task_index=np.int32(task_index),
                sample_id=np.asarray(str(row["sample_id"])),
                scene_points=np.asarray(row["scene_points"], dtype=np.float32),
                scene_point_mask=np.asarray(row["scene_point_mask"], dtype=np.bool_),
                grasp_q=np.asarray(grasp_stage["paths_np"], dtype=np.float32),
                grasp_tcp=np.asarray(
                    grasp_stage["tcp_positions_np"], dtype=np.float32
                ),
                grasp_duration=np.asarray(
                    grasp_stage["durations_np"], dtype=np.float32
                ),
                grasp_success=metric_array(grasp_stage, "success", np.bool_),
                grasp_environment_clearance=metric_array(
                    grasp_stage, "environment_clearance_m"
                ),
                grasp_self_clearance=metric_array(
                    grasp_stage, "self_clearance_m"
                ),
                grasp_selected_index=np.int32(grasp_stage["multi"]["index"]),
                grasp_raw_cfm_q=np.asarray(
                    grasp_stage["raw_paths_for_debug"][grasp_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                grasp_minimum_jerk_q=np.asarray(
                    grasp_stage["reference_paths_for_debug"][grasp_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                grasp_residual_control_q=np.asarray(
                    grasp_stage["residual_controls_for_debug"][grasp_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                grasp_reconstructed_control_q=np.asarray(
                    grasp_stage["reconstructed_controls_for_debug"][grasp_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                place_q=np.asarray(place_stage["paths_np"], dtype=np.float32),
                place_tcp=np.asarray(
                    place_stage["tcp_positions_np"], dtype=np.float32
                ),
                place_duration=np.asarray(
                    place_stage["durations_np"], dtype=np.float32
                ),
                place_success=metric_array(place_stage, "success", np.bool_),
                place_environment_clearance=metric_array(
                    place_stage, "environment_clearance_m"
                ),
                place_self_clearance=metric_array(
                    place_stage, "self_clearance_m"
                ),
                place_selected_index=np.int32(place_stage["multi"]["index"]),
                place_raw_cfm_q=np.asarray(
                    place_stage["raw_paths_for_debug"][place_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                place_minimum_jerk_q=np.asarray(
                    place_stage["reference_paths_for_debug"][place_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                place_residual_control_q=np.asarray(
                    place_stage["residual_controls_for_debug"][place_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
                place_reconstructed_control_q=np.asarray(
                    place_stage["reconstructed_controls_for_debug"][place_stage["multi"]["index"]],
                    dtype=np.float32,
                ),
            )
        print(
            f"[EVAL] {local_task_index + 1}/{len(rows)} global={task_index} "
            f"target_ok={component_ok} "
            f"grasp_err={grasp_stage['multi']['terminal_error_m']:.4f} "
            f"place_err={place_stage['multi']['terminal_error_m']:.4f} "
            f"two_stage={two_stage_success}",
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "seed": args.seed,
        "tasks": len(records),
        "task_start_index": task_offset,
        "candidates": args.candidates,
        "ode_steps": args.ode_steps,
        "input_view_indices": view_indices or list(range(source_camera_views)),
        "rgb_noise_std": args.rgb_noise_std,
        "depth_noise_std": args.depth_noise_std,
        "camera_blackout_indices": blackout_indices,
        "semantic_mode": args.semantic_mode,
        "zero_geometry_tokens": args.zero_geometry_tokens,
        "disable_film_condition": args.disable_film_condition,
        "scene_point_limit": args.scene_point_limit,
        "independent_stage_generators": args.independent_stage_generators,
        "exclude_carried_target_from_cloud": args.exclude_carried_target_from_cloud,
        "carried_target_exclusion_padding_m": args.carried_target_exclusion_padding_m,
        "position_tolerance_m": args.position_tolerance_m,
        "orientation_tolerance_rad": args.orientation_tolerance_rad,
        "postprocess": args.postprocess,
        "safety_refine_steps": args.safety_refine_steps,
        "postprocess_mode": args.postprocess_mode,
        "geometry_control_points": args.geometry_control_points,
        "timing_points": args.timing_points,
        "unclipped_duration": args.unclipped_duration,
        "nominal_max_duration_s": args.nominal_max_duration_s,
        "max_output_duration_s": args.max_output_duration_s,
        "grasp": summarize_stage(records, "grasp"),
        "place": summarize_stage(records, "place"),
        "two_stage": {
            "tasks": len(records),
            "successes": int(sum(record["two_stage_success"] for record in records)),
            "success_rate": float(np.mean([record["two_stage_success"] for record in records])),
            "success_rate_wilson_95": wilson(
                int(sum(record["two_stage_success"] for record in records)), len(records)
            ),
        },
        "target_component_accuracy": float(
            np.mean([record["target_component_ok"] for record in records])
        ),
        "target_center_mean_error_m": (
            float(np.mean([record["target_center_error_m"] for record in records]))
            if args.model_type == "visual_cfm"
            else None
        ),
        "target_type_accuracy": (
            float(np.mean([record["target_type_ok"] for record in records]))
            if all(record["target_type_ok"] is not None for record in records)
            else None
        ),
        "mean_timing_breakdown_ms": {
            "component_encoding_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"]["component_encoding_ms"]
                        for record in records
                    ]
                )
            ),
            "vision_encoding_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage]["vision_encoding_ms"]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "cfm_sampling_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage]["cfm_sampling_ms"]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "terminal_projection_residual_reconstruction_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage][
                            "terminal_projection_residual_reconstruction_ms"
                        ]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "densification_time_scaling_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage][
                            "densification_time_scaling_ms"
                        ]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "collision_screening_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage]["collision_screening_ms"]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "candidate_ranking_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"][stage]["candidate_ranking_ms"]
                        for record in records
                        for stage in ("grasp", "place")
                    ]
                )
            ),
            "total_planning_ms": float(
                np.mean(
                    [
                        record["timing_breakdown_ms"]["total_planning_ms"]
                        for record in records
                    ]
                )
            ),
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
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
