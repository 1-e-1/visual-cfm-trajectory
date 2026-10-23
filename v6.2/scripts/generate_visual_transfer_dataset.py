"""Generate 1111-scene RGB-D trajectory transfer data in small reproducible shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import h5py  # noqa: F401
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--side", choices=("left", "right"), required=True)
parser.add_argument("--count", type=int, default=64)
parser.add_argument("--goals_per_image", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--expert_candidates", type=int, default=64)
parser.add_argument("--expert_labels", type=int, default=8, help="multi-modal expert labels stored per sample")
parser.add_argument("--ik_candidates", type=int, default=4, help="multi-restart IK solutions per goal (P2)")
parser.add_argument("--goal_distance_buckets", type=int, default=3, help="stratify goal pool by TCP distance")
parser.add_argument("--max_attempt_multiplier", type=int, default=200)
parser.add_argument("--stored_scene_points", type=int, default=2048)
parser.add_argument(
    "--require_obstacle_detour",
    action="store_true",
    help=(
        "Keep only tasks whose direct path collides with reconstructed environment "
        "geometry while remaining self-collision-free, and whose selected detour is safe."
    ),
)
parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data" / "visual_transfer")
parser.add_argument("--report", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.utils import stage as stage_utils
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom

from h1_1111_grasp.env_cfg import H1_1111_CFG
from h1_1111_grasp.rgbd_geometry import (
    fuse_depth_views,
    point_cloud_component_tokens,
    remove_current_arm_points,
)
from h1_1111_grasp.trajectory_safety import (
    environment_clearance,
    environment_clearance_batch,
    link_segments,
    self_clearance,
    self_clearance_batch,
)
from src.datasets.npz_dataset import write_npz_shards
from src.planning.ik_solver import solve_ik_candidates
from src.planning.trajectory_math import minimum_jerk_duration
from src.robot.torch_kinematics import DifferentiableZBotFK


CAMERAS = {
    "left": [
        ("wrist", "/World/UnitreeH1/ZBot6S_left/foot_1/Sensors/LeftWristCamera", "foot_1_0"),
        ("cube", "/World/UnitreeH1/Cube/Sensors/LeftCubeSideCamera", "Cube"),
        ("chest", "/World/UnitreeH1/torso_link/Sensors/ChestLeftCamera", "torso_link"),
        ("abdomen", "/World/UnitreeH1/pelvis/Sensors/LeftHipCamera", "pelvis"),
        ("head", "/World/UnitreeH1/torso_link/Sensors/HeadTopCamera", "torso_link"),
    ],
    "right": [
        ("wrist", "/World/UnitreeH1/ZBot6S_right/foot_1/Sensors/RightWristCamera", "foot_1"),
        ("cube", "/World/UnitreeH1/Cube/Sensors/RightCubeSideCamera", "Cube"),
        ("chest", "/World/UnitreeH1/torso_link/Sensors/ChestRightCamera", "torso_link"),
        ("abdomen", "/World/UnitreeH1/pelvis/Sensors/RightHipCamera", "pelvis"),
        ("head", "/World/UnitreeH1/torso_link/Sensors/HeadTopCamera", "torso_link"),
    ],
}
LOWER = np.full(6, -np.pi, dtype=np.float64)
UPPER = np.full(6, np.pi, dtype=np.float64)
VELOCITY_LIMITS = np.array([2.0, 2.0, 2.0, 2.5, 2.5, 3.0], dtype=np.float64)
ACCELERATION_LIMITS = np.array([8.0, 8.0, 8.0, 10.0, 10.0, 12.0], dtype=np.float64)
JERK_LIMITS = np.array([80.0, 80.0, 80.0, 100.0, 100.0, 120.0], dtype=np.float64)


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _minimum_jerk_progress(point_count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _candidate_paths(
    q_start: np.ndarray,
    q_goal_unwrapped: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build direct and one-waypoint periodic minimum-jerk candidates."""
    if count < 1:
        raise ValueError("expert_candidates must be positive")
    direct_progress = _minimum_jerk_progress(64)
    direct = q_start[None] + direct_progress[:, None] * (q_goal_unwrapped - q_start)[None]
    paths = [direct]
    first_progress = _minimum_jerk_progress(33)
    second_progress = _minimum_jerk_progress(32)
    midpoint = 0.5 * (q_start + q_goal_unwrapped)
    for _ in range(count - 1):
        waypoint = midpoint + rng.uniform(-1.2, 1.2, size=6)
        waypoint = q_start + _wrap_angles(waypoint - q_start)
        goal_from_waypoint = waypoint + _wrap_angles(q_goal_unwrapped - waypoint)
        first = q_start[None] + first_progress[:, None] * (waypoint - q_start)[None]
        second = waypoint[None] + second_progress[:, None] * (
            goal_from_waypoint - waypoint
        )[None]
        paths.append(np.concatenate((first, second[1:]), axis=0))
    return _wrap_angles(np.stack(paths))


def _select_expert_candidate(
    fk_model: torch.nn.Module,
    paths: np.ndarray,
    planning_cloud: np.ndarray,
) -> tuple[int | None, float, float, float, float]:
    flat_segments = link_segments(
        fk_model, paths.reshape(-1, 6), device="cuda:0"
    ).reshape(len(paths), 64, -1, 2, 3)
    environment_values = environment_clearance_batch(
        flat_segments[:, :, 1:], planning_cloud, capsule_radius_m=0.05
    )
    self_values = self_clearance_batch(
        flat_segments.reshape(-1, flat_segments.shape[2], 2, 3),
        capsule_radius_m=0.045,
    ).reshape(len(paths), 64).min(axis=1)
    selected = None
    selected_score = -float("inf")
    selected_environment = -float("inf")
    selected_self = -float("inf")
    for index, (environment, self_distance) in enumerate(
        zip(environment_values, self_values)
    ):
        if min(environment, self_distance) < 0.01:
            continue
        continuous = np.unwrap(paths[index], axis=0, period=2.0 * np.pi)
        path_length = float(np.abs(np.diff(continuous, axis=0)).sum())
        score = min(environment, self_distance) - 2.0e-4 * path_length
        if score > selected_score:
            selected = index
            selected_score = score
            selected_environment = environment
            selected_self = self_distance
    return (
        selected,
        selected_environment,
        selected_self,
        float(environment_values[0]),
        float(self_values[0]),
    )


def _rank_expert_candidates(
    fk_model: torch.nn.Module,
    paths: np.ndarray,
    planning_cloud: np.ndarray,
    min_clearance_m: float = 0.01,
) -> list[int]:
    """Return candidate indices that pass env/self clearance, sorted by
    clearance margin minus path length penalty (best first)."""
    flat_segments = link_segments(
        fk_model, paths.reshape(-1, 6), device="cuda:0"
    ).reshape(len(paths), 64, -1, 2, 3)
    environment_values = environment_clearance_batch(
        flat_segments[:, :, 1:], planning_cloud, capsule_radius_m=0.05
    )
    self_values = self_clearance_batch(
        flat_segments.reshape(-1, flat_segments.shape[2], 2, 3),
        capsule_radius_m=0.045,
    ).reshape(len(paths), 64).min(axis=1)
    ranked: list[tuple[float, int]] = []
    for index, (environment, self_distance) in enumerate(
        zip(environment_values, self_values)
    ):
        if min(environment, self_distance) < min_clearance_m:
            continue
        continuous = np.unwrap(paths[index], axis=0, period=2.0 * np.pi)
        path_length = float(np.abs(np.diff(continuous, axis=0)).sum())
        score = min(environment, self_distance) - 2.0e-4 * path_length
        ranked.append((score, index))
    ranked.sort(reverse=True)
    return [index for _, index in ranked]


def _bucket_goal_pool(
    fk_model: torch.nn.Module,
    pool: np.ndarray,
    rng: np.random.Generator,
    buckets: int = 3,
) -> np.ndarray:
    """Stratify the goal pool by FK TCP distance so short and long reaches are
    represented instead of concentrating on the medium-distance shell."""
    if buckets <= 1 or len(pool) < buckets * 8:
        return pool
    with torch.no_grad():
        positions, _ = fk_model(torch.as_tensor(pool, dtype=torch.float32, device="cuda:0"))
    distances = torch.linalg.norm(positions, dim=-1).cpu().numpy()
    quantiles = np.quantile(distances, np.linspace(0.0, 1.0, buckets + 1)[1:-1])
    per_bucket = len(pool) // buckets
    selected = []
    for index in range(buckets):
        lower = -np.inf if index == 0 else quantiles[index - 1]
        upper = np.inf if index == buckets - 1 else quantiles[index]
        mask = (distances >= lower) & (distances < upper)
        indices = np.flatnonzero(mask)
        if len(indices):
            chosen = rng.choice(indices, size=min(per_bucket, len(indices)), replace=False)
            selected.append(pool[chosen])
    if not selected:
        return pool
    result = np.concatenate(selected, axis=0)
    rng.shuffle(result)
    return result


def _sample_self_collision_free_states(
    fk_model: torch.nn.Module,
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, int]:
    selected = []
    draws = 0
    while sum(len(batch) for batch in selected) < count:
        remaining = count - sum(len(batch) for batch in selected)
        batch_size = min(65536, max(4096, remaining * 70))
        proposals = rng.uniform(LOWER, UPPER, size=(batch_size, 6))
        segments = link_segments(fk_model, proposals, device="cuda:0")
        clearance = self_clearance_batch(segments, capsule_radius_m=0.045)
        safe = proposals[clearance >= 0.01]
        if len(safe):
            selected.append(safe[:remaining])
        draws += batch_size
    return np.concatenate(selected, axis=0), draws


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _fixed_scene_points(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("stored_scene_points must be positive")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    output = np.zeros((count, 3), dtype=np.float32)
    mask = np.zeros(count, dtype=np.bool_)
    selected_count = min(len(points), count)
    if selected_count:
        indices = np.linspace(0, len(points) - 1, selected_count).round().astype(int)
        output[:selected_count] = points[indices]
        mask[:selected_count] = True
    return output, mask


def _planning_depth_without_active_arm(
    depth_views: np.ndarray,
    semantic_packets: list[dict],
    side: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    filtered = np.asarray(depth_views, dtype=np.float32).copy()
    active_label = f"aux_arm_{side}"
    reports = []
    for view, packet in enumerate(semantic_packets):
        semantic = np.asarray(packet["data"], dtype=np.int32)
        id_to_labels = packet.get("info", {}).get("idToLabels", {})
        removed_ids = []
        for semantic_id, labels in id_to_labels.items():
            class_name = str(labels.get("class", ""))
            # Finite UNLABELLED pixels can be real scene geometry (for example the
            # floor or an imported mesh without semantics).  Only suppress the
            # active arm here; invalid/background depth is rejected downstream.
            if active_label in class_name.split(","):
                removed_ids.append(int(semantic_id))
        remove = np.isin(semantic, removed_ids)
        filtered[view, remove] = np.nan
        reports.append(
            {
                "removed_semantic_ids": removed_ids,
                "removed_pixel_ratio": float(remove.mean()),
                "semantic_id_to_labels": id_to_labels,
            }
        )
    return filtered, reports


def _quat_from_matrix(matrix) -> tuple[float, float, float, float]:
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        float(rotation.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _camera_local_poses(stage, camera_specs):
    poses = []
    for _, camera_path, parent_name in camera_specs:
        camera_prim = stage.GetPrimAtPath(camera_path)
        parent_path = camera_path.split("/Sensors/", 1)[0]
        parent_prim = stage.GetPrimAtPath(parent_path)
        camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        relative = camera_world * parent_world.GetInverse()
        poses.append(
            (
                parent_name,
                tuple(float(value) for value in relative.ExtractTranslation()),
                _quat_from_matrix(relative),
            )
        )
    return poses


def _intrinsic_matrix(stage, camera_path: str, resolution: int) -> np.ndarray:
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    fx = focal / horizontal_aperture * resolution
    fy = focal / vertical_aperture * resolution
    center = 0.5 * (resolution - 1)
    return np.array([[fx, 0.0, center], [0.0, fy, center], [0.0, 0.0, 1.0]])


def main() -> None:
    if args.count < 8 or args.goals_per_image < 1:
        raise ValueError("count must be >= 8 and goals_per_image must be positive")
    rng = _seed_everything(args.seed + (0 if args.side == "left" else 1))
    stage_utils.open_stage(str(WORKSPACE_ROOT / "1111_copy.usd"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=1))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))
    stage = omni.usd.get_context().get_stage()
    camera_specs = CAMERAS[args.side]
    local_camera_poses = _camera_local_poses(stage, camera_specs)
    intrinsics = np.stack(
        [_intrinsic_matrix(stage, path, args.resolution) for _, path, _ in camera_specs]
    )
    products = []
    annotators = []
    for _, path, _ in camera_specs:
        product = rep.create.render_product(path, resolution=(args.resolution, args.resolution))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
        semantic = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"semanticTypes": ["class"], "colorize": False},
            device="cpu",
        )
        rgb.attach([product])
        depth.attach([product])
        semantic.attach([product])
        products.append(product)
        annotators.append((rgb, depth, semantic))
    sim.reset()
    robot.reset()

    suffix = "_0" if args.side == "left" else ""
    arm_ids, arm_names = robot.find_joints([f"^joint{i}{suffix}$" for i in range(1, 7)], preserve_order=True)
    base_name = f"foot_0{suffix}"
    base_ids, _ = robot.find_bodies([f"^{base_name}$"])
    if len(arm_ids) != 6 or len(base_ids) != 1:
        raise RuntimeError(f"failed to resolve {args.side} arm: joints={arm_names}, base={base_name}")
    parent_ids = {}
    for parent_name, _, _ in local_camera_poses:
        ids, names = robot.find_bodies([f"^{parent_name}$"])
        if len(ids) != 1:
            raise RuntimeError(f"failed to resolve camera parent {parent_name}: {names}")
        parent_ids[parent_name] = ids[0]

    fk_model = DifferentiableZBotFK(WORKSPACE_ROOT / "configs" / "kinematic_chain.json").to("cuda:0")
    samples = []
    attempts = 0
    rejection_counts = {
        "start_self_collision": 0,
        "goal_self_collision": 0,
        "start_environment_collision": 0,
        "goal_environment_collision": 0,
        "duration_infeasible": 0,
        "direct_path_not_obstacle_blocked": 0,
        "all_candidates_rejected": 0,
        "ik_failed": 0,
    }
    if args.require_obstacle_detour:
        start_pool_count = max(
            8,
            int(np.ceil(args.count / args.goals_per_image))
            * args.max_attempt_multiplier,
        )
        goal_pool_count = max(16, start_pool_count * args.goals_per_image)
    else:
        start_pool_count = max(8, int(np.ceil(args.count / args.goals_per_image)) * 2)
        goal_pool_count = max(16, args.count * 2)
    start_pool, start_uniform_draws = _sample_self_collision_free_states(
        fk_model, rng, start_pool_count
    )
    goal_pool, goal_uniform_draws = _sample_self_collision_free_states(
        fk_model, rng, goal_pool_count
    )
    goal_pool = _bucket_goal_pool(fk_model, goal_pool, rng, args.goal_distance_buckets)
    rejection_counts["start_self_collision"] = start_uniform_draws - len(start_pool)
    rejection_counts["goal_self_collision"] = goal_uniform_draws - len(goal_pool)
    goal_cursor = 0
    started = perf_counter()
    all_joint_pos = robot.data.default_joint_pos.clone()
    zero_joint_vel = torch.zeros_like(all_joint_pos)
    while len(samples) < args.count and attempts < len(start_pool):
        attempts += 1
        q_start = start_pool[attempts - 1]
        all_joint_pos[:, arm_ids] = torch.as_tensor(
            q_start,
            dtype=all_joint_pos.dtype,
            device=robot.device,
        )
        robot.write_root_state_to_sim(robot.data.default_root_state)
        robot.write_joint_state_to_sim(all_joint_pos, zero_joint_vel)
        robot.set_joint_position_target(all_joint_pos)
        for _ in range(3):
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
        rgb_views = np.stack(
            [np.asarray(rgb.get_data())[..., :3] for rgb, _, _ in annotators]
        ).astype(np.uint8)
        depth_views = np.stack(
            [np.asarray(depth.get_data()) for _, depth, _ in annotators]
        ).astype(np.float32)
        semantic_packets = [semantic.get_data() for _, _, semantic in annotators]
        semantic_views = np.stack(
            [np.asarray(packet["data"]) for packet in semantic_packets]
        ).astype(np.int32)
        if any(value.size == 0 for value in (rgb_views, depth_views)):
            continue

        camera_positions = []
        camera_quaternions_ros = []
        for parent_name, local_position, local_quaternion in local_camera_poses:
            parent_id = parent_ids[parent_name]
            position, quaternion = combine_frame_transforms(
                robot.data.body_pos_w[:, parent_id],
                robot.data.body_quat_w[:, parent_id],
                torch.tensor([local_position], dtype=torch.float32, device=robot.device),
                torch.tensor([local_quaternion], dtype=torch.float32, device=robot.device),
            )
            quaternion_ros = convert_camera_frame_orientation_convention(
                quaternion, origin="opengl", target="ros"
            )
            camera_positions.append(position[0].cpu().numpy())
            camera_quaternions_ros.append(quaternion_ros[0].cpu().numpy())
        base_position = robot.data.body_pos_w[0, base_ids[0]].cpu().numpy()
        base_quaternion = robot.data.body_quat_w[0, base_ids[0]].cpu().numpy()
        planning_depth_views, semantic_filter_report = _planning_depth_without_active_arm(
            depth_views, semantic_packets, args.side
        )
        cloud = fuse_depth_views(
            planning_depth_views,
            intrinsics,
            np.asarray(camera_positions),
            np.asarray(camera_quaternions_ros),
            base_position,
            base_quaternion,
            stride=3,
            voxel_size_m=0.015,
        )
        current_segments = link_segments(fk_model, q_start[None], device="cuda:0")[0]
        planning_cloud = remove_current_arm_points(cloud, current_segments, removal_radius_m=0.075)
        current_environment = environment_clearance(
            current_segments[1:], planning_cloud, capsule_radius_m=0.05
        )
        if current_environment < 0.01:
            rejection_counts["start_environment_collision"] += 1
            print(
                f"[DATA] side={args.side} reject_start_environment={current_environment:.4f} "
                f"attempt={attempts}",
                flush=True,
            )
            continue
        tokens, mask = point_cloud_component_tokens(planning_cloud)
        stored_scene_points, stored_scene_point_mask = _fixed_scene_points(
            planning_cloud, args.stored_scene_points
        )

        for _ in range(args.goals_per_image):
            if len(samples) >= args.count:
                break
            if goal_cursor >= len(goal_pool):
                continue
            q_goal = goal_pool[goal_cursor]
            goal_cursor += 1
            shortest_delta = _wrap_angles(q_goal - q_start)
            q_goal_unwrapped = q_start + shortest_delta
            duration, feasible, _ = minimum_jerk_duration(
                q_start,
                q_goal_unwrapped,
                VELOCITY_LIMITS,
                ACCELERATION_LIMITS,
                JERK_LIMITS,
                min_duration=1.0,
                max_duration=5.0,
            )
            if not feasible:
                rejection_counts["duration_infeasible"] += 1
                continue
            goal_segments = link_segments(fk_model, _wrap_angles(q_goal)[None], device="cuda:0")[0]
            goal_environment = environment_clearance(
                goal_segments[1:], planning_cloud, capsule_radius_m=0.05
            )
            if goal_environment < 0.01:
                rejection_counts["goal_environment_collision"] += 1
                continue
            if args.require_obstacle_detour:
                direct_path = _candidate_paths(q_start, q_goal_unwrapped, 1, rng)
                (
                    _,
                    _,
                    _,
                    direct_environment,
                    direct_self_distance,
                ) = _select_expert_candidate(fk_model, direct_path, planning_cloud)
                if not (
                    direct_environment < 0.01 and direct_self_distance >= 0.01
                ):
                    rejection_counts["direct_path_not_obstacle_blocked"] += 1
                    continue
            paths = _candidate_paths(q_start, q_goal_unwrapped, args.expert_candidates, rng)
            (
                candidate_index,
                environment,
                self_distance,
                direct_environment,
                direct_self_distance,
            ) = _select_expert_candidate(fk_model, paths, planning_cloud)
            if candidate_index is None:
                rejection_counts["all_candidates_rejected"] += 1
                continue
            wrapped_trajectory = paths[candidate_index]
            unwrapped_trajectory = np.unwrap(
                wrapped_trajectory, axis=0, period=2.0 * np.pi
            )
            unwrapped_trajectory += (q_start - unwrapped_trajectory[0])[None]
            duration = max(duration, 4.0 if candidate_index else duration)
            with torch.no_grad():
                goal_position, _ = fk_model(torch.as_tensor(q_goal, dtype=torch.float32, device="cuda:0"))
            ranked_indices = _rank_expert_candidates(fk_model, paths, planning_cloud)
            selected_experts = []
            for ranked_index in ranked_indices[: args.expert_labels]:
                selected_experts.append(_wrap_angles(paths[ranked_index]))
            while len(selected_experts) < args.expert_labels:
                selected_experts.append(wrapped_trajectory)
            q_experts = np.stack(selected_experts[: args.expert_labels]).astype(np.float32)

            def _ik_clearance(q_row: np.ndarray) -> float:
                q_row_2d = np.asarray(q_row, dtype=np.float64).reshape(1, 6)
                segments = link_segments(fk_model, q_row_2d, device="cuda:0")
                if segments.ndim != 4:
                    raise ValueError(f"unexpected segments shape {segments.shape}")
                return float(self_clearance_batch(segments, capsule_radius_m=0.045).min())

            q_ik_candidates = solve_ik_candidates(
                fk_model,
                goal_position.cpu().numpy(),
                n_candidates=args.ik_candidates,
                restarts=8,
                iterations=100,
                position_tolerance_m=0.005,
                self_clearance_fn=_ik_clearance,
                seed=int(attempts) * 7919 + goal_cursor,
                device="cuda:0",
            )
            if len(q_ik_candidates) == 0:
                rejection_counts["ik_failed"] += 1
                continue
            if len(q_ik_candidates) < args.ik_candidates:
                pad_rows = args.ik_candidates - len(q_ik_candidates)
                q_ik_candidates = np.concatenate(
                    (q_ik_candidates, np.repeat(q_ik_candidates[:1], pad_rows, axis=0)),
                    axis=0,
                )
            condition_v6 = np.concatenate(
                (q_start, q_ik_candidates[0], goal_position.cpu().numpy())
            ).astype(np.float32)
            samples.append(
                {
                    "sample_id": np.asarray(f"{args.side}_{len(samples):06d}"),
                    "side": np.asarray(args.side),
                    "condition": condition_v6,
                    "q_start": q_start.astype(np.float32),
                    "q_goal": _wrap_angles(q_goal).astype(np.float32),
                    "q": wrapped_trajectory.astype(np.float32),
                    "q_unwrapped": unwrapped_trajectory.astype(np.float32),
                    "q_experts": q_experts,
                    "q_ik_candidates": q_ik_candidates.astype(np.float32),
                    "duration": np.float32(duration),
                    "expert_candidate_index": np.int32(candidate_index),
                    "expert_candidates": np.int32(args.expert_candidates),
                    "obstacle_tokens": tokens,
                    "obstacle_mask": mask,
                    "scene_points": stored_scene_points,
                    "scene_point_mask": stored_scene_point_mask,
                    "semantic_filter_removed_ratio": np.asarray(
                        [item["removed_pixel_ratio"] for item in semantic_filter_report],
                        dtype=np.float32,
                    ),
                    "rgb": np.repeat(rgb_views[:, None], 2, axis=1),
                    "depth": np.repeat(depth_views[:, None, ..., None], 2, axis=1).astype(np.float16),
                    "semantic": np.repeat(
                        semantic_views[:, None, ..., None], 2, axis=1
                    ).astype(np.int32),
                    "visual_clearance_m": np.float32(environment),
                    "self_clearance_m": np.float32(self_distance),
                    "direct_environment_clearance_m": np.float32(direct_environment),
                    "direct_self_clearance_m": np.float32(direct_self_distance),
                    "collision_free": np.bool_(True),
                }
            )
        print(f"[DATA] side={args.side} accepted={len(samples)}/{args.count} attempts={attempts}", flush=True)

    if len(samples) < args.count:
        raise RuntimeError(
            f"generated only {len(samples)}/{args.count} samples; rejections={rejection_counts}"
        )
    indices = rng.permutation(len(samples))
    train_end = max(1, int(0.8 * len(samples)))
    validation_end = max(train_end + 1, int(0.9 * len(samples)))
    side_dir = args.output_dir / args.side
    shards = {}
    for split, selected in {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }.items():
        shards[split] = [
            str(path) for path in write_npz_shards(samples, selected, side_dir, split, shard_size=64)
        ]
        for shard_path in shards[split]:
            with np.load(shard_path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["camera_view_names"] = np.asarray(
                ["wrist", "cube", "chest", "abdomen", "head"]
            )
            np.savez_compressed(shard_path, **payload)
    report = {
        "side": args.side,
        "seed": args.seed,
        "accepted": len(samples),
        "attempts": attempts,
        "elapsed_s": perf_counter() - started,
        "shards": shards,
        "condition": "q_start[6] + q_ik[6] + target_position_in_arm_foot_0[3] (v6)",
        "condition_version": "v6",
        "h1_state_read": False,
        "trajectory_shape": [64, 6],
        "periodic_joint_range_rad": [-float(np.pi), float(np.pi)],
        "expert_labels": args.expert_labels,
        "ik_candidates": args.ik_candidates,
        "goal_distance_buckets": args.goal_distance_buckets,
        "expert_candidates": args.expert_candidates,
        "require_obstacle_detour": args.require_obstacle_detour,
        "camera_views": ["wrist", "cube", "chest", "abdomen", "head"],
        "rejection_counts": rejection_counts,
        "uniform_draws": {
            "start": start_uniform_draws,
            "goal": goal_uniform_draws,
        },
    }
    report_path = args.report or side_dir / "generation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=True)
