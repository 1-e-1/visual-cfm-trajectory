"""Generate grasp-place (two-stage) dataset with randomized reachable objects (v6.2.1).

Each sample: 2..N randomized objects in the arm reachable shell; one is the
target object (semantic class "target_object"), the others are obstacles
("graspable_<shape>").  The task is two-stage:
  grasp:  q_start -> q_grasp_ik (pregrasp point above the target object)
  place:  q_hold  -> q_place_ik (random reachable placement point)
Target objects must be reachable (IK solvable, self/env clearance safe), otherwise
the sample is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--side", choices=("left", "right"), required=True)
parser.add_argument("--count", type=int, default=64)
parser.add_argument("--seed", type=int, default=20260807)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--expert_candidates", type=int, default=64)
parser.add_argument("--expert_labels", type=int, default=8)
parser.add_argument("--ik_candidates", type=int, default=4)
parser.add_argument("--n_objects_min", type=int, default=2)
parser.add_argument("--n_objects_max", type=int, default=5)
parser.add_argument("--object_size_min", type=float, default=0.04)
parser.add_argument("--object_size_max", type=float, default=0.10)
parser.add_argument("--reach_min_m", type=float, default=0.42)
parser.add_argument("--reach_max_m", type=float, default=0.60)
parser.add_argument("--z_min_m", type=float, default=0.20)
parser.add_argument("--z_max_m", type=float, default=0.40)
parser.add_argument("--pregrasp_offset_m", type=float, default=0.14, help="clearance above the object top surface (grasp release point)")
parser.add_argument("--min_object_spacing_m", type=float, default=0.16)
parser.add_argument("--min_place_distance_m", type=float, default=0.15)
parser.add_argument("--layouts", type=int, default=1, help="number of independent object layouts per run")
parser.add_argument("--anchor_pool_size", type=int, default=1024, help="self-collision-free FK anchor pool size")
parser.add_argument("--max_attempts_per_layout", type=int, default=200)
parser.add_argument("--max_attempt_multiplier", type=int, default=200)
parser.add_argument(
    "--minimum_target_pixels_total",
    type=int,
    default=0,
    help="Reject a sample when the target occupies fewer semantic pixels across all cameras.",
)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--stored_scene_points", type=int, default=2048)
parser.add_argument(
    "--physical_contact",
    action="store_true",
    help="Generate target centers from calibrated gripper contact poses.",
)
parser.add_argument("--approach_distance_m", type=float, default=0.08)
parser.add_argument("--lift_distance_m", type=float, default=0.10)
parser.add_argument("--preplace_distance_m", type=float, default=0.08)
parser.add_argument(
    "--physical_anchor_limit",
    type=int,
    default=16,
    help="Maximum complete contact anchors precomputed per generator process.",
)
parser.add_argument(
    "--physical_anchor_source_dir",
    type=Path,
    default=None,
    help="Reuse validated contact/pregrasp/place anchors from physical-contact shards.",
)
parser.add_argument(
    "--physical_target_anchor_index",
    type=int,
    default=-1,
    help="Fix the pickup anchor for an application dataset; -1 samples all anchors.",
)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument("--report", type=Path, default=None)
parser.add_argument(
    "--usd",
    type=Path,
    default=PROJECT_ROOT / "1111_copy.usd",
    help="Composed scene USD used for camera and robot data generation.",
)
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade

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
    self_clearance_batch,
)
from src.datasets.npz_dataset import write_npz_shards
from src.planning.grasp_pose import (
    grasp_center_pose_to_tcp_pose,
    quaternion_to_matrix_wxyz,
    tcp_pose_to_grasp_center_pose,
)
from src.planning.ik_solver import solve_ik_candidates, solve_pose_ik_candidates
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
SHAPES = ["box", "cylinder", "sphere", "cone", "capsule"]
TCP_TO_GRASP_POSITION = np.array([0.0001097, 0.0083994, 0.1396553], dtype=np.float64)
TCP_TO_GRASP_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
WORLD_UP_FOOT0 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
NOMINAL_TOP_GRASP_Q = np.array(
    [-2.1070001, 0.96, 0.597, 0.476, 0.098, -0.029], dtype=np.float32
)


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
def _rank_expert_candidates(
    fk_model: torch.nn.Module,
    paths: np.ndarray,
    planning_cloud: np.ndarray,
    min_clearance_m: float = 0.01,
    self_capsule_radius_m: float = 0.045,
) -> list[int]:
    flat_segments = link_segments(
        fk_model, paths.reshape(-1, 6), device="cuda:0"
    ).reshape(len(paths), 64, -1, 2, 3)
    environment_values = environment_clearance_batch(
        flat_segments[:, :, 1:], planning_cloud, capsule_radius_m=0.05
    )
    self_values = self_clearance_batch(
        flat_segments.reshape(-1, flat_segments.shape[2], 2, 3),
        capsule_radius_m=self_capsule_radius_m,
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


def _select_best_candidate(
    fk_model: torch.nn.Module,
    paths: np.ndarray,
    planning_cloud: np.ndarray,
    self_capsule_radius_m: float = 0.045,
) -> tuple[int | None, float, float]:
    flat_segments = link_segments(
        fk_model, paths.reshape(-1, 6), device="cuda:0"
    ).reshape(len(paths), 64, -1, 2, 3)
    environment_values = environment_clearance_batch(
        flat_segments[:, :, 1:], planning_cloud, capsule_radius_m=0.05
    )
    self_values = self_clearance_batch(
        flat_segments.reshape(-1, flat_segments.shape[2], 2, 3),
        capsule_radius_m=self_capsule_radius_m,
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
    return selected, selected_environment, selected_self


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


def _fixed_scene_points(
    points: np.ndarray,
    count: int,
    priority_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("stored_scene_points must be positive")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    priority = (
        np.asarray(priority_points, dtype=np.float32).reshape(-1, 3)
        if priority_points is not None and len(priority_points)
        else np.empty((0, 3), dtype=np.float32)
    )
    output = np.zeros((count, 3), dtype=np.float32)
    mask = np.zeros(count, dtype=np.bool_)
    priority_count = min(len(priority), count)
    if priority_count:
        output[:priority_count] = priority[:priority_count]
        mask[:priority_count] = True
    remaining = count - priority_count
    if remaining > 0 and len(points):
        selected_count = min(len(points), remaining)
        indices = np.linspace(0, len(points) - 1, selected_count).round().astype(int)
        output[priority_count : priority_count + selected_count] = points[indices]
        mask[priority_count : priority_count + selected_count] = True
    return output, mask


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
        camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
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


def _planning_depth_without_active_arm(
    depth_views: np.ndarray,
    semantic_packets: list[dict],
    side: str,
) -> np.ndarray:
    filtered = np.asarray(depth_views, dtype=np.float32).copy()
    active_label = f"aux_arm_{side}"
    for view, packet in enumerate(semantic_packets):
        semantic = np.asarray(packet["data"], dtype=np.int32)
        id_to_labels = packet.get("info", {}).get("idToLabels", {})
        removed_ids = []
        for semantic_id, labels in id_to_labels.items():
            class_name = str(labels.get("class", ""))
            if active_label in class_name.split(","):
                removed_ids.append(int(semantic_id))
        remove = np.isin(semantic, removed_ids)
        filtered[view, remove] = np.nan
    return filtered


def _sample_object_layout(
    rng: np.random.Generator,
    count: int,
    reach_min: float,
    reach_max: float,
    z_min: float,
    z_max: float,
    spacing: float,
) -> np.ndarray:
    """Sample object centers in the arm-foot0 local frame (meters)."""
    centers = []
    attempts = 0
    while len(centers) < count and attempts < 5000:
        attempts += 1
        radius = rng.uniform(reach_min, reach_max)
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        z = rng.uniform(z_min, z_max)
        xy_radius = np.sqrt(max(radius**2 - z**2, 1.0e-6))
        center = np.array(
            [xy_radius * np.cos(azimuth), xy_radius * np.sin(azimuth), z]
        )
        if any(np.linalg.norm(center - existing) < spacing for existing in centers):
            continue
        centers.append(center)
    if len(centers) < count:
        raise RuntimeError(f"failed to place {count} non-overlapping objects")
    return np.asarray(centers, dtype=np.float64)

def _create_shape_prim(
    stage: Usd.Stage,
    parent_path: str,
    name: str,
    shape: str,
    size: float,
    color: np.ndarray | None = None,
) -> Usd.Prim:
    prim_path = f"{parent_path}/{name}"
    prim = stage.DefinePrim(prim_path, "Xform")
    geometry_path = f"{prim_path}/Geometry"
    geom_type = {
        "box": "Cube",
        "cylinder": "Cylinder",
        "sphere": "Sphere",
        "cone": "Cone",
        "capsule": "Capsule",
    }[shape]
    geom = stage.DefinePrim(geometry_path, geom_type)
    if shape == "box":
        UsdGeom.Cube(geom).CreateSizeAttr(size)
    elif shape == "cylinder":
        UsdGeom.Cylinder(geom).CreateHeightAttr(size)
        UsdGeom.Cylinder(geom).CreateRadiusAttr(0.5 * size)
    elif shape == "sphere":
        UsdGeom.Sphere(geom).CreateRadiusAttr(0.5 * size)
    elif shape == "cone":
        UsdGeom.Cone(geom).CreateHeightAttr(size)
        UsdGeom.Cone(geom).CreateRadiusAttr(0.5 * size)
    else:
        UsdGeom.Capsule(geom).CreateHeightAttr(size)
        UsdGeom.Capsule(geom).CreateRadiusAttr(0.5 * size)
    UsdPhysics.CollisionAPI.Apply(geom)
    UsdPhysics.MeshCollisionAPI.Apply(geom)
    if color is not None:
        rgb = np.asarray(color, dtype=np.float32).reshape(3)
        material_path = f"{prim_path}/Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(value) for value in rgb])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(geom).Bind(material)
    return prim


def _set_semantic_class(stage: Usd.Stage, prim: Usd.Prim, label: str) -> None:
    labels_api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    labels_api.CreateLabelsAttr().Set([label])


def _semantic_class_ids(packet: dict, class_name: str) -> list[int]:
    result = []
    for semantic_id, labels in packet.get("info", {}).get("idToLabels", {}).items():
        if str(labels.get("class", "")) == class_name:
            result.append(int(semantic_id))
    return result


def _target_visible_pixels(semantic_packets: list[dict]) -> np.ndarray:
    counts = []
    for packet in semantic_packets:
        target_ids = _semantic_class_ids(packet, "target_object")
        semantic = np.asarray(packet["data"], dtype=np.int32)
        counts.append(int(np.isin(semantic, target_ids).sum()) if target_ids else 0)
    return np.asarray(counts, dtype=np.int32)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_world_from_foot0(
    foot0_position_w: np.ndarray,
    foot0_quat_wxyz: np.ndarray,
    local_position: np.ndarray,
) -> np.ndarray:
    rotation = _quat_wxyz_to_matrix(foot0_quat_wxyz)
    return foot0_position_w + np.asarray(local_position, dtype=np.float64) @ rotation.T


def _target_point_mask(
    scene_points: np.ndarray,
    object_center_foot0: np.ndarray,
    object_size: float,
    padding_m: float = 0.012,
) -> np.ndarray:
    points = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    lower = np.asarray(object_center_foot0, dtype=np.float64) - object_size / 2.0 - padding_m
    upper = np.asarray(object_center_foot0, dtype=np.float64) + object_size / 2.0 + padding_m
    return np.all((points >= lower) & (points <= upper), axis=1)


def _fk_tcp_positions(fk_model: torch.nn.Module, qs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        positions, _ = fk_model(torch.as_tensor(qs, dtype=torch.float32, device="cuda:0"))
    return positions.cpu().numpy()


def _physical_grasp_anchors(
    fk_model: torch.nn.Module, qs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top-down contact configurations and calibrated pad-center poses."""
    with torch.no_grad():
        tcp_positions, tcp_quaternions = fk_model(
            torch.as_tensor(qs, dtype=torch.float32, device="cuda:0")
        )
    tcp_positions = tcp_positions.cpu().numpy()
    tcp_quaternions = tcp_quaternions.cpu().numpy()
    centers = []
    forwards = []
    for position, quaternion in zip(tcp_positions, tcp_quaternions):
        center, _ = tcp_pose_to_grasp_center_pose(
            position,
            quaternion,
            TCP_TO_GRASP_POSITION,
            TCP_TO_GRASP_QUATERNION,
        )
        forward = center - position
        forward /= np.linalg.norm(forward)
        centers.append(center)
        forwards.append(forward)
    centers = np.asarray(centers, dtype=np.float32)
    forwards = np.asarray(forwards, dtype=np.float32)
    # In this installed left-arm frame, top-down world motion is approximately +foot0 X.
    keep = forwards[:, 0] >= np.cos(np.deg2rad(30.0))
    return qs[keep], centers[keep], tcp_quaternions[keep], forwards[keep]


def _solve_offset_grasp_pose(
    fk_model: torch.nn.Module,
    grasp_center_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    seed_q: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    tcp_position, tcp_quaternion = grasp_center_pose_to_tcp_pose(
        grasp_center_position,
        grasp_quaternion,
        TCP_TO_GRASP_POSITION,
        TCP_TO_GRASP_QUATERNION,
    )
    local_rng = np.random.default_rng(seed)
    local_seeds = np.asarray(seed_q, dtype=np.float32).reshape(1, 6) + local_rng.normal(
        0.0, 0.12, size=(24, 6)
    ).astype(np.float32)
    local_seeds[0] = np.asarray(seed_q, dtype=np.float32).reshape(6)
    candidates = solve_pose_ik_candidates(
        fk_model,
        tcp_position,
        tcp_quaternion,
        n_candidates=8,
        restarts=24,
        iterations=200,
        self_clearance_fn=lambda q: float(
            self_clearance_batch(
                link_segments(fk_model, q, device="cuda:0"), capsule_radius_m=0.040
            ).min()
        ),
        clearance_threshold_m=0.01,
        seed_configurations=local_seeds,
        seed=seed,
        device="cuda:0",
    )
    if len(candidates) == 0:
        return None, np.asarray(tcp_position, dtype=np.float32)
    distances = np.linalg.norm(_wrap_angles(candidates - seed_q[None]), axis=1)
    return candidates[int(np.argmin(distances))], np.asarray(tcp_position, dtype=np.float32)


def _sample_layout_anchors(
    rng: np.random.Generator,
    anchor_tcp: np.ndarray,
    count: int,
    spacing: float,
) -> np.ndarray | None:
    """Pick ``count`` distinct anchor indices whose TCPs are at least ``spacing`` apart."""
    pool = list(range(len(anchor_tcp)))
    rng.shuffle(pool)
    chosen: list[int] = []
    for index in pool:
        if all(
            float(np.linalg.norm(anchor_tcp[index] - anchor_tcp[existing])) >= spacing
            for existing in chosen
        ):
            chosen.append(index)
        if len(chosen) >= count:
            return np.asarray(chosen, dtype=np.int64)
    return None


def _unproject_object_points(
    depth_views: np.ndarray,
    semantic_packets: list[dict],
    intrinsics: np.ndarray,
    camera_positions: np.ndarray,
    camera_quaternions_ros: np.ndarray,
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
    stride: int = 2,
) -> np.ndarray:
    """Unproject depth pixels whose semantic class is a graspable/target object.

    The fused planning cloud removes the active arm within a radius that also
    erases the target object below the pregrasp point; re-injecting the object
    surface points from the semantic masks guarantees the objects are always
    represented in the point cloud (v6.2.1 requirement: distinguish target vs
    obstacles).
    """
    from h1_1111_grasp.rgbd_geometry import depth_to_points_ros, transform_points

    object_ids: set[int] = set()
    for packet in semantic_packets:
        for semantic_id, labels in packet.get("info", {}).get("idToLabels", {}).items():
            class_name = str(labels.get("class", ""))
            if class_name == "target_object" or class_name.startswith("graspable_"):
                object_ids.add(int(semantic_id))
    if not object_ids:
        print(f"[DBG] no object class ids found in semantic packets", flush=True)
        return np.empty((0, 3), dtype=np.float64)
    clouds = []
    for view in range(depth_views.shape[0]):
        semantic = np.asarray(semantic_packets[view]["data"], dtype=np.int32)
        masked = np.asarray(depth_views[view], dtype=np.float64).copy()
        masked[~np.isin(semantic, sorted(object_ids))] = np.nan
        points, _ = depth_to_points_ros(masked, intrinsics[view], stride=stride)
        if len(points) == 0:
            continue
        clouds.append(
            transform_points(
                points,
                camera_positions[view],
                camera_quaternions_ros[view],
                base_position,
                base_quaternion,
            )
        )
    if not clouds:
        print(f"[DBG] object ids {sorted(object_ids)} but no unprojected points", flush=True)
        return np.empty((0, 3), dtype=np.float64)
    result = np.concatenate(clouds, axis=0)
    print(f"[DBG] object ids {sorted(object_ids)} unprojected points={len(result)}", flush=True)
    return result


def _sample_object_surface_points(
    shape: str,
    size: float,
    center_foot0: np.ndarray,
    count: int = 160,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Deterministic surface points for a known object layout (foot_0 frame).

    The objects are authored by us, so their geometry is exact: injecting these
    points guarantees the target/obstacle objects are always represented in the
    planning cloud (v6.2.1 requirement: the cloud must distinguish the target
    object from obstacles).  Sampling is seeded so the stored points are stable.
    """
    rng = rng or np.random.default_rng(0)
    center = np.asarray(center_foot0, dtype=np.float64).reshape(3)
    radius = 0.5 * float(size)
    height = float(size)
    if shape == "box":
        half = radius
        face = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=np.float64)
        axes = np.array([[0, 1], [0, 2], [1, 2], [1, 2], [0, 1], [0, 1]], dtype=np.int64)
        points = []
        for face_index in range(6):
            tangent = np.zeros(2)
            for local_count in range(count // 6):
                u, v = rng.uniform(-half, half, size=2)
                point = np.zeros(3)
                point[axes[face_index]] = [u, v]
                point[face_index // 2 if face_index % 2 == 0 else face_index // 2] += face[face_index][face_index // 2] * half
                points.append(point)
        points = np.asarray(points, dtype=np.float64)
    elif shape == "cylinder":
        z = rng.uniform(-height / 2.0, height / 2.0, size=count // 2)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=count // 2)
        top = rng.uniform(-radius, radius, size=count - count // 2)
        side = np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)
        caps = np.concatenate(
            (
                np.stack((top, np.zeros_like(top), np.full_like(top, -height / 2.0)), axis=-1),
                np.stack((top, np.zeros_like(top), np.full_like(top, height / 2.0)), axis=-1),
            ),
            axis=0,
        )
        points = np.concatenate((side, caps), axis=0)
    elif shape == "sphere":
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        points = direction * radius
    elif shape == "cone":
        z = rng.uniform(-height / 2.0, height / 2.0, size=count // 2)
        local_radius = radius * (0.5 - z / height)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=count // 2)
        side = np.stack(
            (local_radius * np.cos(angle), local_radius * np.sin(angle), z), axis=-1
        )
        top = rng.uniform(-radius, radius, size=count - count // 2)
        base = np.stack((top, np.zeros_like(top), np.full_like(top, -height / 2.0)), axis=-1)
        points = np.concatenate((side, base), axis=0)
    else:  # capsule
        z = rng.uniform(-height / 2.0, height / 2.0, size=count // 2)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=count // 2)
        side = np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)
        caps = []
        for sign in (-1.0, 1.0):
            direction = rng.normal(size=(count // 4, 3))
            direction[:, 2] = np.abs(direction[:, 2]) * sign
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            hemisphere = direction * radius
            hemisphere[:, 2] += sign * (height / 2.0 - radius)
            caps.append(hemisphere)
        points = np.concatenate((side, caps[0], caps[1]), axis=0)
    return points + center


def main() -> None:
    if args.count < 8:
        raise ValueError("count must be >= 8")
    rng = _seed_everything(args.seed + (0 if args.side == "left" else 1))
    stage_utils.open_stage(str(args.usd.resolve()))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=1))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))
    stage = omni.usd.get_context().get_stage()
    camera_specs = CAMERAS[args.side]
    local_camera_poses = _camera_local_poses(stage, camera_specs)
    intrinsics = np.stack(
        [_intrinsic_matrix(stage, path, args.resolution) for _, path, _ in camera_specs]
    )

    sim.reset()
    robot.reset()
    suffix = "_0" if args.side == "left" else ""
    arm_ids, arm_names = robot.find_joints(
        [f"^joint{i}{suffix}$" for i in range(1, 7)], preserve_order=True
    )
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

    base_position = robot.data.body_pos_w[0, base_ids[0]].cpu().numpy()
    base_quaternion = robot.data.body_quat_w[0, base_ids[0]].cpu().numpy()

    n_objects = (
        args.n_objects_max
        if args.physical_contact
        else int(rng.integers(args.n_objects_min, args.n_objects_max + 1))
    )
    # Sample pregrasp anchors inside the self-collision-free reachable shell, then
    # derive each object center directly below its anchor.  This guarantees the
    # grasp target itself is reachable (unreachable tasks are rejected upstream).
    pregrasp_anchors_foot0 = _sample_object_layout(
        rng,
        n_objects,
        args.reach_min_m,
        args.reach_max_m,
        args.z_min_m,
        args.z_max_m,
        args.min_object_spacing_m,
    )
    target_index = int(rng.integers(0, n_objects))
    object_shapes = (
        ["box"] * n_objects
        if args.physical_contact
        else [SHAPES[int(rng.integers(0, len(SHAPES)))] for _ in range(n_objects)]
    )
    object_sizes = rng.uniform(args.object_size_min, args.object_size_max, size=n_objects)
    initial_colors = rng.uniform(0.15, 0.90, size=(n_objects, 3))
    pregrasp_offsets = object_sizes / 2.0 + args.pregrasp_offset_m
    object_centers_foot0 = pregrasp_anchors_foot0 - np.stack(
        (
            np.zeros_like(object_sizes),
            np.zeros_like(object_sizes),
            pregrasp_offsets,
        ),
        axis=-1,
    )
    objects_root = f"/World/DynParts{args.side.capitalize()}"
    if stage.GetPrimAtPath(objects_root).IsValid():
        stage.RemovePrim(objects_root)
    for index in range(n_objects):
        center_foot0 = object_centers_foot0[index]
        center_world = _pose_world_from_foot0(base_position, base_quaternion, center_foot0)
        prim = _create_shape_prim(
            stage,
            objects_root,
            f"Obj{index}",
            object_shapes[index],
            float(object_sizes[index]),
            initial_colors[index],
        )
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(Gf.Vec3d(*[float(value) for value in center_world]))
        label = "target_object" if index == target_index else f"graspable_{object_shapes[index]}"
        _set_semantic_class(stage, prim, label)
    if args.physical_contact:
        # Rebuild Fabric/PhysX once after authoring the stable object pool.  Later
        # layouts only edit attributes and transforms; prim paths are never recycled.
        sim.reset()
        robot.reset()
    for _ in range(3):
        sim.step(render=True)
        robot.update(sim.get_physics_dt())

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

    fk_model = DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to("cuda:0")
    samples = []
    attempts = 0
    rejection_counts = {
        "start_self_collision": 0,
        "anchor_self_collision": 0,
        "start_environment_collision": 0,
        "grasp_config_environment_collision": 0,
        "grasp_duration_infeasible": 0,
        "grasp_experts_rejected": 0,
        "place_config_environment_collision": 0,
        "place_duration_infeasible": 0,
        "place_experts_rejected": 0,
        "object_too_close": 0,
        "target_not_visible": 0,
    }
    # Self-collision-free pools: start states and FK anchor configurations.  Each
    # anchor configuration is itself an exact position-only IK solution (its own
    # FK TCP), so per-point IK is never required during generation.
    start_pool, start_uniform_draws = _sample_self_collision_free_states(
        fk_model, rng, max(8, int(np.ceil(args.count)) * 2)
    )
    rejection_counts["start_self_collision"] = start_uniform_draws - len(start_pool)
    anchor_pool_q, anchor_uniform_draws = _sample_self_collision_free_states(
        fk_model, rng, args.anchor_pool_size
    )
    rejection_counts["anchor_self_collision"] = anchor_uniform_draws - len(anchor_pool_q)
    anchor_tcp = _fk_tcp_positions(fk_model, anchor_pool_q)
    anchor_grasp_centers = None
    anchor_grasp_quaternions = None
    anchor_grasp_forwards = None
    anchor_q_pregrasp = None
    anchor_q_lift = None
    anchor_pregrasp_tcp = None
    anchor_lift_tcp = None
    anchor_q_preplace = None
    anchor_preplace_tcp = None
    if args.physical_contact:
        if args.object_size_max > 0.05:
            raise ValueError("--physical_contact requires object_size_max <= 0.05 m")
        if args.physical_anchor_source_dir is not None:
            source_paths = sorted(args.physical_anchor_source_dir.glob("*_shard_*.npz"))
            if not source_paths:
                raise FileNotFoundError(
                    f"no physical anchor shards under {args.physical_anchor_source_dir}"
                )
            anchor_rows: list[dict[str, np.ndarray]] = []
            seen_contacts: set[tuple[float, ...]] = set()
            for source_path in source_paths:
                with np.load(source_path, allow_pickle=False) as source:
                    for row_index in range(len(source["q_contact"])):
                        variants = (
                            {
                                "q_contact": source["q_contact"][row_index],
                                "center": source["object_center_foot0"][row_index],
                                "quaternion": source["goal_quaternion_grasp"][row_index],
                                "q_pregrasp": source["q_pregrasp"][row_index],
                                "q_lift": source["q_lift"][row_index],
                                "q_preplace": source["q_pregrasp"][row_index],
                                "pregrasp_tcp": source["p_goal"][row_index],
                                "preplace_tcp": source["p_goal"][row_index],
                            },
                            {
                                "q_contact": source["q_place_contact"][row_index],
                                "center": source["place_center_foot0"][row_index],
                                "quaternion": source["goal_quaternion_place"][row_index],
                                "q_pregrasp": source["q_preplace"][row_index],
                                "q_lift": source["q_preplace"][row_index],
                                "q_preplace": source["q_preplace"][row_index],
                                "pregrasp_tcp": source["p_place"][row_index],
                                "preplace_tcp": source["p_place"][row_index],
                            },
                        )
                        for variant in variants:
                            key = tuple(np.round(variant["q_contact"], 5))
                            if key not in seen_contacts:
                                anchor_rows.append(variant)
                                seen_contacts.add(key)
            anchor_pool_q = np.asarray(
                [row["q_contact"] for row in anchor_rows], dtype=np.float32
            )
            anchor_grasp_centers = np.asarray(
                [row["center"] for row in anchor_rows], dtype=np.float32
            )
            anchor_grasp_quaternions = np.asarray(
                [row["quaternion"] for row in anchor_rows], dtype=np.float32
            )
            contact_tcp = _fk_tcp_positions(fk_model, anchor_pool_q)
            anchor_grasp_forwards = anchor_grasp_centers - contact_tcp
            anchor_grasp_forwards /= np.linalg.norm(
                anchor_grasp_forwards, axis=1, keepdims=True
            )
            anchor_q_pregrasp = np.asarray(
                [row["q_pregrasp"] for row in anchor_rows], dtype=np.float32
            )
            anchor_q_lift = np.asarray(
                [row["q_lift"] for row in anchor_rows], dtype=np.float32
            )
            anchor_q_preplace = np.asarray(
                [row["q_preplace"] for row in anchor_rows], dtype=np.float32
            )
            anchor_pregrasp_tcp = np.asarray(
                [row["pregrasp_tcp"] for row in anchor_rows], dtype=np.float32
            )
            anchor_preplace_tcp = np.asarray(
                [row["preplace_tcp"] for row in anchor_rows], dtype=np.float32
            )
            anchor_lift_tcp = _fk_tcp_positions(fk_model, anchor_q_lift)
            anchor_tcp = contact_tcp
            print(
                f"[PHYSICAL] reused_complete_contact_anchors={len(anchor_pool_q)} "
                f"source={args.physical_anchor_source_dir}",
                flush=True,
            )
        else:
            anchor_pool_q = np.vstack((NOMINAL_TOP_GRASP_Q[None], anchor_pool_q))
            (
                anchor_pool_q,
                anchor_grasp_centers,
                anchor_grasp_quaternions,
                anchor_grasp_forwards,
            ) = _physical_grasp_anchors(fk_model, anchor_pool_q)
            anchor_self_clearances = self_clearance_batch(
                link_segments(fk_model, anchor_pool_q, device="cuda:0"),
                capsule_radius_m=0.045,
            )
            anchor_order = np.argsort(-anchor_self_clearances)[: args.physical_anchor_limit]
            print(
                f"[PHYSICAL] top_down_candidates={len(anchor_pool_q)} "
                f"self_clearance_max={float(anchor_self_clearances.max()):.4f} "
                f"selected_min={float(anchor_self_clearances[anchor_order].min()):.4f}",
                flush=True,
            )
            anchor_pool_q = anchor_pool_q[anchor_order]
            anchor_grasp_centers = anchor_grasp_centers[anchor_order]
            anchor_grasp_quaternions = anchor_grasp_quaternions[anchor_order]
            anchor_grasp_forwards = anchor_grasp_forwards[anchor_order]
            valid_indices = []
            q_pregrasp_rows = []
            q_lift_rows = []
            pregrasp_tcp_rows = []
            lift_tcp_rows = []
            q_preplace_rows = []
            preplace_tcp_rows = []
            for anchor_index, (contact_q, center, quaternion, forward) in enumerate(
                zip(
                    anchor_pool_q,
                    anchor_grasp_centers,
                    anchor_grasp_quaternions,
                    anchor_grasp_forwards,
                )
            ):
                q_pregrasp, pregrasp_tcp = _solve_offset_grasp_pose(
                    fk_model,
                    center - forward * args.approach_distance_m,
                    quaternion,
                    contact_q,
                    args.seed + anchor_index * 2,
                )
                q_lift_row, lift_tcp = _solve_offset_grasp_pose(
                    fk_model,
                    center + WORLD_UP_FOOT0 * args.lift_distance_m,
                    quaternion,
                    contact_q,
                    args.seed + anchor_index * 2 + 1,
                )
                q_preplace_row, preplace_tcp = _solve_offset_grasp_pose(
                    fk_model,
                    center + WORLD_UP_FOOT0 * args.preplace_distance_m,
                    quaternion,
                    contact_q,
                    args.seed + 100000 + anchor_index,
                )
                if q_pregrasp is None or q_lift_row is None or q_preplace_row is None:
                    continue
                valid_indices.append(anchor_index)
                q_pregrasp_rows.append(q_pregrasp)
                q_lift_rows.append(q_lift_row)
                pregrasp_tcp_rows.append(pregrasp_tcp)
                lift_tcp_rows.append(lift_tcp)
                q_preplace_rows.append(q_preplace_row)
                preplace_tcp_rows.append(preplace_tcp)
            valid_indices = np.asarray(valid_indices, dtype=np.int64)
            anchor_pool_q = anchor_pool_q[valid_indices]
            anchor_grasp_centers = anchor_grasp_centers[valid_indices]
            anchor_grasp_quaternions = anchor_grasp_quaternions[valid_indices]
            anchor_grasp_forwards = anchor_grasp_forwards[valid_indices]
            anchor_q_pregrasp = np.asarray(q_pregrasp_rows, dtype=np.float32)
            anchor_q_lift = np.asarray(q_lift_rows, dtype=np.float32)
            anchor_pregrasp_tcp = np.asarray(pregrasp_tcp_rows, dtype=np.float32)
            anchor_lift_tcp = np.asarray(lift_tcp_rows, dtype=np.float32)
            anchor_q_preplace = np.asarray(q_preplace_rows, dtype=np.float32)
            anchor_preplace_tcp = np.asarray(preplace_tcp_rows, dtype=np.float32)
            anchor_tcp = _fk_tcp_positions(fk_model, anchor_pool_q)
            print(
                f"[PHYSICAL] complete_contact_anchors={len(anchor_pool_q)} "
                f"from_uniform_safe_pool={args.anchor_pool_size}",
                flush=True,
            )
        if len(anchor_pool_q) < 2:
            raise RuntimeError("physical mode needs at least two complete contact anchors")
    started = perf_counter()
    all_joint_pos = robot.data.default_joint_pos.clone()
    zero_joint_vel = torch.zeros_like(all_joint_pos)
    for layout_index in range(args.layouts):
        if len(samples) >= args.count:
            break
        n_objects = int(rng.integers(args.n_objects_min, args.n_objects_max + 1))
        if args.physical_contact:
            target_anchor_index = (
                int(args.physical_target_anchor_index) % len(anchor_pool_q)
                if args.physical_target_anchor_index >= 0
                else int(rng.integers(0, len(anchor_pool_q)))
            )
            target_center = anchor_grasp_centers[target_anchor_index]
            obstacle_centers = np.empty((0, 3), dtype=np.float32)
            if n_objects > 1:
                for _ in range(50):
                    obstacle_centers = _sample_object_layout(
                        rng,
                        n_objects - 1,
                        args.reach_min_m,
                        args.reach_max_m,
                        args.z_min_m,
                        args.z_max_m,
                        args.min_object_spacing_m,
                    ).astype(np.float32)
                    if np.all(
                        np.linalg.norm(obstacle_centers - target_center[None], axis=1)
                        >= args.min_object_spacing_m
                    ):
                        break
                else:
                    print("[OBJ] failed to place physical distractors", flush=True)
                    continue
            object_centers_foot0 = np.vstack((target_center[None], obstacle_centers))
            anchor_indices = np.full(n_objects, target_anchor_index, dtype=np.int64)
        else:
            anchor_indices = _sample_layout_anchors(
                rng, anchor_tcp, n_objects, args.min_object_spacing_m
            )
            if anchor_indices is None:
                print("[OBJ] failed to pick spaced anchors, skipping layout", flush=True)
                continue
        object_shapes = (
            ["box"] * n_objects
            if args.physical_contact
            else [SHAPES[int(rng.integers(0, len(SHAPES)))] for _ in range(n_objects)]
        )
        object_sizes = rng.uniform(args.object_size_min, args.object_size_max, size=n_objects)
        # Appearance is sampled independently of target identity.  The target
        # position, not a privileged target color, is the association cue.
        object_colors = rng.uniform(0.15, 0.90, size=(n_objects, 3)).astype(np.float32)
        if not args.physical_contact:
            pregrasp_offsets = object_sizes / 2.0 + args.pregrasp_offset_m
            object_centers_foot0 = anchor_tcp[anchor_indices] - np.stack(
                (
                    np.zeros_like(object_sizes),
                    np.zeros_like(object_sizes),
                    pregrasp_offsets,
                ),
                axis=-1,
            )
        objects_root = f"/World/DynParts{args.side.capitalize()}"
        if not args.physical_contact and stage.GetPrimAtPath(objects_root).IsValid():
            stage.RemovePrim(objects_root)
        for index in range(n_objects):
            center_foot0 = object_centers_foot0[index]
            center_world = _pose_world_from_foot0(base_position, base_quaternion, center_foot0)
            if args.physical_contact:
                prim = stage.GetPrimAtPath(f"{objects_root}/Obj{index}")
                UsdGeom.Cube(stage.GetPrimAtPath(f"{objects_root}/Obj{index}/Geometry")).GetSizeAttr().Set(
                    float(object_sizes[index])
                )
                shader = UsdShade.Shader(
                    stage.GetPrimAtPath(f"{objects_root}/Obj{index}/Material/Shader")
                )
                shader.GetInput("diffuseColor").Set(
                    Gf.Vec3f(*[float(value) for value in object_colors[index]])
                )
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                prim = _create_shape_prim(
                    stage,
                    objects_root,
                    f"Obj{index}",
                    object_shapes[index],
                    float(object_sizes[index]),
                    object_colors[index],
                )
            xform = UsdGeom.XformCommonAPI(prim)
            xform.SetTranslate(Gf.Vec3d(*[float(value) for value in center_world]))
            _set_semantic_class(stage, prim, f"graspable_{object_shapes[index]}")
        if args.physical_contact:
            for index in range(n_objects, args.n_objects_max):
                UsdGeom.Imageable(
                    stage.GetPrimAtPath(f"{objects_root}/Obj{index}")
                ).MakeInvisible()
        # Layout-level target pre-selection: pick the anchor whose own grasp
        # configuration clears every object (offline FK + injected surface
        # points).  Without this, a bad random target anchor can block a whole
        # layout at grasp_config_environment_collision.
        precheck_rng = np.random.default_rng(int(args.seed) + layout_index * 7919)
        object_points_all = np.concatenate(
            [
                _sample_object_surface_points(
                    object_shapes[index],
                    float(object_sizes[index]),
                    object_centers_foot0[index],
                    count=140,
                    rng=precheck_rng,
                )
                for index in range(n_objects)
            ],
            axis=0,
        )
        grasp_clearances = []
        physical_waypoints = []
        for index in range(n_objects):
            contact_q = anchor_pool_q[anchor_indices[index]]
            if args.physical_contact:
                if index != 0:
                    grasp_clearances.append(float("-inf"))
                    physical_waypoints.append(None)
                    continue
                grasp_quaternion_i = anchor_grasp_quaternions[anchor_indices[index]]
                q_pregrasp_i = anchor_q_pregrasp[anchor_indices[index]]
                pregrasp_tcp_i = anchor_pregrasp_tcp[anchor_indices[index]]
                q_lift_i = anchor_q_lift[anchor_indices[index]]
                lift_tcp_i = anchor_lift_tcp[anchor_indices[index]]
                grasp_segments = link_segments(
                    fk_model, q_pregrasp_i[None], device="cuda:0"
                )[0]
                physical_waypoints.append(
                    {
                        "q_pregrasp": q_pregrasp_i,
                        "q_contact": contact_q,
                        "q_lift": q_lift_i,
                        "pregrasp_tcp": pregrasp_tcp_i,
                        "lift_tcp": lift_tcp_i,
                        "quaternion": grasp_quaternion_i,
                    }
                )
            else:
                grasp_segments = link_segments(
                    fk_model, contact_q[None], device="cuda:0"
                )[0]
                physical_waypoints.append(None)
            grasp_clearances.append(
                float(
                    environment_clearance(
                        grasp_segments[1:], object_points_all, capsule_radius_m=0.05
                    )
                )
            )
        if args.physical_contact:
            target_index = 0
        else:
            target_candidates = [
                index for index, clearance in enumerate(grasp_clearances) if clearance >= 0.02
            ]
            if not target_candidates:
                target_candidates = [int(np.argmax(grasp_clearances))]
            target_index = int(rng.choice(target_candidates))
        target_prim = stage.GetPrimAtPath(f"{objects_root}/Obj{target_index}")
        if target_prim.IsValid():
            _set_semantic_class(stage, target_prim, "target_object")
        # Flush USD attribute edits into Fabric before reading RTX annotators.
        simulation_app.update()
        simulation_app.update()
        for _ in range(5):
            sim.step(render=True)
            robot.update(sim.get_physics_dt())

        target_center_foot0 = object_centers_foot0[target_index]
        target_size = float(object_sizes[target_index])
        target_shape = object_shapes[target_index]
        if args.physical_contact:
            selected_waypoints = physical_waypoints[target_index]
            if selected_waypoints is None:
                print("[PHYSICAL] selected target has no safe contact waypoints", flush=True)
                continue
            pregrasp_foot0 = selected_waypoints["pregrasp_tcp"].copy()
            q_grasp_ik = selected_waypoints["q_pregrasp"].copy()
            q_contact = selected_waypoints["q_contact"].copy()
            q_lift = selected_waypoints["q_lift"].copy()
            grasp_goal_quaternion = selected_waypoints["quaternion"].copy()
        else:
            pregrasp_foot0 = anchor_tcp[anchor_indices[target_index]].copy()
            q_grasp_ik = anchor_pool_q[anchor_indices[target_index]]
            q_contact = q_grasp_ik.copy()
            q_lift = q_grasp_ik.copy()
            with torch.no_grad():
                _, grasp_quaternion_t = fk_model(
                    torch.as_tensor(q_grasp_ik[None], dtype=torch.float32, device="cuda:0")
                )
            grasp_goal_quaternion = grasp_quaternion_t[0].cpu().numpy()
        print(
            f"[OBJ] layout={layout_index} side={args.side} n_objects={n_objects} "
            f"target_index={target_index} shape={target_shape} size={target_size:.3f} "
            f"grasp_clearances={[round(value, 3) for value in grasp_clearances]} "
            f"pregrasp={np.round(pregrasp_foot0, 3).tolist()}",
            flush=True,
        )

        for start_attempt in range(args.max_attempts_per_layout):
            if len(samples) >= args.count:
                break
            q_start = start_pool[start_attempt % len(start_pool)]
            attempts += 1
            all_joint_pos[:, arm_ids] = torch.as_tensor(
                q_start, dtype=all_joint_pos.dtype, device=robot.device
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
            target_visible_pixels = _target_visible_pixels(semantic_packets)
            if args.debug:
                from PIL import Image

                debug_dir = PROJECT_ROOT / "outputs" / "physical_camera_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                if start_attempt == 0:
                    for view_index, (view_name, _, _) in enumerate(camera_specs):
                        Image.fromarray(rgb_views[view_index]).save(
                            debug_dir / f"layout_{layout_index:03d}_{view_name}.png"
                        )
                semantic_labels = [
                    packet.get("info", {}).get("idToLabels", {})
                    for packet in semantic_packets
                ]
                print(
                    f"[VIS] target_pixels={target_visible_pixels.tolist()} "
                    f"total={int(target_visible_pixels.sum())} labels={semantic_labels}",
                    flush=True,
                )
            if int(target_visible_pixels.sum()) < args.minimum_target_pixels_total:
                rejection_counts["target_not_visible"] += 1
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
            planning_depth_views = _planning_depth_without_active_arm(
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
            planning_cloud = remove_current_arm_points(
                cloud, current_segments, removal_radius_m=0.075
            )
            object_points_foot0 = np.concatenate(
                [
                    _sample_object_surface_points(
                        object_shapes[object_index],
                        float(object_sizes[object_index]),
                        object_centers_foot0[object_index],
                        count=140,
                        rng=rng,
                    )
                    for object_index in range(n_objects)
                ],
                axis=0,
            )
            planning_cloud = np.concatenate(
                (planning_cloud, object_points_foot0), axis=0
            )
            # Crop the fused cloud to the arm-reachable vicinity (foot_0 frame).
            # The room floor/ceiling/walls are far outside the arm workspace and
            # otherwise dominate the top-N component slots, hiding the target
            # object from the component tokenizer.
            planning_cloud = planning_cloud[
                (planning_cloud[:, 2] >= -0.35)
                & (planning_cloud[:, 2] <= 0.9)
                & (np.linalg.norm(planning_cloud[:, :2], axis=1) <= 0.85)
            ]
            current_environment = environment_clearance(
                current_segments[1:], planning_cloud, capsule_radius_m=0.05
            )
            if current_environment < 0.01:
                rejection_counts["start_environment_collision"] += 1
                continue

            grasp_segments = link_segments(fk_model, q_grasp_ik[None], device="cuda:0")[0]
            grasp_config_env = environment_clearance(
                grasp_segments[1:], planning_cloud, capsule_radius_m=0.05
            )
            if grasp_config_env < 0.01:
                rejection_counts["grasp_config_environment_collision"] += 1
                continue
            grasp_duration, grasp_feasible, _ = minimum_jerk_duration(
                q_start,
                q_start + _wrap_angles(q_grasp_ik - q_start),
                VELOCITY_LIMITS,
                ACCELERATION_LIMITS,
                JERK_LIMITS,
                min_duration=1.0,
                max_duration=5.0,
            )
            if not grasp_feasible:
                rejection_counts["grasp_duration_infeasible"] += 1
                continue

            q_place_ik = None
            q_place_contact = None
            place_center_foot0 = None
            preplace_tcp_foot0 = None
            place_goal_quaternion = None
            place_duration = 0.0
            place_start_q = q_lift if args.physical_contact else q_grasp_ik
            transfer_cloud = planning_cloud
            if args.physical_contact:
                transfer_cloud = planning_cloud[
                    np.linalg.norm(planning_cloud - target_center_foot0[None], axis=1)
                    > max(0.06, target_size)
                ]
            for _ in range(200):
                candidate = int(rng.integers(0, len(anchor_pool_q)))
                if candidate == int(anchor_indices[target_index]):
                    continue
                candidate_place_center = (
                    anchor_grasp_centers[candidate]
                    if args.physical_contact
                    else anchor_tcp[candidate]
                )
                if (
                    np.linalg.norm(candidate_place_center - target_center_foot0)
                    < args.min_place_distance_m
                ):
                    rejection_counts["object_too_close"] += 1
                    continue
                if args.physical_contact:
                    candidate_quaternion = anchor_grasp_quaternions[candidate]
                    candidate_q = anchor_q_preplace[candidate]
                    candidate_tcp = anchor_preplace_tcp[candidate]
                    candidate_contact_q = anchor_pool_q[candidate]
                else:
                    candidate_q = anchor_pool_q[candidate]
                    candidate_tcp = anchor_tcp[candidate]
                    candidate_contact_q = candidate_q
                    candidate_quaternion = None
                place_segments = link_segments(
                    fk_model, candidate_q[None], device="cuda:0"
                )[0]
                place_env = environment_clearance(
                    place_segments[1:], transfer_cloud, capsule_radius_m=0.05
                )
                if place_env < 0.01:
                    rejection_counts["place_config_environment_collision"] += 1
                    continue
                candidate_duration, candidate_feasible, _ = minimum_jerk_duration(
                    place_start_q,
                    place_start_q + _wrap_angles(candidate_q - place_start_q),
                    VELOCITY_LIMITS,
                    ACCELERATION_LIMITS,
                    JERK_LIMITS,
                    min_duration=1.0,
                    max_duration=5.0,
                )
                if not candidate_feasible:
                    rejection_counts["place_duration_infeasible"] += 1
                    continue
                q_place_ik = candidate_q
                q_place_contact = candidate_contact_q
                place_center_foot0 = candidate_place_center
                preplace_tcp_foot0 = candidate_tcp
                place_goal_quaternion = candidate_quaternion
                place_duration = candidate_duration
                break
            if q_place_ik is None or place_center_foot0 is None:
                continue

            grasp_paths = _candidate_paths(
                q_start, q_start + _wrap_angles(q_grasp_ik - q_start), args.expert_candidates, rng
            )
            grasp_index, grasp_env, grasp_self = _select_best_candidate(
                fk_model,
                grasp_paths,
                planning_cloud,
                self_capsule_radius_m=0.040 if args.physical_contact else 0.045,
            )
            if grasp_index is None:
                rejection_counts["grasp_experts_rejected"] += 1
                continue
            place_paths = _candidate_paths(
                place_start_q,
                place_start_q + _wrap_angles(q_place_ik - place_start_q),
                args.expert_candidates,
                rng,
            )
            place_index, place_env, place_self = _select_best_candidate(
                fk_model,
                place_paths,
                transfer_cloud,
                self_capsule_radius_m=0.040 if args.physical_contact else 0.045,
            )
            if place_index is None:
                rejection_counts["place_experts_rejected"] += 1
                continue

            grasp_trajectory = grasp_paths[grasp_index]
            grasp_unwrapped = np.unwrap(grasp_trajectory, axis=0, period=2.0 * np.pi)
            grasp_unwrapped += (q_start - grasp_unwrapped[0])[None]
            place_trajectory = place_paths[place_index]
            place_unwrapped = np.unwrap(place_trajectory, axis=0, period=2.0 * np.pi)
            place_unwrapped += (place_start_q - place_unwrapped[0])[None]

            grasp_ranked = _rank_expert_candidates(
                fk_model,
                grasp_paths,
                planning_cloud,
                self_capsule_radius_m=0.040 if args.physical_contact else 0.045,
            )
            place_ranked = _rank_expert_candidates(
                fk_model,
                place_paths,
                transfer_cloud,
                self_capsule_radius_m=0.040 if args.physical_contact else 0.045,
            )
            grasp_experts = [_wrap_angles(grasp_paths[i]) for i in grasp_ranked[: args.expert_labels]]
            place_experts = [_wrap_angles(place_paths[i]) for i in place_ranked[: args.expert_labels]]
            while len(grasp_experts) < args.expert_labels:
                grasp_experts.append(grasp_trajectory)
            while len(place_experts) < args.expert_labels:
                place_experts.append(place_trajectory)

            stored_scene_points, stored_scene_point_mask = _fixed_scene_points(
                planning_cloud,
                args.stored_scene_points,
                priority_points=object_points_foot0,
            )
            target_mask = _target_point_mask(stored_scene_points, target_center_foot0, target_size)
            if args.debug:
                print(
                    f"[DBG] obj_pts={len(object_points_foot0)} cloud={len(planning_cloud)} "
                    f"mask_frac={target_mask.mean():.4f}",
                    flush=True,
                )
            tokens, mask, component_rows = point_cloud_component_tokens(
                planning_cloud, max_components=8, return_point_indices=True
            )
            # Target label from the injected target-object surface points (known
            # geometry, always inside the target AABB).  Component merging with
            # nearby structure only dilutes all-points statistics; counting the
            # injected target rows is robust to such merging.
            injected_objects_per_row = 140
            base_cloud_count = len(planning_cloud) - len(object_points_foot0)
            target_injected_rows = np.arange(
                base_cloud_count + target_index * injected_objects_per_row,
                base_cloud_count + (target_index + 1) * injected_objects_per_row,
            )
            injected_counts = np.zeros(tokens.shape[0], dtype=np.int64)
            for component_index in range(tokens.shape[0]):
                if not mask[component_index]:
                    continue
                injected_counts[component_index] = int(
                    np.isin(target_injected_rows, component_rows[component_index]).sum()
                )
            target_component_score = np.zeros(tokens.shape[0], dtype=np.float32)
            best_component = int(np.argmax(injected_counts))
            if injected_counts[best_component] > 0:
                target_component_score[best_component] = 1.0
            target_component_label = target_component_score >= 0.5
            condition_grasp = np.concatenate(
                (q_start, q_grasp_ik, pregrasp_foot0.astype(np.float32))
            ).astype(np.float32)
            place_goal_position = (
                preplace_tcp_foot0 if args.physical_contact else place_center_foot0
            )
            condition_place = np.concatenate(
                (place_start_q, q_place_ik, place_goal_position.astype(np.float32))
            ).astype(np.float32)
            samples.append(
                {
                    "sample_id": np.asarray(f"{args.side}_{len(samples):06d}"),
                    "side": np.asarray(args.side),
                    "layout_index": np.int32(layout_index),
                    "condition_grasp": condition_grasp,
                    "condition_place": condition_place,
                    "p_goal": pregrasp_foot0.astype(np.float32),
                    "p_place": place_goal_position.astype(np.float32),
                    "q_start": q_start.astype(np.float32),
                    "q_hold": place_start_q.astype(np.float32),
                    "q_pregrasp": q_grasp_ik.astype(np.float32),
                    "q_contact": q_contact.astype(np.float32),
                    "q_lift": q_lift.astype(np.float32),
                    "q_preplace": q_place_ik.astype(np.float32),
                    "q_place_contact": q_place_contact.astype(np.float32),
                    "q_grasp": grasp_trajectory.astype(np.float32),
                    "q_grasp_unwrapped": grasp_unwrapped.astype(np.float32),
                    "q_place": place_trajectory.astype(np.float32),
                    "q_place_unwrapped": place_unwrapped.astype(np.float32),
                    "q_grasp_experts": np.stack(grasp_experts[: args.expert_labels]).astype(np.float32),
                    "q_place_experts": np.stack(place_experts[: args.expert_labels]).astype(np.float32),
                    "q_ik_candidates_grasp": q_grasp_ik[None].astype(np.float32),
                    "q_ik_candidates_place": q_place_ik[None].astype(np.float32),
                    "goal_quaternion_grasp": grasp_goal_quaternion.astype(np.float32),
                    "goal_quaternion_place": (
                        place_goal_quaternion.astype(np.float32)
                        if args.physical_contact
                        else grasp_goal_quaternion.astype(np.float32)
                    ),
                    "duration_grasp": np.float32(grasp_duration),
                    "duration_place": np.float32(place_duration),
                    "object_type": np.int32(SHAPES.index(target_shape)),
                    "object_size": np.float32([target_size] * 3),
                    "object_center_foot0": target_center_foot0.astype(np.float32),
                    "place_center_foot0": place_center_foot0.astype(np.float32),
                    "physical_contact_labels": np.bool_(args.physical_contact),
                    "n_objects": np.int32(n_objects),
                    "obstacle_tokens": tokens,
                    "obstacle_mask": mask,
                    "target_component_label": target_component_label,
                    "target_component_score": target_component_score,
                    "scene_points": stored_scene_points,
                    "scene_point_mask": stored_scene_point_mask,
                    "target_point_mask": target_mask,
                    "target_visible_pixels_per_view": target_visible_pixels,
                    "grasp_environment_clearance_m": np.float32(grasp_env),
                    "grasp_self_clearance_m": np.float32(grasp_self),
                    "place_environment_clearance_m": np.float32(place_env),
                    "place_self_clearance_m": np.float32(place_self),
                    "rgb": np.repeat(rgb_views[:, None], 2, axis=1),
                    "depth": np.repeat(depth_views[:, None, ..., None], 2, axis=1).astype(np.float16),
                    "semantic": np.repeat(
                        semantic_views[:, None, ..., None], 2, axis=1
                    ).astype(np.int32),
                    "collision_free": np.bool_(True),
                }
            )
            print(
                f"[DATA] layout={layout_index} accepted={len(samples)}/{args.count} attempts={attempts}",
                flush=True,
            )
        if attempts:
            print(
                f"[REJ] layout={layout_index} accepted={len(samples)}/{args.count} "
                f"rejections={json.dumps(rejection_counts, sort_keys=True)}",
                flush=True,
            )

    if len(samples) < args.count:
        raise RuntimeError(
            f"generated only {len(samples)}/{args.count} samples; rejections={rejection_counts}"
        )
    indices = rng.permutation(len(samples))
    train_end = max(1, int(0.8 * len(samples)))
    validation_end = max(train_end + 1, int(0.9 * len(samples)))
    output_dir = args.output_dir or PROJECT_ROOT / "data" / "visual_v621"
    side_dir = output_dir / args.side
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
        "condition": "grasp: q_start[6]+q_pregrasp[6]+p_tcp_pregrasp[3]; place: q_lift[6]+q_preplace[6]+p_tcp_preplace[3]" if args.physical_contact else "grasp: q_start[6]+q_grasp_ik[6]+p_goal[3]; place: q_hold[6]+q_place_ik[6]+p_place[3]",
        "condition_version": "v6.2.2-physical-contact" if args.physical_contact else "v6.2.1",
        "physical_contact": bool(args.physical_contact),
        "minimum_target_pixels_total": int(args.minimum_target_pixels_total),
        "tcp_to_grasp_center_position_m": TCP_TO_GRASP_POSITION.tolist(),
        "trajectory_shape": [64, 6],
        "n_objects": n_objects,
        "target_index": target_index,
        "target_shape": target_shape,
        "target_size": target_size,
        "pregrasp_foot0": pregrasp_foot0.tolist(),
        "rejection_counts": rejection_counts,
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
