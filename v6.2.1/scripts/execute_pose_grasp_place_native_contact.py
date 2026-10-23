"""Audit a soft native-contact grasp without creating an attachment joint.

The task cube is initialized once on a pickup pedestal.  After task start it is
never teleported, pinned, or gravity-disabled.  The H1 floating base is pinned
only to isolate auxiliary-arm trajectory execution from balance control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
parser.add_argument(
    "--plan", type=Path,
    default=PROJECT_ROOT / "outputs/pose_grasp_place_plan_true_fingertips_lift.npz",
)
parser.add_argument(
    "--usd", type=Path,
    default=PROJECT_ROOT / "1111_copy_native_contact.usd",
    help="Scene USD containing H1, auxiliary arms, grippers, and cameras.",
)
parser.add_argument(
    "--output", type=Path,
    default=PROJECT_ROOT / "outputs/native_contact/native_contact_lift.json",
)
parser.add_argument("--cube_size", type=float, default=0.04)
parser.add_argument("--cube_mass", type=float, default=0.08)
parser.add_argument("--cube_static_friction", type=float, default=2.0)
parser.add_argument("--cube_dynamic_friction", type=float, default=1.8)
parser.add_argument("--cube_angular_damping", type=float, default=2.0)
parser.add_argument(
    "--primitive_box_colliders",
    action="store_true",
    help="Use native UsdGeom.Cube colliders for the isolated task cube and pedestals.",
)
parser.add_argument("--close_duration", type=float, default=6.0)
parser.add_argument("--preclose_hold_duration", type=float, default=1.0)
parser.add_argument("--contact_hold_duration", type=float, default=1.5)
parser.add_argument(
    "--approach_stop_fraction",
    type=float,
    default=0.85,
    help="Fraction of open-loop approach completed before geometric centering.",
)
parser.add_argument(
    "--preclose_joint_correction",
    type=float,
    nargs=6,
    default=(0.00298833, 0.00642053, 0.00242572, 0.01799011, -0.01248824, 0.01322534),
    metavar=("DQ1", "DQ2", "DQ3", "DQ4", "DQ5", "DQ6"),
    help="Contact-free calibration correction applied after the guarded approach.",
)
parser.add_argument("--preclose_correction_duration", type=float, default=2.0)
parser.add_argument(
    "--preclose_mode",
    choices=("online", "free_space_waypoints"),
    default="online",
)
parser.add_argument("--preload_gap", type=float, default=0.00075)
parser.add_argument("--gap_servo_tolerance", type=float, default=0.00015)
parser.add_argument("--release_duration", type=float, default=2.0)
parser.add_argument("--settle_duration", type=float, default=1.0)
parser.add_argument("--maximum_pre_task_support_drift", type=float, default=0.003)
parser.add_argument("--place_hold_duration", type=float, default=4.0)
parser.add_argument("--contact_impulse_threshold", type=float, default=1.0e-4)
parser.add_argument("--root_z_offset", type=float, default=0.0)
parser.add_argument("--physics_dt", type=float, default=0.005)
parser.add_argument("--motion_time_scale", type=float, default=3.0)
parser.add_argument("--lift_time_scale", type=float, default=5.0)
parser.add_argument("--pad_contact_offset", type=float, default=0.001)
parser.add_argument("--compliant_contact_stiffness", type=float, default=800.0)
parser.add_argument("--compliant_contact_damping", type=float, default=50.0)
parser.add_argument("--gripper_effort_limit", type=float, default=10.0)
parser.add_argument("--gripper_velocity_limit", type=float, default=0.3)
parser.add_argument("--gripper_stiffness", type=float, default=20.0)
parser.add_argument("--gripper_damping", type=float, default=10.0)
parser.add_argument("--grasp_finger_target", type=float, default=None)
parser.add_argument("--auto_closing_midpoint_drift", action="store_true")
parser.add_argument("--skip_gap_servo", action="store_true")
parser.add_argument("--maximum_cube_linear_speed", type=float, default=0.8)
parser.add_argument("--maximum_cube_angular_speed", type=float, default=8.0)
parser.add_argument(
    "--maximum_pregrasp_object_displacement",
    type=float,
    default=0.015,
    help="Emergency-stop distance for object motion before finger closing starts.",
)
parser.add_argument("--maximum_target_contact_impulse", type=float, default=0.08)
parser.add_argument("--minimum_bilateral_contact_force", type=float, default=0.02)
parser.add_argument("--contact_sensor_decimation", type=int, default=4)
parser.add_argument("--maximum_contact_center_error", type=float, default=0.008)
parser.add_argument("--maximum_preclose_lateral_error", type=float, default=0.015)
parser.add_argument(
    "--closing_midpoint_drift",
    type=float,
    nargs=3,
    default=(-0.00195, 0.00692, -0.01140),
    metavar=("DX", "DY", "DZ"),
    help="Measured open-to-contact pad-midpoint drift in world coordinates.",
)
parser.add_argument(
    "--pickup_center_offset",
    type=float,
    nargs=3,
    default=(-0.0018, -0.0022, 0.0009),
    metavar=("DX", "DY", "DZ"),
    help="Measured world-frame correction from the nominal grasp center to the pad midpoint.",
)
parser.add_argument(
    "--place_center_offset",
    type=float,
    nargs=3,
    default=(-0.00255, 0.00091, 0.00048),
    metavar=("DX", "DY", "DZ"),
    help="Measured world-frame correction from the nominal place center to the pad midpoint.",
)
parser.add_argument(
    "--grasp_point_offset_object",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="Desired pad midpoint relative to the object center, expressed in the object frame.",
)
parser.add_argument("--calibration_only", action="store_true")
parser.add_argument("--reference_only", action="store_true")
parser.add_argument("--stop_after_lift", action="store_true")
parser.add_argument("--camera_preview_only", action="store_true")
parser.add_argument(
    "--grasp_mode",
    choices=("contact", "attachment_constraint"),
    default="contact",
    help="Native contact audit or an explicitly labelled high-level grasp constraint.",
)
parser.add_argument("--record_dir", type=Path, default=None)
parser.add_argument("--record_fps", type=float, default=10.0)
parser.add_argument("--record_width", type=int, default=640)
parser.add_argument("--record_height", type=int, default=368)
parser.add_argument(
    "--timeseries_output",
    type=Path,
    default=None,
    help="Optional compressed NPZ with full task dynamics/contact timeseries.",
)
parser.add_argument("--camera_track_object", action="store_true")
parser.add_argument(
    "--camera_eye", type=float, nargs=3, default=(-0.58, -0.45, 1.02),
    metavar=("X", "Y", "Z"),
)
parser.add_argument(
    "--camera_target", type=float, nargs=3, default=(-0.259, 0.049, 0.749),
    metavar=("X", "Y", "Z"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_dir is not None:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim.utils import stage as stage_utils
import omni.usd
from omni.physx import get_physx_simulation_interface
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from h1_1111_grasp.env_cfg import H1_1111_CFG
from src.robot.torch_kinematics import DifferentiableZBotFK


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        nxt = (index + 1) % 3
        last = (index + 2) % 3
        scale = np.sqrt(1.0 + matrix[index, index] - matrix[nxt, nxt] - matrix[last, last]) * 2.0
        quat = np.zeros(4)
        quat[0] = (matrix[last, nxt] - matrix[nxt, last]) / scale
        quat[index + 1] = 0.25 * scale
        quat[nxt + 1] = (matrix[nxt, index] + matrix[index, nxt]) / scale
        quat[last + 1] = (matrix[last, index] + matrix[index, last]) / scale
    return quat / np.linalg.norm(quat)


def _look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.stack((right, camera_up, -forward), axis=1)
    return _matrix_quat(rotation)


def _align_z_to(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cosine = float(np.dot(z_axis, direction))
    if np.linalg.norm(cross) < 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if cosine > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    quat = np.concatenate(([1.0 + cosine], cross))
    return quat / np.linalg.norm(quat)


# Centers of the actual inward-facing white fingertip meshes, measured in each
# inner-finger rigid body's local frame. The right visual mesh is mirrored and
# therefore uses positive local Y; treating both sides as negative Y shifts the
# inferred grasp center outside the jaws.
LEFT_PAD_FACE = (
    np.array([0.0, -0.0435484238, 0.1294200905]),
    np.array([0.0, 1.0, 0.0]),
)
RIGHT_PAD_FACE = (
    np.array([0.0, 0.0435484238, 0.1294200905]),
    np.array([0.0, -1.0, 0.0]),
)


def _add_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/TaskGrip")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(args.cube_static_friction))
    physics.CreateDynamicFrictionAttr(float(args.cube_dynamic_friction))
    physics.CreateRestitutionAttr(0.0)
    return material


def _add_box(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    position: np.ndarray,
    material: UsdShade.Material,
    dynamic: bool,
) -> Usd.Prim:
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))
    if args.primitive_box_colliders:
        geometry = UsdGeom.Cube.Define(stage, f"{path}/Geometry")
        geometry.CreateSizeAttr(1.0)
        geometry.AddScaleOp().Set(Gf.Vec3d(*[float(value) for value in size]))
    else:
        geometry = UsdGeom.Mesh.Define(stage, f"{path}/Geometry")
        hx, hy, hz = (0.5 * float(value) for value in size)
        geometry.GetPointsAttr().Set(
            [
                Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, -hy, -hz),
                Gf.Vec3f(hx, hy, -hz), Gf.Vec3f(-hx, hy, -hz),
                Gf.Vec3f(-hx, -hy, hz), Gf.Vec3f(hx, -hy, hz),
                Gf.Vec3f(hx, hy, hz), Gf.Vec3f(-hx, hy, hz),
            ]
        )
        geometry.GetFaceVertexCountsAttr().Set([3] * 12)
        geometry.GetFaceVertexIndicesAttr().Set(
            [
                0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
                0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
                1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
            ]
        )
    geometry.GetDoubleSidedAttr().Set(True)
    collision = UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    if not args.primitive_box_colliders:
        UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(
        material, materialPurpose="physics"
    )
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(args.cube_mass))
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        rigid_api.CreateAngularDampingAttr().Set(float(args.cube_angular_damping))
        rigid_api.CreateSolverPositionIterationCountAttr().Set(16)
        rigid_api.CreateSolverVelocityIterationCountAttr().Set(4)
    return root.GetPrim()


def main() -> None:
    if args.grasp_mode != "contact":
        raise ValueError(
            "this isolated executor only supports native contact; use "
            "execute_pose_grasp_place_physx.py for the frozen attachment baseline"
        )
    with np.load(args.plan, allow_pickle=False) as plan_data:
        plan = {key: np.array(plan_data[key], copy=True) for key in plan_data.files}
    grasp_point_offset_object = np.asarray(
        args.grasp_point_offset_object, dtype=np.float64
    )
    if np.any(np.abs(grasp_point_offset_object) > 0.5 * float(args.cube_size)):
        raise ValueError(
            "grasp_point_offset_object must remain inside the object: "
            f"offset={grasp_point_offset_object.tolist()}, cube_size={args.cube_size}"
        )

    scene_usd = args.usd.resolve()
    if not scene_usd.is_file():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    stage_utils.open_stage(str(scene_usd))
    stage = omni.usd.get_context().get_stage()
    foot0_static = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/UnitreeH1/ZBot6S_left/foot_0")
    ).ComputeLocalToWorldTransform(0)

    def static_from_foot0(point: np.ndarray) -> np.ndarray:
        return np.asarray(foot0_static.Transform(Gf.Vec3d(*[float(v) for v in point])))

    pickup_center = static_from_foot0(plan["grasp_center_position_foot0"])
    place_center = static_from_foot0(plan["place_center_position_foot0"])
    material = _add_material(stage)
    for task_path in ("/World/GraspTask",):
        if stage.GetPrimAtPath(task_path).IsValid():
            stage.RemovePrim(task_path)
    UsdGeom.Xform.Define(stage, "/World/GraspTask")
    measured_pickup_center = pickup_center + np.asarray(args.pickup_center_offset)
    measured_place_center = place_center + np.asarray(args.place_center_offset)
    authored_pickup_center = measured_pickup_center.copy()
    authored_place_center = measured_place_center.copy()
    authored_cube_center = pickup_center.copy()
    if args.calibration_only:
        # Keep all task colliders away from the arm.  Calibration must measure
        # unconstrained rigid kinematics, not a contact-deflected articulation.
        authored_pickup_center += np.array([5.0, 0.0, 0.0])
        authored_place_center += np.array([5.0, 0.0, 0.0])
        authored_cube_center += np.array([5.0, 0.0, 0.0])
    # Keep a narrow support under the object. Its collision with the gripper is
    # filtered below because it is a task fixture, while cube-support contact
    # remains active throughout the approach and close phases.
    pickup_support_width = max(0.08, float(args.cube_size))
    pickup_support_xy = (pickup_support_width, pickup_support_width)
    place_support_xy = (0.022, 0.022) if args.calibration_only else (0.12, 0.12)
    for name, center in (
        ("PickupPedestal", authored_pickup_center),
        ("PlacePedestal", authored_place_center),
    ):
        if name == "PlacePedestal" and np.linalg.norm(
            authored_place_center - authored_pickup_center
        ) < 1.0e-6:
            continue
        pedestal_xy = pickup_support_xy if name == "PickupPedestal" else place_support_xy
        height = max(0.08, float(center[2] - 0.5 * args.cube_size))
        _add_box(
            stage,
            f"/World/GraspTask/{name}",
            (pedestal_xy[0], pedestal_xy[1], height),
            np.array([center[0], center[1], 0.5 * height]),
            material,
            dynamic=False,
        )
    cube_prim = _add_box(
        stage,
        "/World/GraspTask/Cube",
        (args.cube_size, args.cube_size, args.cube_size),
        authored_cube_center,
        material,
        dynamic=True,
    )
    # Native mode keeps all cube-gripper contacts enabled. Only the artificial
    # pickup fixture is filtered from the gripper so it cannot jam the linkage.
    filtered_gripper_body_paths: list[str] = []
    support_filtered_gripper_body_paths: list[str] = []
    gripper_root = Sdf.Path("/World/UnitreeH1/ZBot6S_left/Robotiq2F85")
    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        if (
            prim_path == gripper_root or prim_path.HasPrefix(gripper_root)
        ) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            support_filtered_gripper_body_paths.append(str(prim_path))
    if len(support_filtered_gripper_body_paths) <= 1:
        raise RuntimeError("no gripper rigid bodies found for support filtering")
    pickup_support_collision = stage.GetPrimAtPath(
        "/World/GraspTask/PickupPedestal/Geometry"
    )
    if not pickup_support_collision.IsValid():
        raise RuntimeError("missing pickup support collision geometry")
    support_relation = UsdPhysics.FilteredPairsAPI.Apply(
        pickup_support_collision
    ).CreateFilteredPairsRel()
    for body_path in support_filtered_gripper_body_paths:
        support_relation.AddTarget(Sdf.Path(body_path))
    # A bright visual material is independent from collision material.
    visual = UsdShade.Material.Define(stage, "/World/GraspTask/CubeVisual")
    shader = UsdShade.Shader.Define(stage, "/World/GraspTask/CubeVisual/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.95, 0.25, 0.12)
    )
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/GraspTask/Cube/Geometry")).Bind(visual)

    contact_report_body_paths = [
        "/World/GraspTask/Cube",
        "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/left_inner_finger",
        "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/right_inner_finger",
    ]
    for body_path in contact_report_body_paths:
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"contact-report body is invalid: {body_path}")
        if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            report_api = PhysxSchema.PhysxContactReportAPI.Get(stage, body_path)
        else:
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
        report_api.CreateThresholdAttr().Set(0.0)

    pad_material_prim = stage.GetPrimAtPath(
        "/World/PhysicsMaterials/NativeContactGripperPad"
    )
    if not pad_material_prim.IsValid():
        raise RuntimeError("missing native-contact fingertip material")
    pad_material_api = PhysxSchema.PhysxMaterialAPI.Apply(pad_material_prim)
    pad_material_api.CreateFrictionCombineModeAttr().Set("max")
    pad_material_api.CreateCompliantContactStiffnessAttr().Set(
        float(args.compliant_contact_stiffness)
    )
    pad_material_api.CreateCompliantContactDampingAttr().Set(
        float(args.compliant_contact_damping)
    )
    for finger_path in contact_report_body_paths[1:]:
        pad_root = stage.GetPrimAtPath(f"{finger_path}/collisions/pad")
        pad_prims = [
            prim for prim in pad_root.GetChildren()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ] if pad_root.IsValid() else []
        if not pad_prims:
            raise RuntimeError(f"missing native-contact pad colliders: {finger_path}")
        for pad_prim in pad_prims:
            pad_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(pad_prim)
            pad_collision_api.CreateContactOffsetAttr().Set(float(args.pad_contact_offset))
            pad_collision_api.CreateRestOffsetAttr().Set(0.0)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(args.physics_dt),
            render_interval=1,
            physx=sim_utils.PhysxCfg(
                enable_enhanced_determinism=True,
                enable_external_forces_every_iteration=True,
            ),
        )
    )
    execution_robot_cfg = H1_1111_CFG.copy()
    # The imported Robotiq linkage otherwise enters a velocity-limited limit
    # cycle after arm motion.  Keep this stabilization local to this executor;
    # the balance and trajectory-training configurations remain untouched.
    gripper_actuator = execution_robot_cfg.actuators["dual_grippers"]
    gripper_actuator.effort_limit_sim = float(args.gripper_effort_limit)
    gripper_actuator.velocity_limit_sim = float(args.gripper_velocity_limit)
    gripper_actuator.stiffness = float(args.gripper_stiffness)
    gripper_actuator.damping = float(args.gripper_damping)
    robot = Articulation(
        execution_robot_cfg.replace(prim_path="/World/UnitreeH1", spawn=None)
    )
    cube = RigidObject(RigidObjectCfg(prim_path="/World/GraspTask/Cube", spawn=None))
    cube_filter = ["/World/GraspTask/Cube"]
    left_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=(
                "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/left_inner_finger"
            ),
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=cube_filter,
            track_friction_forces=False,
            max_contact_data_count_per_prim=64,
        )
    )
    right_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=(
                "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/right_inner_finger"
            ),
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=cube_filter,
            track_friction_forces=False,
            max_contact_data_count_per_prim=64,
        )
    )
    fk_model = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to(
        args.device
    )
    overview_camera = None
    overview_camera_prim = None
    camera_tracking_offset = None
    if args.record_dir is not None:
        camera_path = "/World/GraspTask/OverviewCamera"
        eye = np.asarray(args.camera_eye, dtype=np.float64)
        target = np.asarray(args.camera_target, dtype=np.float64)
        sim_utils.spawn_camera(
            camera_path,
            sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=4.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 100.0),
            ),
            translation=tuple(float(v) for v in eye),
            orientation=tuple(float(v) for v in _look_at_quat_wxyz(eye, target)),
        )
        overview_camera = Camera(
            CameraCfg(
                prim_path=camera_path,
                update_period=0.0,
                spawn=None,
                data_types=["rgb"],
                width=int(args.record_width),
                height=int(args.record_height),
            )
        )
        overview_camera_prim = stage.GetPrimAtPath(camera_path)
        camera_tracking_offset = eye - target
    sim.reset()
    robot.reset()
    cube.reset()
    left_contact_sensor.reset()
    right_contact_sensor.reset()

    arm_ids, _ = robot.find_joints([f"^joint{i}_0$" for i in range(1, 7)], preserve_order=True)
    finger_ids, _ = robot.find_joints(["^finger_joint_0$"], preserve_order=True)
    base_ids, _ = robot.find_bodies(["^foot_0_0$"])
    finger_body_ids, _ = robot.find_bodies(
        ["^left_inner_finger_0$", "^right_inner_finger_0$"], preserve_order=True
    )
    tcp_body_ids, _ = robot.find_bodies(["^foot_1_0$"])
    if (
        len(arm_ids) != 6 or len(finger_ids) != 1
        or len(finger_body_ids) != 2 or len(tcp_body_ids) != 1
    ):
        raise RuntimeError("failed to resolve left arm/gripper joints and bodies")

    tcp_prim_path = "/World/UnitreeH1/ZBot6S_left/foot_1"
    if not stage.GetPrimAtPath(tcp_prim_path).IsValid():
        raise RuntimeError(f"missing attachment body: {tcp_prim_path}")

    joint_target = robot.data.default_joint_pos.clone()
    zero_joint_vel = torch.zeros_like(joint_target)
    root_position = robot.data.root_pos_w.clone()
    root_position[:, 2] += args.root_z_offset
    root_quaternion = robot.data.root_quat_w.clone()
    zero_root_velocity = torch.zeros((1, 6), device=robot.device)
    open_finger = 0.0
    fully_closed_finger = 0.7792895
    grasp_reference_finger = open_finger
    recording_active = [False]
    recording_phase = ["initialization"]
    recording_step = [0]
    recording_frame = [0]
    video_writer = None
    gif_writer = None
    video_path = None
    gif_path = None
    if args.record_dir is not None:
        import imageio.v2 as imageio

        args.record_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.record_dir / "pose_grasp_place_final.mp4"
        gif_path = args.record_dir / "pose_grasp_place_final.gif"
        video_writer = imageio.get_writer(
            video_path, fps=float(args.record_fps), codec="libx264", quality=7
        )
        gif_writer = imageio.get_writer(gif_path, mode="I", fps=float(args.record_fps))
    record_interval = max(
        1, int(round(1.0 / (float(args.record_fps) * sim.get_physics_dt())))
    )

    current_phase = ["initialization"]
    contact_events: list[dict[str, object]] = []
    contact_event_counts = {
        "arm_self_collision": 0,
        "arm_h1_collision": 0,
        "arm_environment_collision": 0,
    }
    maximum_contact_impulse = [0.0]
    task_contact_audit_active = [False]
    native_pad_contact_counts = {"left": 0, "right": 0}
    native_non_pad_object_contacts = [0]
    maximum_target_contact_impulse = [0.0]
    target_contact_events: list[dict[str, object]] = []
    native_monitor_active = [False]
    contact_phase_active = [False]
    maximum_cube_linear_speed = [0.0]
    maximum_cube_angular_speed = [0.0]
    current_finger_force_n = {"left": 0.0, "right": 0.0}
    peak_finger_force_n = {"left": 0.0, "right": 0.0}
    bilateral_force_steps = [0]
    contact_sensor_step = [0]
    force_sample_step = [0]
    finger_force_samples: list[dict[str, object]] = []
    timeseries_active = [False]
    emergency_stop_flag = [False]
    timeseries: dict[str, list[object]] = {
        key: []
        for key in (
            "t", "q", "qd", "qdd", "qddd", "gripper_left_force",
            "gripper_right_force", "object_world_pos", "object_world_quat",
            "gripper_world_pos", "gripper_world_quat", "object_in_gripper_pos",
            "object_in_gripper_quat", "object_linear_velocity",
            "object_angular_velocity", "base_reaction_force",
            "base_reaction_torque", "state_machine_phase", "emergency_stop",
        )
    }
    previous_qd: list[np.ndarray | None] = [None]
    previous_qdd: list[np.ndarray | None] = [None]

    def flush_timeseries() -> None:
        if args.timeseries_output is None or not timeseries["t"]:
            return
        args.timeseries_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.timeseries_output,
            **{key: np.asarray(value) for key, value in timeseries.items()},
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "physics_dt_s": float(args.physics_dt),
                        "quaternion_order": "wxyz",
                        "base_reaction_available": False,
                        "undefined_derivatives_are_nan": True,
                    },
                    ensure_ascii=False,
                )
            ),
        )

    def on_contacts(headers, contact_data) -> None:
        if not task_contact_audit_active[0]:
            return
        arm_marker = "/ZBot6S_left/"
        robot_root = "/World/UnitreeH1/"
        for header in headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            impulse = sum(float(contact_data[index].impulse) for index in range(start, stop))
            if impulse < float(args.contact_impulse_threshold):
                continue
            cube0 = actor0.startswith("/World/GraspTask/Cube") or collider0.startswith(
                "/World/GraspTask/Cube"
            )
            cube1 = actor1.startswith("/World/GraspTask/Cube") or collider1.startswith(
                "/World/GraspTask/Cube"
            )
            gripper0 = "/Robotiq2F85/" in actor0 or "/Robotiq2F85/" in collider0
            gripper1 = "/Robotiq2F85/" in actor1 or "/Robotiq2F85/" in collider1
            if (cube0 and gripper1) or (cube1 and gripper0):
                gripper_actor = actor1 if gripper1 else actor0
                gripper_collider = collider1 if gripper1 else collider0
                is_pad = (
                    "/collisions/pad" in gripper_collider
                    and (
                        "/left_inner_finger" in gripper_collider
                        or "/right_inner_finger" in gripper_collider
                    )
                )
                if is_pad:
                    side = (
                        "left"
                        if "/left_inner_finger" in gripper_collider
                        else "right"
                    )
                    native_pad_contact_counts[side] += 1
                    maximum_target_contact_impulse[0] = max(
                        maximum_target_contact_impulse[0], impulse
                    )
                    if len(target_contact_events) < 200:
                        target_contact_events.append(
                            {
                                "phase": current_phase[0],
                                "side": side,
                                "gripper_actor": gripper_actor,
                                "gripper_collider": gripper_collider,
                                "impulse": impulse,
                            }
                        )
                    continue
                native_non_pad_object_contacts[0] += 1
            in_arm0 = arm_marker in actor0
            in_arm1 = arm_marker in actor1
            if not (in_arm0 or in_arm1):
                continue
            if "/Robotiq2F85/" in actor0 and "/Robotiq2F85/" in actor1:
                continue
            if in_arm0 and in_arm1:
                category = "arm_self_collision"
            elif actor0.startswith(robot_root) and actor1.startswith(robot_root):
                category = "arm_h1_collision"
            else:
                category = "arm_environment_collision"
            contact_event_counts[category] += 1
            maximum_contact_impulse[0] = max(maximum_contact_impulse[0], impulse)
            if len(contact_events) < 200:
                contact_events.append({
                    "phase": current_phase[0],
                    "category": category,
                    "actor0": actor0,
                    "actor1": actor1,
                    "collider0": collider0,
                    "collider1": collider1,
                    "impulse": impulse,
                })

    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        on_contacts
    )

    def collision_summary() -> dict[str, object]:
        total = sum(contact_event_counts.values())
        return {
            "collision_free": total == 0,
            "counts": dict(contact_event_counts),
            "maximum_impulse": maximum_contact_impulse[0],
            "events_truncated": total > len(contact_events),
            "events": list(contact_events),
        }

    def native_contact_summary() -> dict[str, object]:
        target_events_observed = len(target_contact_events) > 0
        sensor_contact_observed = bilateral_force_steps[0] > 0
        return {
            "pad_contact_counts": dict(native_pad_contact_counts),
            "bilateral_pad_contact": all(
                count > 0 for count in native_pad_contact_counts.values()
            ),
            "non_pad_object_contacts": native_non_pad_object_contacts[0],
            "maximum_target_contact_impulse_n_s": maximum_target_contact_impulse[0],
            "target_contact_impulse_limit_n_s": float(
                args.maximum_target_contact_impulse
            ),
            "target_contact_events_observed": target_events_observed,
            "target_impulse_auditable": target_events_observed,
            "target_impulse_within_limit": (
                maximum_target_contact_impulse[0]
                <= float(args.maximum_target_contact_impulse)
                if target_events_observed
                else None
            ),
            "contact_force_sensor": {
                "source": "Isaac Lab ContactSensor filtered to /World/GraspTask/Cube",
                "minimum_bilateral_force_n": float(
                    args.minimum_bilateral_contact_force
                ),
                "sample_decimation": int(args.contact_sensor_decimation),
                "sample_frequency_hz": float(
                    1.0
                    / (
                        sim.get_physics_dt()
                        * max(1, int(args.contact_sensor_decimation))
                    )
                ),
                "peak_force_n": dict(peak_finger_force_n),
                "bilateral_force_steps": int(bilateral_force_steps[0]),
                "bilateral_contact_observed": sensor_contact_observed,
                "samples": list(finger_force_samples),
            },
            "events": list(target_contact_events),
        }

    def pin_root() -> None:
        robot.write_root_state_to_sim(
            torch.cat((root_position, root_quaternion, zero_root_velocity), dim=-1)
        )

    def step(q: np.ndarray, finger: float, render: bool = True) -> None:
        joint_target[:, arm_ids] = torch.as_tensor(q, dtype=torch.float32, device=robot.device)
        joint_target[:, finger_ids] = finger
        robot.set_joint_position_target(joint_target)
        pin_root()
        robot.write_data_to_sim()
        if overview_camera_prim is not None and args.camera_track_object:
            tracked_target = cube.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            tracked_eye = tracked_target + camera_tracking_offset
            tracked_quaternion = _look_at_quat_wxyz(tracked_eye, tracked_target)
            overview_camera_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(*[float(v) for v in tracked_eye])
            )
            overview_camera_prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(
                    float(tracked_quaternion[0]),
                    *[float(v) for v in tracked_quaternion[1:]],
                )
            )
        sim.step(render=render)
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        contact_sensor_step[0] += 1
        sensor_sampled = (
            contact_sensor_step[0] % max(1, int(args.contact_sensor_decimation)) == 0
        )
        if sensor_sampled:
            left_contact_sensor.update(sim.get_physics_dt(), force_recompute=True)
            right_contact_sensor.update(sim.get_physics_dt(), force_recompute=True)
            for side, sensor in (
                ("left", left_contact_sensor),
                ("right", right_contact_sensor),
            ):
                force_matrix = sensor.data.force_matrix_w
                force_value = (
                    0.0
                    if force_matrix is None
                    else float(torch.linalg.vector_norm(force_matrix[0, 0, 0]).item())
                )
                current_finger_force_n[side] = force_value
                peak_finger_force_n[side] = max(
                    peak_finger_force_n[side], force_value
                )
        if task_contact_audit_active[0] and sensor_sampled:
            threshold = float(args.minimum_bilateral_contact_force)
            if all(value >= threshold for value in current_finger_force_n.values()):
                bilateral_force_steps[0] += 1
            force_sample_step[0] += 1
            if force_sample_step[0] % 20 == 0 and len(finger_force_samples) < 1000:
                finger_force_samples.append(
                    {
                        "phase": current_phase[0],
                        "left_n": current_finger_force_n["left"],
                        "right_n": current_finger_force_n["right"],
                    }
                )
        if timeseries_active[0]:
            dt = float(sim.get_physics_dt())
            measured_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
            measured_qd = robot.data.joint_vel[0, arm_ids].detach().cpu().numpy().astype(np.float64)
            if previous_qd[0] is None:
                measured_qdd = np.full(6, np.nan, dtype=np.float64)
            else:
                measured_qdd = (measured_qd - previous_qd[0]) / dt
            if previous_qdd[0] is None or not np.isfinite(measured_qdd).all():
                measured_qddd = np.full(6, np.nan, dtype=np.float64)
            else:
                measured_qddd = (measured_qdd - previous_qdd[0]) / dt
            previous_qd[0] = measured_qd.copy()
            previous_qdd[0] = measured_qdd.copy()
            object_position = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            object_quaternion = cube.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            gripper_position = robot.data.body_link_pos_w[0, tcp_body_ids[0]].detach().cpu().numpy().astype(np.float64)
            gripper_quaternion = robot.data.body_link_quat_w[0, tcp_body_ids[0]].detach().cpu().numpy().astype(np.float64)
            gripper_rotation = _quat_matrix(gripper_quaternion)
            object_rotation = _quat_matrix(object_quaternion)
            relative_position = gripper_rotation.T @ (object_position - gripper_position)
            relative_quaternion = _matrix_quat(gripper_rotation.T @ object_rotation)
            timeseries["t"].append(len(timeseries["t"]) * dt)
            timeseries["q"].append(measured_q)
            timeseries["qd"].append(measured_qd)
            timeseries["qdd"].append(measured_qdd)
            timeseries["qddd"].append(measured_qddd)
            timeseries["gripper_left_force"].append(current_finger_force_n["left"])
            timeseries["gripper_right_force"].append(current_finger_force_n["right"])
            timeseries["object_world_pos"].append(object_position)
            timeseries["object_world_quat"].append(object_quaternion)
            timeseries["gripper_world_pos"].append(gripper_position)
            timeseries["gripper_world_quat"].append(gripper_quaternion)
            timeseries["object_in_gripper_pos"].append(relative_position)
            timeseries["object_in_gripper_quat"].append(relative_quaternion)
            timeseries["object_linear_velocity"].append(cube.data.root_lin_vel_w[0].detach().cpu().numpy())
            timeseries["object_angular_velocity"].append(cube.data.root_ang_vel_w[0].detach().cpu().numpy())
            timeseries["base_reaction_force"].append(np.full(3, np.nan))
            timeseries["base_reaction_torque"].append(np.full(3, np.nan))
            timeseries["state_machine_phase"].append(current_phase[0])
            timeseries["emergency_stop"].append(emergency_stop_flag[0])
        if native_monitor_active[0] and (
            contact_phase_active[0] or target_contact_events or contact_events
        ):
            linear_speed = float(torch.linalg.vector_norm(cube.data.root_lin_vel_w[0]).item())
            angular_speed = float(torch.linalg.vector_norm(cube.data.root_ang_vel_w[0]).item())
            maximum_cube_linear_speed[0] = max(maximum_cube_linear_speed[0], linear_speed)
            maximum_cube_angular_speed[0] = max(maximum_cube_angular_speed[0], angular_speed)
            if linear_speed > float(args.maximum_cube_linear_speed):
                emergency_stop_flag[0] = True
                if timeseries["emergency_stop"]:
                    timeseries["emergency_stop"][-1] = True
                flush_timeseries()
                raise RuntimeError(
                    "native-contact emergency stop: "
                    f"phase={current_phase[0]}, cube linear speed={linear_speed:.3f} m/s, "
                    f"contacts={contact_events[-3:]}, target_contacts={target_contact_events[-3:]}"
                )
            if angular_speed > float(args.maximum_cube_angular_speed):
                emergency_stop_flag[0] = True
                if timeseries["emergency_stop"]:
                    timeseries["emergency_stop"][-1] = True
                flush_timeseries()
                raise RuntimeError(
                    "native-contact emergency stop: "
                    f"phase={current_phase[0]}, cube angular speed={angular_speed:.3f} rad/s, "
                    f"contacts={contact_events[-3:]}, target_contacts={target_contact_events[-3:]}"
                )
        if overview_camera is not None:
            overview_camera.update(sim.get_physics_dt())
            if recording_active[0]:
                recording_step[0] += 1
                if recording_step[0] % record_interval == 0:
                    from PIL import Image, ImageDraw

                    packet = overview_camera.data.output.get("rgb")
                    if packet is not None and packet.numel() > 0:
                        rgb = packet[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                        frame = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(frame)
                        label = (
                            f"{recording_phase[0]}  |  Visual CFM + Pose IK  |  "
                            f"native PhysX contact  |  "
                            f"force L/R: {current_finger_force_n['left']:.2f}/"
                            f"{current_finger_force_n['right']:.2f} N"
                        )
                        draw.rectangle((0, 0, frame.width, 28), fill=(16, 18, 22))
                        draw.text((10, 7), label, fill=(245, 245, 245))
                        array = np.asarray(frame)
                        video_writer.append_data(array)
                        gif_frame = frame.resize((480, 270))
                        gif_writer.append_data(np.asarray(gif_frame))
                        recording_frame[0] += 1

    pad_local_centers = [LEFT_PAD_FACE[0].copy(), RIGHT_PAD_FACE[0].copy()]

    def calibrate_inward_pad_faces() -> dict[str, object]:
        positions = [
            robot.data.body_pos_w[0, body_id].cpu().numpy()
            for body_id in finger_body_ids
        ]
        rotations = [
            _quat_matrix(robot.data.body_quat_w[0, body_id].cpu().numpy())
            for body_id in finger_body_ids
        ]
        left_local = LEFT_PAD_FACE[0]
        right_local = RIGHT_PAD_FACE[0]
        left_w = positions[0] + rotations[0] @ left_local
        right_w = positions[1] + rotations[1] @ right_local
        measured_gap = float(np.linalg.norm(right_w - left_w))
        if not 0.0 < measured_gap < 0.12:
            raise RuntimeError(f"invalid visual fingertip gap: {measured_gap:.4f} m")
        pad_local_centers[0] = left_local.copy()
        pad_local_centers[1] = right_local.copy()
        return {
            "left_local_face_center": left_local.tolist(),
            "right_local_face_center": right_local.tolist(),
        }

    def pad_calibration() -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        left_position = robot.data.body_pos_w[0, finger_body_ids[0]].cpu().numpy()
        right_position = robot.data.body_pos_w[0, finger_body_ids[1]].cpu().numpy()
        left_rotation = _quat_matrix(
            robot.data.body_quat_w[0, finger_body_ids[0]].cpu().numpy()
        )
        right_rotation = _quat_matrix(
            robot.data.body_quat_w[0, finger_body_ids[1]].cpu().numpy()
        )
        left_center = left_position + left_rotation @ pad_local_centers[0]
        right_center = right_position + right_rotation @ pad_local_centers[1]
        center_distance = float(np.linalg.norm(right_center - left_center))
        normal = (right_center - left_center) / center_distance
        return 0.5 * (left_center + right_center), normal, center_distance, left_center, right_center

    def cube_width_along(axis_w: np.ndarray) -> float:
        cube_rotation = _quat_matrix(cube.data.root_quat_w[0].cpu().numpy())
        local_axis = cube_rotation.T @ np.asarray(axis_w, dtype=np.float64)
        return float(args.cube_size * np.abs(local_axis).sum())

    def grasp_point_w() -> np.ndarray:
        cube_position = cube.data.root_pos_w[0].cpu().numpy()
        cube_rotation = _quat_matrix(cube.data.root_quat_w[0].cpu().numpy())
        return cube_position + cube_rotation @ grasp_point_offset_object

    def interpolate_finger_for_gap(samples: list[dict[str, float]], target_gap: float) -> float:
        ordered = sorted(samples, key=lambda sample: sample["surface_gap_m"])
        gaps = np.asarray([sample["surface_gap_m"] for sample in ordered])
        fingers = np.asarray([sample["finger_joint_rad"] for sample in ordered])
        if not float(gaps[0]) <= target_gap <= float(gaps[-1]):
            raise RuntimeError(
                f"requested grasp gap {target_gap:.4f} m is outside calibrated range "
                f"[{gaps[0]:.4f}, {gaps[-1]:.4f}] m"
            )
        return float(np.interp(target_gap, gaps, fingers))

    def servo_pad_gap(
        q: np.ndarray,
        initial_target: float,
        desired_gap: float,
        track_object_width: bool = False,
    ) -> tuple[float, dict[str, object]]:
        command = float(np.clip(initial_target, open_finger, fully_closed_finger))
        history: list[dict[str, float]] = []
        within_tolerance = 0
        for iteration in range(80):
            for _ in range(20):
                step(q, command)
            _, pad_axis, measured_gap, _, _ = pad_calibration()
            if track_object_width:
                desired_gap = (
                    cube_width_along(pad_axis) - float(args.preload_gap)
                )
            actual_position = float(robot.data.joint_pos[0, finger_ids[0]].item())
            error = float(measured_gap - desired_gap)
            history.append(
                {
                    "iteration": float(iteration),
                    "command_rad": command,
                    "actual_joint_rad": actual_position,
                    "surface_gap_m": float(measured_gap),
                    "gap_error_m": error,
                }
            )
            if abs(error) <= float(args.gap_servo_tolerance):
                within_tolerance += 1
                if within_tolerance >= 3:
                    return command, {
                        "desired_gap_m": float(desired_gap),
                        "converged": True,
                        "history": history,
                    }
            else:
                within_tolerance = 0
            # Positive gap error means the fingers remain too far apart. The
            # imported master joint closes in the positive direction.
            command = float(
                np.clip(command + np.clip(4.0 * error, -0.06, 0.06),
                        open_finger, fully_closed_finger)
            )
        raise RuntimeError(
            "visual pad-gap servo failed: "
            f"desired={desired_gap:.6f}, last={history[-1]}"
        )

    def grasp_geometry() -> dict[str, object]:
        midpoint, normal, surface_gap, left_center, right_center = pad_calibration()
        cube_position = cube.data.root_pos_w[0].cpu().numpy()
        desired_grasp_point = grasp_point_w()
        delta = desired_grasp_point - midpoint
        axial_offset = float(np.dot(delta, normal))
        lateral_offset = float(np.linalg.norm(delta - axial_offset * normal))
        object_width = cube_width_along(normal)
        return {
            "cube_center_w": cube_position.tolist(),
            "cube_quaternion_wxyz": cube.data.root_quat_w[0].cpu().numpy().tolist(),
            "grasp_point_w": desired_grasp_point.tolist(),
            "grasp_point_offset_object_m": grasp_point_offset_object.tolist(),
            "pad_midpoint_w": midpoint.tolist(),
            "pad_normal_w": normal.tolist(),
            "left_pad_center_w": left_center.tolist(),
            "right_pad_center_w": right_center.tolist(),
            "surface_gap_m": float(surface_gap),
            "object_width_along_closing_axis_m": object_width,
            "axial_center_offset_m": axial_offset,
            "lateral_center_offset_m": lateral_offset,
            "finger_joint_position_rad": float(
                robot.data.joint_pos[0, finger_ids[0]].item()
            ),
            "finger_joint_velocity_rad_s": float(
                robot.data.joint_vel[0, finger_ids[0]].item()
            ),
            "between_fingers": bool(
                abs(axial_offset) <= 0.005
                and lateral_offset <= 0.012
                and surface_gap <= object_width + 0.004
                and surface_gap >= object_width - 0.006
            ),
        }

    attachment_path = "/World/GraspTask/GraspAttachment"

    def create_grasp_attachment(geometry: dict[str, object]) -> float:
        if not bool(geometry["between_fingers"]):
            raise RuntimeError(f"grasp attachment rejected by finger geometry: {geometry}")
        tcp_position = robot.data.body_link_pos_w[0, tcp_body_ids[0]].cpu().numpy()
        tcp_rotation = _quat_matrix(
            robot.data.body_link_quat_w[0, tcp_body_ids[0]].cpu().numpy()
        )
        cube_position = cube.data.root_pos_w[0].cpu().numpy()
        cube_rotation = _quat_matrix(cube.data.root_quat_w[0].cpu().numpy())
        anchor_position = np.asarray(geometry["pad_midpoint_w"], dtype=np.float64)
        local_position = tcp_rotation.T @ (anchor_position - tcp_position)
        local_quaternion = _matrix_quat(tcp_rotation.T @ cube_rotation)
        joint = UsdPhysics.FixedJoint.Define(stage, attachment_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(tcp_prim_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/GraspTask/Cube")])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_position]))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(float(local_quaternion[0]), *[float(v) for v in local_quaternion[1:]])
        )
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(*[float(v) for v in grasp_point_offset_object])
        )
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return float(np.linalg.norm(cube_position - calibrated_pickup))

    # Initialization calibration at contact pose; no task metrics are collected yet.
    q_contact = np.asarray(plan["q_grasp"], dtype=np.float64)
    joint_target[:, arm_ids] = torch.as_tensor(q_contact, dtype=torch.float32, device=robot.device)
    joint_target[:, finger_ids] = grasp_reference_finger
    robot.write_joint_state_to_sim(joint_target, zero_joint_vel)
    for _ in range(160):
        step(q_contact, grasp_reference_finger)
    pad_face_calibration = calibrate_inward_pad_faces()
    measured_reference_midpoint, pad_normal, pad_gap, _, _ = pad_calibration()
    # Use the repeatable free-space calibration supplied on the command line.
    # A contact-state midpoint is diagnostic only because support contact can
    # elastically deflect the articulation.
    calibrated_pickup = measured_pickup_center.copy()
    if args.calibration_only:
        if args.reference_only:
            calibration_report = {
                "planned_pickup_center_w": pickup_center.tolist(),
                "calibrated_pickup_center_w": calibrated_pickup.tolist(),
                "measured_reference_midpoint_w": measured_reference_midpoint.tolist(),
                "reference_pad_normal_w": pad_normal.tolist(),
                "pad_face_calibration": pad_face_calibration,
                "reference_pad_gap_m": float(pad_gap),
                "reference_finger_joint_rad": float(grasp_reference_finger),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(calibration_report, indent=2))
            return
        aperture_samples = []
        for finger_value in (open_finger, 0.2, 0.4, 0.6, fully_closed_finger):
            for _ in range(160):
                step(q_contact, finger_value)
            midpoint, _, surface_gap, _, _ = pad_calibration()
            aperture_samples.append(
                {
                    "finger_joint_rad": float(finger_value),
                    "pad_midpoint_w": midpoint.tolist(),
                    "surface_gap_m": float(surface_gap),
                }
            )
        closed_pickup = np.asarray(aperture_samples[-1]["pad_midpoint_w"])
        closed_pad_gap = float(aperture_samples[-1]["surface_gap_m"])
        q_place = np.asarray(plan["q_place"], dtype=np.float64)
        joint_target[:, arm_ids] = torch.as_tensor(q_place, dtype=torch.float32, device=robot.device)
        joint_target[:, finger_ids] = grasp_reference_finger
        robot.write_joint_state_to_sim(joint_target, zero_joint_vel)
        for _ in range(40):
            step(q_place, grasp_reference_finger)
        place_grip_midpoint, _, place_grip_gap, _, _ = pad_calibration()
        calibration_report = {
            "planned_pickup_center_w": pickup_center.tolist(),
            "calibrated_pickup_center_w": calibrated_pickup.tolist(),
            "measured_reference_midpoint_w": measured_reference_midpoint.tolist(),
            "error_m": float(np.linalg.norm(calibrated_pickup - pickup_center)),
            "delta_w_m": (calibrated_pickup - pickup_center).tolist(),
            "open_pad_gap_m": pad_gap,
            "closed_pad_midpoint_w": closed_pickup.tolist(),
            "closed_pad_gap_m": closed_pad_gap,
            "open_finger_joint_rad": open_finger,
            "closed_finger_joint_rad": fully_closed_finger,
            "aperture_samples": aperture_samples,
            "place_grip_midpoint_w": place_grip_midpoint.tolist(),
            "planned_place_center_w": place_center.tolist(),
            "place_center_delta_w_m": (place_grip_midpoint - place_center).tolist(),
            "place_grip_gap_m": float(place_grip_gap),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(calibration_report, indent=2))
        return
    # Keep the cube upright, but align one horizontal face normal with the
    # horizontal projection of the measured jaw closing axis.  This preserves a
    # stable support contact and avoids fingertip-on-edge grasping.
    yaw = float(np.arctan2(pad_normal[1], pad_normal[0]))
    cube_quaternion = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64
    )
    # The given object position is the single task target and is also the exact
    # pedestal center.  The gripper must servo to the object, never vice versa.
    calibrated_pickup = measured_pickup_center.copy()
    cube_pose = torch.tensor(
        [[*calibrated_pickup.tolist(), *cube_quaternion.tolist()]],
        dtype=torch.float32,
        device=robot.device,
    )
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))

    q_start = np.asarray(plan["q_start"], dtype=np.float64)
    # Calibrate the opening curve at the remote trajectory start so the fingers
    # cannot be jammed by the pickup pedestal or object during the sweep.
    for _ in range(80):
        step(q_start, open_finger, render=False)
    # Choose a holding aperture that matches the object's projection along the
    # actual jaw-closing axis.
    aperture_samples: list[dict[str, float]] = []
    if args.grasp_finger_target is None:
        for finger_value in (
            open_finger, 0.10, 0.20, 0.30, 0.40, 0.60, fully_closed_finger
        ):
            for _ in range(160):
                step(q_start, finger_value, render=False)
            _, _, sample_gap, _, _ = pad_calibration()
            aperture_samples.append(
                {
                    "finger_joint_rad": float(finger_value),
                    "measured_finger_joint_rad": float(
                        robot.data.joint_pos[0, finger_ids[0]].item()
                    ),
                    "surface_gap_m": float(sample_gap),
                }
            )
        full_travel_settle_steps = max(
            320,
            int(
                np.ceil(
                    (
                        fully_closed_finger
                        / max(float(args.gripper_velocity_limit), 1.0e-3)
                        + 0.5
                    )
                    / sim.get_physics_dt()
                )
            ),
        )
    else:
        if not 0.0 <= float(args.grasp_finger_target) <= fully_closed_finger:
            raise ValueError("grasp_finger_target is outside the gripper range")
        aperture_samples.append(
            {
                "finger_joint_rad": float(args.grasp_finger_target),
                "calibration_skipped": True,
            }
        )
        full_travel_settle_steps = max(
            320,
            int(
                np.ceil(
                    (
                        fully_closed_finger
                        / max(float(args.gripper_velocity_limit), 1.0e-3)
                        + 0.5
                    )
                    / sim.get_physics_dt()
                )
            ),
        )
    for _ in range(full_travel_settle_steps):
        step(q_start, open_finger, render=False)
    # Measure aperture and midpoint drift at the exact grasp posture with the
    # cube temporarily moved out of the workspace. This supplies feed-forward
    # compensation without making any post-contact Cartesian correction.
    remote_cube_pose = cube_pose.clone()
    remote_cube_pose[:, 0] += 5.0
    cube.write_root_pose_to_sim(remote_cube_pose)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
    for _ in range(full_travel_settle_steps):
        step(q_contact, open_finger, render=False)
    open_midpoint, open_normal, open_gap, _, _ = pad_calibration()
    projected_cube_width = cube_width_along(open_normal)
    if open_gap < projected_cube_width + 0.0005:
        raise RuntimeError(
            f"open gripper does not clear cube: gap={open_gap:.4f} m, "
            f"projected width={projected_cube_width:.4f} m"
        )
    grasp_finger = (
        interpolate_finger_for_gap(
            aperture_samples, projected_cube_width - float(args.preload_gap)
        )
        if args.grasp_finger_target is None
        else float(args.grasp_finger_target)
    )
    if args.auto_closing_midpoint_drift:
        close_calibration_steps = max(
            320,
            int(
                np.ceil(
                    (
                        grasp_finger
                        / max(float(args.gripper_velocity_limit), 1.0e-3)
                        + 0.5
                    )
                    / sim.get_physics_dt()
                )
            ),
        )
        for _ in range(close_calibration_steps):
            step(q_contact, grasp_finger, render=False)
        closed_midpoint, _, _, _, _ = pad_calibration()
        closing_midpoint_drift_w = closed_midpoint - open_midpoint
        for _ in range(close_calibration_steps):
            step(q_contact, open_finger, render=False)
    else:
        closing_midpoint_drift_w = np.asarray(
            args.closing_midpoint_drift, dtype=np.float64
        )
    print(
        json.dumps(
            {
                "aperture_calibration": aperture_samples,
                "projected_cube_width_m": float(projected_cube_width),
                "requested_grasp_gap_m": float(
                    projected_cube_width - float(args.preload_gap)
                ),
                "grasp_finger_target_rad": float(grasp_finger),
                "closing_midpoint_drift_w_m": closing_midpoint_drift_w.tolist(),
            },
            indent=2,
        ),
        flush=True,
    )

    # The contact-pose aperture measurement above is pre-task calibration and
    # may touch the cube. Return the arm to the remote start first, then perform
    # the single final object reset. No object pose write occurs after the task
    # audit is enabled below.
    for _ in range(240):
        step(q_start, open_finger, render=False)
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
    for _ in range(80):
        step(q_start, open_finger, render=False)
    support_reference_w = cube_pose[0, :3].detach().cpu().numpy().astype(np.float64)
    support_after_reset_w = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)

    if args.camera_preview_only:
        recording_phase[0] = "camera_preview"
        recording_active[0] = True
        for _ in range(max(record_interval * 4, 40)):
            step(q_contact, open_finger)
        if video_writer is not None:
            video_writer.close()
        if gif_writer is not None:
            gif_writer.close()
        preview_report = {
            "camera_eye_w": list(args.camera_eye),
            "camera_target_w": list(args.camera_target),
            "frames": int(recording_frame[0]),
            "video": None if video_path is None else str(video_path),
            "gif": None if gif_path is None else str(gif_path),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(preview_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(preview_report, indent=2))
        return

    joint_target[:, arm_ids] = torch.as_tensor(q_start, dtype=torch.float32, device=robot.device)
    joint_target[:, finger_ids] = open_finger
    robot.write_joint_state_to_sim(joint_target, zero_joint_vel)
    task_started = False
    initial_cube = None
    samples: list[dict[str, object]] = []
    phase_max_z: dict[str, float] = {}

    def run_path(
        name: str,
        path: np.ndarray,
        duration: float,
        finger: float,
        smooth_time: bool = False,
    ) -> None:
        nonlocal task_started, initial_cube
        recording_phase[0] = name
        current_phase[0] = name
        values = np.asarray(path, dtype=np.float64)
        steps = max(2, int(round(float(duration) / sim.get_physics_dt())))
        for index in range(steps):
            phase = index / max(1, steps - 1)
            if smooth_time:
                # Quintic time scaling gives zero velocity and acceleration at
                # both ends, reducing the contact wrench at lift onset.
                phase = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
            coordinate = phase * (len(values) - 1)
            low = int(np.floor(coordinate))
            high = min(low + 1, len(values) - 1)
            alpha = coordinate - low
            q = (1.0 - alpha) * values[low] + alpha * values[high]
            step(q, finger)
            position = cube.data.root_pos_w[0].cpu().numpy().copy()
            phase_max_z[name] = max(phase_max_z.get(name, -np.inf), float(position[2]))
            if task_started and initial_cube is None:
                initial_cube = position.copy()
            elif (
                task_started
                and initial_cube is not None
                and name in {
                    "visual_cfm_to_pregrasp",
                    "approach",
                    "preclose_centering",
                }
            ):
                displacement = float(np.linalg.norm(position - initial_cube))
                if displacement > float(args.maximum_pregrasp_object_displacement):
                    raise RuntimeError(
                        "pregrasp object-displacement emergency stop: "
                        f"phase={name}, index={index}/{steps}, "
                        f"displacement={displacement:.6f} m, q={q.tolist()}, "
                        f"position_w={position.tolist()}, "
                        f"contacts={contact_events[-5:]}"
                    )
            if index % 20 == 0 or index == steps - 1:
                samples.append({"phase": name, "position_w": position.tolist()})

    def run_finger_phase(
        name: str,
        q: np.ndarray,
        duration: float,
        finger_start: float,
        finger_end: float,
    ) -> None:
        nonlocal task_started, initial_cube
        recording_phase[0] = name
        current_phase[0] = name
        steps = max(2, int(round(float(duration) / sim.get_physics_dt())))
        for index in range(steps):
            phase = index / max(1, steps - 1)
            blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
            finger = (1.0 - blend) * finger_start + blend * finger_end
            step(q, float(finger))
            position = cube.data.root_pos_w[0].cpu().numpy().copy()
            phase_max_z[name] = max(phase_max_z.get(name, -np.inf), float(position[2]))
            if task_started and initial_cube is None:
                initial_cube = position.copy()
            if index % 20 == 0 or index == steps - 1:
                samples.append(
                    {
                        "phase": name,
                        "position_w": position.tolist(),
                        "finger_joint_rad": float(finger),
                    }
                )

    def center_gripper(
        q_seed: np.ndarray,
        finger_value: float,
        target_point_w: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        q_command = np.asarray(q_seed, dtype=np.float64).copy()
        start_q = q_command.copy()
        errors = []
        measured_midpoint = pad_calibration()[0]
        measured_tcp_position = robot.data.body_link_pos_w[0, tcp_body_ids[0]].cpu().numpy()
        measured_tcp_rotation = _quat_matrix(
            robot.data.body_link_quat_w[0, tcp_body_ids[0]].cpu().numpy()
        )
        center_offset_tcp = measured_tcp_rotation.T @ (
            measured_midpoint - measured_tcp_position
        )
        search_step_count = 0

        def search_step(q_value: np.ndarray) -> None:
            nonlocal search_step_count
            search_step_count += 1
            # Pump the desktop UI periodically without rendering every search sample.
            render_heartbeat = not args.headless and search_step_count % 20 == 0
            step(q_value, finger_value, render=render_heartbeat)
            if task_started and initial_cube is not None:
                position = cube.data.root_pos_w[0].detach().cpu().numpy()
                displacement = float(np.linalg.norm(position - initial_cube))
                if displacement > float(args.maximum_pregrasp_object_displacement):
                    raise RuntimeError(
                        "preclose search object-displacement emergency stop: "
                        f"displacement={displacement:.6f} m, "
                        f"position_w={position.tolist()}"
                    )

        def analytic_center_world(q_value: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tcp_position, tcp_quaternion = fk_model(
                    torch.as_tensor(
                        q_value[None], dtype=torch.float32, device=args.device
                    )
                )
            tcp_position_np = tcp_position[0].cpu().numpy().astype(np.float64)
            tcp_rotation = _quat_matrix(
                tcp_quaternion[0].cpu().numpy().astype(np.float64)
            )
            center_foot0 = tcp_position_np + tcp_rotation @ center_offset_tcp
            return static_from_foot0(center_foot0)

        for _ in range(12):
            midpoint, _, _, _, _ = pad_calibration()
            target = (
                grasp_point_w().copy()
                if target_point_w is None
                else np.asarray(target_point_w, dtype=np.float64).copy()
            )
            error = target - midpoint
            errors.append(float(np.linalg.norm(error)))
            if errors[-1] <= 0.0005:
                break
            epsilon = 0.002
            position_jacobian = np.zeros((3, 6), dtype=np.float64)
            for joint_index in range(6):
                q_plus = q_command.copy()
                q_minus = q_command.copy()
                q_plus[joint_index] += epsilon
                q_minus[joint_index] -= epsilon
                plus = analytic_center_world(q_plus)
                minus = analytic_center_world(q_minus)
                position_jacobian[:, joint_index] = (plus - minus) / (2.0 * epsilon)
            damping = 1.0e-3
            delta = position_jacobian.T @ np.linalg.solve(
                position_jacobian @ position_jacobian.T
                + damping * np.eye(3),
                error,
            )
            candidate = (q_command + np.clip(delta, -0.03, 0.03) + np.pi) % (
                2.0 * np.pi
            ) - np.pi
            previous_q = q_command.copy()
            tested = []
            for blend in np.linspace(0.1, 1.0, 10):
                trial_q = (1.0 - blend) * previous_q + blend * candidate
                for _ in range(20):
                    search_step(trial_q)
                trial_midpoint = pad_calibration()[0]
                tested.append(
                    (float(np.linalg.norm(target - trial_midpoint)), trial_q.copy())
                )
            candidate_error, best_q = min(tested, key=lambda value: value[0])
            if candidate_error >= errors[-1] - 1.0e-5:
                for blend in np.linspace(0.0, 1.0, 21)[1:]:
                    search_step((1.0 - blend) * candidate + blend * previous_q)
                break
            for blend in np.linspace(0.0, 1.0, 21)[1:]:
                search_step((1.0 - blend) * candidate + blend * best_q)
            q_command = best_q
        for _ in range(max(2, int(round(args.preclose_hold_duration / sim.get_physics_dt())))):
            step(q_command, finger_value)
        midpoint, _, _, _, _ = pad_calibration()
        final_error = float(
            np.linalg.norm(
                (
                    grasp_point_w()
                    if target_point_w is None
                    else np.asarray(target_point_w, dtype=np.float64)
                )
                - midpoint
            )
        )
        return q_command, {
            "initial_error_m": errors[0],
            "iteration_errors_m": errors,
            "final_error_m": final_error,
            "joint_correction_rad": (q_command - start_q).tolist(),
        }

    approach_values = np.asarray(plan["approach_q"], dtype=np.float64)
    approach_stop_fraction = float(np.clip(args.approach_stop_fraction, 0.5, 1.0))
    approach_coordinate = approach_stop_fraction * (len(approach_values) - 1)
    approach_low = int(np.floor(approach_coordinate))
    approach_high = min(approach_low + 1, len(approach_values) - 1)
    approach_alpha = approach_coordinate - approach_low
    q_approach_stop = (
        (1.0 - approach_alpha) * approach_values[approach_low]
        + approach_alpha * approach_values[approach_high]
    )
    approach_prefix = np.concatenate(
        (approach_values[: approach_low + 1], q_approach_stop[None]), axis=0
    )

    free_space_centering_waypoints = None
    free_space_preclose_servo = None
    if args.preclose_mode == "free_space_waypoints":
        # Diagnostic alternative: calibrate a straight pad-midpoint path with
        # the object outside the workspace.
        cube.write_root_pose_to_sim(remote_cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
        free_space_target_w = measured_pickup_center - closing_midpoint_drift_w
        run_path(
            "calibration_visual_cfm_to_pregrasp",
            plan["transfer_to_pregrasp_q"],
            float(plan["transfer_to_pregrasp_duration"]),
            open_finger,
        )
        run_path(
            "calibration_approach",
            approach_prefix,
            float(plan["approach_duration"])
            * args.motion_time_scale
            * approach_stop_fraction,
            open_finger,
        )
        for _ in range(80):
            step(q_approach_stop, open_finger, render=False)
        free_space_start_w = pad_calibration()[0]
        free_space_centering_waypoints = [q_approach_stop.copy()]
        free_space_waypoint_reports = []
        q_free_space_centered = q_approach_stop.copy()
        for phase in np.linspace(0.0, 1.0, 13)[1:]:
            waypoint_target_w = (
                (1.0 - phase) * free_space_start_w + phase * free_space_target_w
            )
            q_free_space_centered, waypoint_report = center_gripper(
                q_free_space_centered,
                open_finger,
                target_point_w=waypoint_target_w,
            )
            free_space_centering_waypoints.append(q_free_space_centered.copy())
            free_space_waypoint_reports.append(waypoint_report)
        free_space_preclose_servo = {
            "initial_midpoint_w": free_space_start_w.tolist(),
            "target_midpoint_w": free_space_target_w.tolist(),
            "waypoint_count": len(free_space_centering_waypoints),
            "waypoint_reports": free_space_waypoint_reports,
            "final_error_m": free_space_waypoint_reports[-1]["final_error_m"],
            "joint_correction_rad": (
                q_free_space_centered - q_approach_stop
            ).tolist(),
        }
        for _ in range(240):
            step(q_start, open_finger, render=False)
        cube.write_root_pose_to_sim(cube_pose)
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
        for _ in range(80):
            step(q_start, open_finger, render=False)

    for _ in range(int(round(args.settle_duration / sim.get_physics_dt()))):
        step(q_start, open_finger)
    support_after_settle_w = cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    support_audit = {
        "primitive_box_colliders": bool(args.primitive_box_colliders),
        "reference_w": support_reference_w.tolist(),
        "after_reset_w": support_after_reset_w.tolist(),
        "after_settle_w": support_after_settle_w.tolist(),
        "reset_drift_m": float(np.linalg.norm(support_after_reset_w - support_reference_w)),
        "settle_drift_m": float(np.linalg.norm(support_after_settle_w - support_reference_w)),
        "maximum_allowed_drift_m": float(args.maximum_pre_task_support_drift),
    }
    print(json.dumps({"pre_task_support_audit": support_audit}, indent=2), flush=True)
    if support_audit["settle_drift_m"] > float(args.maximum_pre_task_support_drift):
        raise RuntimeError(f"pre-task support stability audit failed: {support_audit}")
    task_started = True
    task_contact_audit_active[0] = True
    native_monitor_active[0] = True
    recording_active[0] = True
    timeseries_active[0] = args.timeseries_output is not None
    run_path(
        "visual_cfm_to_pregrasp",
        plan["transfer_to_pregrasp_q"],
        float(plan["transfer_to_pregrasp_duration"]),
        open_finger,
    )
    run_path(
        "approach",
        approach_prefix,
        float(plan["approach_duration"])
        * args.motion_time_scale
        * approach_stop_fraction,
        open_finger,
    )
    if args.preclose_mode == "online":
        recording_phase[0] = "preclose_centering"
        current_phase[0] = "preclose_centering"
        preclose_feedforward_target_w = (
            grasp_point_w().copy() - closing_midpoint_drift_w
        )
        q_grasp_execution, preclose_servo = center_gripper(
            q_contact,
            open_finger,
            target_point_w=preclose_feedforward_target_w,
        )
        preclose_servo["mode"] = "online_geometric_correction"
    else:
        q_grasp_execution = np.asarray(q_free_space_centered, dtype=np.float64)
        preclose_path = np.asarray(free_space_centering_waypoints, dtype=np.float64)
        run_path(
            "preclose_centering",
            preclose_path,
            float(args.preclose_correction_duration),
            open_finger,
            smooth_time=True,
        )
        preclose_servo = {
            **free_space_preclose_servo,
            "mode": "free_space_geometric_calibration",
            "approach_stop_fraction": approach_stop_fraction,
            "final_error_m": float(
                np.linalg.norm(grasp_point_w() - pad_calibration()[0])
            ),
        }
    preclose_geometry = grasp_geometry()
    available_centering_margin = 0.5 * (
        float(preclose_geometry["surface_gap_m"])
        - float(preclose_geometry["object_width_along_closing_axis_m"])
    )
    preclose_centered = bool(
        available_centering_margin >= 0.0
        and abs(float(preclose_geometry["axial_center_offset_m"]))
        <= available_centering_margin + 0.0005
        and float(preclose_geometry["lateral_center_offset_m"])
        <= float(args.maximum_preclose_lateral_error)
    )
    preclose_clear = bool(
        float(preclose_geometry["surface_gap_m"])
        >= float(preclose_geometry["object_width_along_closing_axis_m"])
    )
    print(
        json.dumps(
            {"preclose_servo": preclose_servo, "preclose_geometry": preclose_geometry},
            indent=2,
        ),
        flush=True,
    )
    if not (preclose_centered and preclose_clear):
        raise RuntimeError(
            "object is not centered inside the open fingers before closing: "
            f"{preclose_geometry}"
        )
    native_monitor_active[0] = True
    contact_phase_active[0] = True
    run_finger_phase(
        "close", q_grasp_execution, args.close_duration, open_finger, grasp_finger
    )
    if args.skip_gap_servo:
        close_gap_servo = {
            "skipped": True,
            "reason": "use the contact-validated fixed finger command",
            "command_rad": float(grasp_finger),
        }
    else:
        grasp_finger, close_gap_servo = servo_pad_gap(
            q_grasp_execution,
            grasp_finger,
            float(preclose_geometry["object_width_along_closing_axis_m"])
            - float(args.preload_gap),
            track_object_width=True,
        )
    # Do not reposition the arm after first contact. Post-contact Cartesian
    # centering injects lateral energy and is a common source of object launch.
    recording_phase[0] = "soft_contact_hold"
    current_phase[0] = "soft_contact_hold"
    q_grasp_closed = q_grasp_execution.copy()
    closed_grasp_servo = {
        "skipped_for_native_contact": True,
        "reason": "avoid post-contact Cartesian correction",
    }
    post_center_gap_servo = {
        "skipped_for_native_contact": True,
        "reason": "first gap servo command retained as preload",
    }
    for _ in range(
        max(2, int(round(args.contact_hold_duration / sim.get_physics_dt())))
    ):
        step(q_grasp_closed, grasp_finger)
    grasp_geometry_audit = grasp_geometry()
    print(
        json.dumps(
            {
                "closed_grasp_servo": closed_grasp_servo,
                "close_gap_servo": close_gap_servo,
                "post_center_gap_servo": post_center_gap_servo,
                "grasp_geometry": grasp_geometry_audit,
            },
            indent=2,
        ),
        flush=True,
    )
    if (
        abs(float(grasp_geometry_audit["axial_center_offset_m"]))
        > float(args.maximum_contact_center_error)
        or float(grasp_geometry_audit["lateral_center_offset_m"])
        > float(args.maximum_contact_center_error)
    ):
        raise RuntimeError(
            "closed fingers are not centered on the object: "
            f"{grasp_geometry_audit}"
        )
    closed_position = cube.data.root_pos_w[0].cpu().numpy().copy()
    attachment_precondition_error = None
    if args.grasp_mode == "attachment_constraint":
        attachment_precondition_error = create_grasp_attachment(grasp_geometry_audit)
        if attachment_precondition_error > 0.015:
            raise RuntimeError(
                f"grasp attachment rejected: object is {attachment_precondition_error:.4f} m from grasp zone"
            )
        for _ in range(20):
            step(q_grasp_closed, grasp_finger)
    post_attachment_geometry = grasp_geometry()
    post_attachment_center_error = float(
        np.linalg.norm(
            np.asarray(post_attachment_geometry["grasp_point_w"])
            - np.asarray(post_attachment_geometry["pad_midpoint_w"])
        )
    )
    native_grasp_contacts = native_contact_summary()
    if int(native_grasp_contacts["non_pad_object_contacts"]) > 0:
        raise RuntimeError(
            f"native grasp rejected: non-pad gripper/object contact: {native_grasp_contacts}"
        )
    if native_grasp_contacts["target_impulse_within_limit"] is False:
        raise RuntimeError(
            f"native grasp rejected: target impulse limit exceeded: {native_grasp_contacts}"
        )
    if post_attachment_center_error > float(args.maximum_contact_center_error):
        raise RuntimeError(
            f"native contact shifted object out of the fingers by "
            f"{post_attachment_center_error:.4f} m"
        )
    lift_execution_q = np.asarray(plan["lift_q"], dtype=np.float64).copy()
    correction = q_grasp_closed - q_contact
    correction_weight = np.linspace(1.0, 0.0, len(lift_execution_q))[:, None]
    lift_execution_q += correction_weight * correction[None]
    run_path(
        "lift", lift_execution_q,
        float(plan["lift_duration"]) * args.lift_time_scale,
        grasp_finger,
        smooth_time=True,
    )
    lifted_position = cube.data.root_pos_w[0].cpu().numpy().copy()
    lifted_grasp_geometry = grasp_geometry()
    lifted_center_error = float(
        np.linalg.norm(
            np.asarray(lifted_grasp_geometry["grasp_point_w"])
            - np.asarray(lifted_grasp_geometry["pad_midpoint_w"])
        )
    )
    if args.stop_after_lift:
        peak_lift_m = float(phase_max_z.get("lift", closed_position[2]) - closed_position[2])
        lift_m = float(lifted_position[2] - closed_position[2])
        collisions = collision_summary()
        native_contacts_after_lift = native_contact_summary()
        attachment_joint_created = stage.GetPrimAtPath(attachment_path).IsValid()
        report = {
            "method_version": "native-contact-v1-experimental-soft-pinch",
            "scene": str(args.usd.resolve()),
            "plan": str(args.plan.resolve()),
            "gravity_enabled_for_entire_task": True,
            "object_teleport_after_task_start": False,
            "object_pin_after_task_start": False,
            "grasp_mode": args.grasp_mode,
            "native_contact_grasp": args.grasp_mode == "contact",
            "filtered_gripper_body_paths": filtered_gripper_body_paths,
            "support_filtered_gripper_body_paths": support_filtered_gripper_body_paths,
            "gripper_object_contact_filtered": False,
            "attachment_joint_created": attachment_joint_created,
            "attachment_precondition_error_m": attachment_precondition_error,
            "cube_mass_kg": float(args.cube_mass),
            "pre_task_support_audit": support_audit,
            "native_pad_contact": {
                "shape": "thin_box",
                "contact_offset_m": float(args.pad_contact_offset),
                "compliant_stiffness_n_m": float(args.compliant_contact_stiffness),
                "compliant_damping_n_s_m": float(args.compliant_contact_damping),
                "lift_time_scale": float(args.lift_time_scale),
                "gripper_effort_limit_nm": float(args.gripper_effort_limit),
                "gripper_velocity_limit_rad_s": float(args.gripper_velocity_limit),
                "gripper_stiffness_nm_rad": float(args.gripper_stiffness),
                "gripper_damping_nm_s_rad": float(args.gripper_damping),
            },
            "cube_yaw_rad": yaw,
            "reference_pad_normal_w": pad_normal.tolist(),
            "pad_face_calibration": pad_face_calibration,
            "aperture_calibration": aperture_samples,
            "grasp_finger_joint_rad": grasp_finger,
            "gripper_sequence": {
                "initial_open_rad": open_finger,
                "approach_rad": open_finger,
                "grasp_hold_rad": grasp_finger,
            },
            "grasp_geometry": grasp_geometry_audit,
            "post_attachment_geometry": post_attachment_geometry,
            "post_attachment_center_error_m": post_attachment_center_error,
            "lifted_grasp_geometry": lifted_grasp_geometry,
            "lifted_center_error_m": lifted_center_error,
            "preclose_geometry": preclose_geometry,
            "preclose_servo": preclose_servo,
            "closed_grasp_servo": closed_grasp_servo,
            "close_gap_servo": close_gap_servo,
            "post_center_gap_servo": post_center_gap_servo,
            "native_contact_audit": native_contacts_after_lift,
            "maximum_cube_linear_speed_m_s": maximum_cube_linear_speed[0],
            "maximum_cube_angular_speed_rad_s": maximum_cube_angular_speed[0],
            "cube_linear_speed_limit_m_s": float(args.maximum_cube_linear_speed),
            "cube_angular_speed_limit_rad_s": float(args.maximum_cube_angular_speed),
            "collision_audit": collisions,
            "closed_cube_w": closed_position.tolist(),
            "lifted_cube_w": lifted_position.tolist(),
            "lift_m": lift_m,
            "peak_lift_m": peak_lift_m,
            "grasp_success": bool(
                lift_m >= 0.05
                and grasp_geometry_audit["between_fingers"]
                and post_attachment_center_error
                <= float(args.maximum_contact_center_error)
                and lifted_center_error <= 0.012
                and native_contacts_after_lift["contact_force_sensor"]
                ["bilateral_contact_observed"]
                and native_contacts_after_lift["non_pad_object_contacts"] == 0
                and native_contacts_after_lift["target_impulse_within_limit"]
                is not False
                and not attachment_joint_created
                and collisions["collision_free"]
            ),
            "phase_max_z_w": phase_max_z,
            "recording": {
                "frames": int(recording_frame[0]),
                "fps": float(args.record_fps),
                "video": None if video_path is None else str(video_path),
                "gif": None if gif_path is None else str(gif_path),
            },
            "samples": samples,
        }
        if video_writer is not None:
            video_writer.close()
        if gif_writer is not None:
            gif_writer.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
        return
    run_path(
        "transfer_to_preplace",
        plan["transfer_to_preplace_q"],
        float(plan["transfer_to_preplace_duration"]) * args.motion_time_scale,
        grasp_finger,
    )
    run_path(
        "descend", plan["descend_q"],
        float(plan["descend_duration"]) * args.motion_time_scale, grasp_finger,
    )
    if "terminal_place_correction_q" in plan:
        run_path(
            "terminal_pose_correction",
            plan["terminal_place_correction_q"],
            float(plan["terminal_place_correction_duration"]),
            grasp_finger,
        )
    run_path(
        "place_hold", np.asarray(plan["q_place"])[None],
        args.place_hold_duration, grasp_finger,
    )
    placed_before_release = cube.data.root_pos_w[0].cpu().numpy().copy()
    place_hold_grasp_geometry = grasp_geometry()
    pre_release_place_error = float(
        np.linalg.norm(placed_before_release - measured_place_center)
    )
    if args.grasp_mode == "attachment_constraint" and stage.GetPrimAtPath(attachment_path).IsValid():
        # The object has support at the target pose.  Release the high-level
        # grasp first, then open smoothly back to the exact initial aperture.
        stage.RemovePrim(attachment_path)
        for _ in range(20):
            step(np.asarray(plan["q_place"]), grasp_finger)
        run_finger_phase(
            "release",
            np.asarray(plan["q_place"]),
            args.release_duration,
            grasp_finger,
            open_finger,
        )
    else:
        run_finger_phase(
            "release",
            np.asarray(plan["q_place"]),
            args.release_duration,
            grasp_finger,
            open_finger,
        )
    if "terminal_place_correction_q" in plan:
        run_path(
            "post_release_terminal_retract",
            np.asarray(plan["terminal_place_correction_q"])[::-1],
            float(plan["terminal_place_correction_duration"]),
            open_finger,
        )
    run_path(
        "post_release_retract",
        np.asarray(plan["descend_q"])[::-1],
        float(plan["descend_duration"]) * args.motion_time_scale,
        open_finger,
    )
    settle_q = np.asarray(plan["q_preplace"])
    for _ in range(int(round(args.settle_duration / sim.get_physics_dt()))):
        step(settle_q, open_finger)

    final_position = cube.data.root_pos_w[0].cpu().numpy().copy()
    final_speed = float(torch.linalg.norm(cube.data.root_lin_vel_w[0]).item())
    lift_m = float(lifted_position[2] - closed_position[2])
    peak_lift_m = float(phase_max_z.get("lift", closed_position[2]) - closed_position[2])
    place_error = float(np.linalg.norm(final_position - measured_place_center))
    collisions = collision_summary()
    native_contacts_complete = native_contact_summary()
    attachment_joint_created = stage.GetPrimAtPath(attachment_path).IsValid()
    report = {
        "method_version": "native-contact-v2-complete-pick-place",
        "scene": str(args.usd.resolve()),
        "plan": str(args.plan),
        "gravity_enabled_for_entire_task": True,
        "object_teleport_after_task_start": False,
        "object_pin_after_task_start": False,
        "robot_base_pinned": True,
        "grasp_mode": args.grasp_mode,
        "native_contact_grasp": args.grasp_mode == "contact",
        "attachment_joint_created": attachment_joint_created,
        "filtered_gripper_body_paths": filtered_gripper_body_paths,
        "gripper_object_contact_filtered": args.grasp_mode == "attachment_constraint",
        "attachment_precondition_error_m": attachment_precondition_error,
        "pickup_center_w": pickup_center.tolist(),
        "calibrated_pickup_center_w": calibrated_pickup.tolist(),
        "planned_to_calibrated_pickup_error_m": float(
            np.linalg.norm(calibrated_pickup - pickup_center)
        ),
        "open_pad_gap_m": pad_gap,
        "reference_pad_normal_w": pad_normal.tolist(),
        "pad_face_calibration": pad_face_calibration,
        "aperture_calibration": aperture_samples,
        "grasp_finger_joint_rad": grasp_finger,
        "gripper_sequence": {
            "initial_open_rad": open_finger,
            "approach_rad": open_finger,
            "grasp_hold_rad": grasp_finger,
            "release_target_rad": open_finger,
            "final_hold_rad": open_finger,
        },
        "cube_size_m": float(args.cube_size),
        "cube_mass_kg": float(args.cube_mass),
        "pre_task_support_audit": support_audit,
        "cube_static_friction": float(args.cube_static_friction),
        "cube_dynamic_friction": float(args.cube_dynamic_friction),
        "preclose_geometry": preclose_geometry,
        "preclose_servo": preclose_servo,
        "closed_grasp_servo": closed_grasp_servo,
        "grasp_geometry": grasp_geometry_audit,
        "post_attachment_geometry": post_attachment_geometry,
        "post_attachment_center_error_m": post_attachment_center_error,
        "lifted_grasp_geometry": lifted_grasp_geometry,
        "lifted_center_error_m": lifted_center_error,
        "place_hold_grasp_geometry": place_hold_grasp_geometry,
        "native_contact_audit": native_contacts_complete,
        "collision_audit": collisions,
        "place_center_w": place_center.tolist(),
        "physical_place_center_w": measured_place_center.tolist(),
        "closed_cube_w": closed_position.tolist(),
        "lifted_cube_w": lifted_position.tolist(),
        "placed_before_release_w": placed_before_release.tolist(),
        "pre_release_place_error_m": pre_release_place_error,
        "final_cube_w": final_position.tolist(),
        "lift_m": lift_m,
        "peak_lift_m": peak_lift_m,
        "phase_max_z_w": phase_max_z,
        "place_error_m": place_error,
        "final_speed_m_s": final_speed,
        "grasp_success": bool(
            lift_m >= 0.05
            and grasp_geometry_audit["between_fingers"]
            and post_attachment_center_error <= float(args.maximum_contact_center_error)
            and lifted_center_error <= 0.015
            and native_contacts_complete["contact_force_sensor"]
            ["bilateral_contact_observed"]
            and not attachment_joint_created
            and collisions["collision_free"]
        ),
        "place_success": bool(place_error <= 0.03 and final_speed <= 0.03),
        "overall_success": bool(
            lift_m >= 0.05
            and place_error <= 0.03
            and final_speed <= 0.03
            and grasp_geometry_audit["between_fingers"]
            and post_attachment_center_error <= float(args.maximum_contact_center_error)
            and lifted_center_error <= 0.015
            and native_contacts_complete["contact_force_sensor"]
            ["bilateral_contact_observed"]
            and not attachment_joint_created
            and collisions["collision_free"]
        ),
        "recording": {
            "frames": int(recording_frame[0]),
            "fps": float(args.record_fps),
            "video": None if video_path is None else str(video_path),
            "gif": None if gif_path is None else str(gif_path),
        },
        "samples": samples,
    }
    if video_writer is not None:
        video_writer.close()
    if gif_writer is not None:
        gif_writer.close()
    flush_timeseries()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=True)
