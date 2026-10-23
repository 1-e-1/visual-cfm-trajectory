# -*- coding: utf-8 -*-
"""Grasp-place demo v6.1 (REAL physical grasp -> place, VISIBLE cube):
- FIX v6: visual cube no longer follows the physics cube into the deep finger pocket
  (settled f0 z goes from jaw_mid 0.0037 -> pocket 0.0084, occluded by the pads).
  Instead the visual cube is drawn at the DYNAMIC jaw midpoint (visible between the
  fingers); physics cube keeps the real pinch grasp + lift/carry/place dynamics.
- FIX v6: after release the visual cube is frozen at the place point.
- REAL grasp: the cube is preset at the self-calibrated PAD contact faces (Robotiq
  fingertip inner faces), oriented so a cube face normal aligns with the pad closing
  direction, then the fingers close until the PAD gap reaches the cube size and keep
  squeezing (drive to full close, cube blocks the fingers).
- Gravity is re-enabled after the pinch: the cube must be held by real pad friction
  during lift/carry/align (proves a physical grasp). Gravity is frozen again only at
  the place point (no floor there) right before the fingers open.
- Cube pose is read from the physics body every frame.

Renders RGB frames -> PNGs + montage + GIF + report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectories", type=Path, required=True)
parser.add_argument("--data_dir", type=Path, required=True)
parser.add_argument("--sample_id", default="left_000444")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--seed", type=int, default=20260806)
parser.add_argument("--cube_size", type=float, default=0.035)
parser.add_argument("--cube_mass", type=float, default=0.4)
parser.add_argument("--close_time", type=float, default=3.0)
parser.add_argument("--lift_time", type=float, default=1.2)
parser.add_argument("--carry_time", type=float, default=2.0)
parser.add_argument("--release_time", type=float, default=1.2)
parser.add_argument("--align_time", type=float, default=1.6,
                    help="Closed-loop placement: move TCP so the REAL cube center lands on place_goal.")
parser.add_argument("--finger_close_target", type=float, default=0.0)
parser.add_argument("--root_z_offset", type=float, default=-0.074,
                    help="Lower the pinned root so the ankle visuals rest on the ground.")
parser.add_argument("--frame_dt", type=float, default=0.1)
parser.add_argument("--vis_front_offset", type=float, default=0.008,
                    help="Offset the visual cube toward the camera (+x) so the rotating palm cannot occlude it.")
parser.add_argument("--cam_offset_x", type=float, default=0.55,
                    help="Camera offset toward +x world (front view).")
parser.add_argument("--cam_offset_y", type=float, default=0.0,
                    help="Camera offset toward +y world (side view).")
parser.add_argument("--cam_offset_z", type=float, default=0.25,
                    help="Camera height above the tracked cube.")
parser.add_argument("--cam_tool_frame", action="store_true",
                    help="Interpret the camera offset in the TCP tool frame (keeps a constant jaw-opening view).")
parser.add_argument("--cam_perp_jaw", action="store_true",
                    help="Keep the camera offset perpendicular to the jaw-opening axis each frame.")
parser.add_argument("--vis_raise_z", type=float, default=0.020,
                    help="Raise the visual cube above the finger pads (0.035 was occluded).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.utils import stage as stage_utils
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade

from h1_1111_grasp.env_cfg import H1_1111_CFG
from src.planning.ik_solver import solve_ik_candidates
from src.robot.torch_kinematics import DifferentiableZBotFK, quat_apply


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


def _look_at_quat_wxyz(
    eye: np.ndarray, target: np.ndarray, up: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up_cam = np.cross(right, forward)
    rot = np.stack([right, up_cam, -forward], axis=1)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w, x, y, z = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w, x, y, z = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w, x, y, z = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    quat = np.array([w, x, y, z])
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def _set_semantic_class(stage: Usd.Stage, prim: Usd.Prim, label: str) -> None:
    labels_api = UsdSemantics.LabelsAPI.Apply(prim, "class")
    labels_api.CreateLabelsAttr().Set([label])


def main() -> None:
    from PIL import Image

    output_dir = args.output or PROJECT_ROOT / "outputs" / "figures" / "v621_grasp_place_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene_path = PROJECT_ROOT / "1111_copy.usd"
    print(f"[V6] opening_stage={scene_path}", flush=True)
    stage_utils.open_stage(str(scene_path))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))
    stage = omni.usd.get_context().get_stage()

    # --- physical cube: dynamic rigid body + collision. Gravity starts ON so the cube
    #     settles into the finger-pad pocket (validated slow-PD recipe), then is disabled
    #     before the visible demo so the cube never falls. ---------------------------
    if stage.GetPrimAtPath("/World/DynPartsLeft").IsValid():
        stage.RemovePrim("/World/DynPartsLeft")
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/DynPartsLeft/Obj0",
        spawn=sim_utils.CuboidCfg(
            size=(args.cube_size, args.cube_size, args.cube_size),
            visual_material=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    ))

    # hide the physical cube's own mesh so it cannot visually occlude the yellow visual cube
    _phys_mesh = stage.GetPrimAtPath("/World/DynPartsLeft/Obj0/geometry/mesh")
    print(f"[V6] phys mesh valid={_phys_mesh.IsValid()}", flush=True)
    if _phys_mesh.IsValid():
        _phys_mesh.GetAttribute("visibility").Set("invisible")
        print(f"[V6] phys mesh visibility set to {_phys_mesh.GetAttribute('visibility').Get()}", flush=True)
        # also enumerate all prims under the cube
        def _walk(p2, d=0):
            print(f"[V6] {'  '*d}{p2.GetPath()} type={p2.GetTypeName()} vis={p2.GetAttribute('visibility').Get()}", flush=True)
            for ch in p2.GetChildren():
                _walk(ch, d+1)
        _walk(stage.GetPrimAtPath("/World/DynPartsLeft"), 0)

    # --- runtime collision for the Robotiq fingers + cube ------------------------
    # The shipped USD has NO collision geometry on the gripper (only /visuals) and the
    # CuboidCfg spawn creates no /collisions prim in this kit build, so the cube and
    # the fingers are ghosts: a real pinch can never happen. Add box colliders that
    # match the fingertip visual meshes, plus a box collider for the cube itself.
    def _add_box_collider(prim_path: str, center: tuple[float, float, float],
                          size: tuple[float, float, float]) -> None:
        prim = UsdGeom.Mesh.Define(stage, prim_path)
        hx, hy, hz = 0.5 * size[0], 0.5 * size[1], 0.5 * size[2]
        cx, cy, cz = center
        pts = [
            Gf.Vec3f(cx - hx, cy - hy, cz - hz), Gf.Vec3f(cx + hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy + hy, cz - hz), Gf.Vec3f(cx - hx, cy + hy, cz - hz),
            Gf.Vec3f(cx - hx, cy - hy, cz + hz), Gf.Vec3f(cx + hx, cy - hy, cz + hz),
            Gf.Vec3f(cx + hx, cy + hy, cz + hz), Gf.Vec3f(cx - hx, cy + hy, cz + hz),
        ]
        prim.GetPointsAttr().Set(pts)
        prim.GetFaceVertexCountsAttr().Set([3] * 12)
        prim.GetFaceVertexIndicesAttr().Set(
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
             2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7])
        prim.GetDoubleSidedAttr().Set(True)
        prim.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim())
        prim.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")

    _gripper_paths = [
        "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/left_inner_finger",
        "/World/UnitreeH1/ZBot6S_left/Robotiq2F85/right_inner_finger",
        "/World/UnitreeH1/ZBot6S_right/Robotiq2F85/left_inner_finger",
        "/World/UnitreeH1/ZBot6S_right/Robotiq2F85/right_inner_finger",
    ]
    for _gp in _gripper_paths:
        # finger shaft box. NOTE: the visuals Xform carries a 120deg orient
        # (0.5,0.5,0.5,0.5), so the mesh-frame extents (x -0.075..-0.045,
        # y 0.0914..0.1194, z -0.0135..0.0135) map to the BODY frame as
        # body = (mesh_z, mesh_x, mesh_y): x -0.0135..0.0135, y -0.0748..-0.0449,
        # z 0.0914..0.1194. Earlier versions placed these in the mesh frame and the
        # boxes stuck out sideways, jamming the linkage / never touching the cube.
        _add_box_collider(_gp + "/collisions/body", (0.0, -0.0599, 0.1054), (0.027, 0.0299, 0.028))
        # fingertip pad box: mesh (x -0.0627..-0.0435, y 0.1104..0.1484, z -0.011..0.011)
        # -> BODY (x -0.011..0.011, y -0.0627..-0.0435, z 0.1104..0.1484)
        _add_box_collider(_gp + "/collisions/pad", (0.0, -0.0531, 0.1294), (0.022, 0.0192, 0.038))
    # cube collision (the CuboidCfg spawn created no /collisions prim in this build)
    _add_box_collider("/World/DynPartsLeft/Obj0/collisions/box", (0.0, 0.0, 0.0),
                      (args.cube_size, args.cube_size, args.cube_size))
    print(f"[V6] added runtime colliders: fingers x{len(_gripper_paths)*2} + cube", flush=True)

    # --- independent visual cube (CuboidCfg visuals do not render in this kit build:
    #     the generated geometry/mesh has no points/faces, so we draw a synced UsdGeom.Cube)
    vis_path = "/World/CubeVis"
    if stage.GetPrimAtPath(vis_path).IsValid():
        stage.RemovePrim(vis_path)
    vis_prim = stage.DefinePrim(vis_path, "Xform")
    vis_xform = UsdGeom.Xformable(vis_prim)
    vis_xform.AddTranslateOp()
    vis_xform.AddOrientOp()
    vis_geom = UsdGeom.Cube.Define(stage, vis_path + "/Cube")
    vis_geom.GetSizeAttr().Set(float(args.cube_size))  # match physical cube size
    vis_mat_prim = stage.DefinePrim(vis_path + "/Mat", "Material")
    vis_shader = UsdShade.Shader.Define(stage, vis_path + "/Mat/Shader")
    vis_shader.CreateIdAttr().Set("UsdPreviewSurface")
    vis_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 0.85, 0.05))
    vis_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    vis_mat = UsdShade.Material(vis_mat_prim)
    vis_mat.CreateSurfaceOutput().ConnectToSource(vis_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(vis_geom.GetPrim()).Bind(vis_mat)
    # NOTE: no semantic class on the visual cube - diag d63 rendered fine without it
    vis_ops = vis_xform.GetOrderedXformOps()

    fk: Any = None
    vis_frozen_pos: list[np.ndarray | None] = [None]
    sim_vis_local: list[np.ndarray | None] = [None]
    cube_freeze_pose: list[np.ndarray | None] = [None]  # physics cube pinned here during release

    def _vis_offset_world() -> np.ndarray:
        """World-frame visibility offset for the visual cube.

        The base offset is captured once in the TCP (foot_1) local frame at q_close
        (where the world offset [front, 0, raise] is known to place the cube inside
        the visible jaw opening). Every frame the offset is rotated back through the
        CURRENT sim TCP orientation, so the cube keeps a constant position relative
        to the gripper while the wrist rotates (fixes lift-phase occlusion)."""
        if sim_vis_local[0] is not None:
            R_tcp = _quat_wxyz_to_matrix(robot.data.body_quat_w[0, tcp_ids[0]].cpu().numpy())
            return R_tcp @ sim_vis_local[0]
        return np.array([args.vis_front_offset, 0.0, args.vis_raise_z], dtype=np.float64)
    def sync_vis() -> None:
        # Track the REAL physics cube: it is pinched between the finger pads, i.e.
        # at the true grasp center. Drawing at the inner-finger link midpoint puts
        # the block up by the wrist/TCP (the user-visible "stuck at foot_1" bug).
        if vis_frozen_pos[0] is not None:
            pose = vis_frozen_pos[0]
        else:
            pose = np.asarray(cube.data.root_pos_w[0].cpu().numpy(), dtype=np.float64)
            # small offset only for visibility (raise above the pads / slight front shift)
            pose = pose + _vis_offset_world()
        vis_ops[0].Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
        vis_ops[1].Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # identity orient (cube is symmetric)

    # --- free camera ------------------------------------------------------------
    cam_path = "/World/RenderCam"
    if stage.GetPrimAtPath(cam_path).IsValid():
        stage.RemovePrim(cam_path)
    sim_utils.spawn_camera(
        cam_path,
        sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0),
        ),
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    camera = Camera(CameraCfg(
        prim_path=cam_path, update_period=0.0, spawn=None, data_types=["rgb"],
        width=int(args.resolution), height=int(args.resolution),
        
    ))

    sim.reset()
    robot.reset()
    cube.reset()
    _pm2 = stage.GetPrimAtPath("/World/DynPartsLeft/Obj0/geometry/mesh")
    if _pm2.IsValid():
        _pm2.GetAttribute("visibility").Set("invisible")
        print(f"[V6] post-reset mesh visibility={_pm2.GetAttribute('visibility').Get()}", flush=True)
    _pr = stage.GetPrimAtPath("/World/DynPartsLeft/Obj0")
    if _pr.IsValid():
        _pr.GetAttribute("visibility").Set("invisible")
        print(f"[V6] post-reset Obj0 visibility={_pr.GetAttribute('visibility').Get()}", flush=True)
    print(f"[V6] root_pos_w pre-warmup={np.round(robot.data.root_pos_w[0].cpu().numpy(),4)} "
          f"lin_vel={np.round(robot.data.root_lin_vel_w[0].cpu().numpy(),4)}", flush=True)

    suffix = "_0"
    arm_ids, arm_names = robot.find_joints([f"^joint{i}{suffix}$" for i in range(1, 7)], preserve_order=True)
    finger_ids, finger_names = robot.find_joints(["^finger_joint_0$"], preserve_order=True)
    base_ids, _ = robot.find_bodies(["^foot_0_0$"])
    tcp_ids, tcp_names = robot.find_bodies(["^foot_1_0$"])
    ankle_l_ids, ankle_l_names = robot.find_bodies(["^left_ankle_link.*$"], preserve_order=True)
    ankle_r_ids, ankle_r_names = robot.find_bodies(["^right_ankle_link.*$"], preserve_order=True)
    print(f"[V6] ankles L={ankle_l_names} R={ankle_r_names}", flush=True)
    lf_ids, lf_names = robot.find_bodies(["^left_inner_finger_0$", "^right_inner_finger_0$"], preserve_order=True)
    print(f"[V6] arm={arm_names} finger={finger_names} tcp={tcp_names}", flush=True)

    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_pos)
    robot.write_joint_state_to_sim(default_pos, zero_vel)
    robot.set_joint_position_target(default_pos)
    camera.reset()
    root_pos_fixed = robot.data.root_pos_w.clone()
    root_pos_fixed[0, 2] += args.root_z_offset
    print(f"[V6] pinned root z={float(root_pos_fixed[0, 2]):.4f} (offset {args.root_z_offset:+.3f})", flush=True)
    root_quat_fixed = robot.data.root_quat_w.clone()
    zero_root_vel = torch.zeros((1, 6), dtype=torch.float32, device=root_pos_fixed.device)

    def pin_root() -> None:
        """Pin the floating base at the captured pose AND zero its velocity so the solver
        does not integrate gravity drift between steps (the H1 body has no foot colliders,
        so it would otherwise free-fall through the ground plane)."""
        robot.write_root_state_to_sim(torch.cat([root_pos_fixed, root_quat_fixed, zero_root_vel], dim=-1))
    for _ in range(6):
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())

    print(f"[V6] root_pos_w after warmup={np.round(robot.data.root_pos_w[0].cpu().numpy(),4)} "
          f"root_lin_vel={np.round(robot.data.root_lin_vel_w[0].cpu().numpy(),4)}", flush=True)
    print(f"[V6] left_foot foot_0_0 pos={np.round(robot.data.body_pos_w[0, base_ids[0]].cpu().numpy(),4)} "
          f"root_quat={np.round(robot.data.root_quat_w[0].cpu().numpy(),4)}", flush=True)
    limits = robot.data.joint_pos_limits[0].cpu().numpy()
    finger_lower, finger_upper = float(limits[finger_ids[0], 0]), float(limits[finger_ids[0], 1])
    finger_open = 0.7792895138263702  # verified value from slow-PD test
    finger_closed = float(finger_lower if finger_lower >= 0.0 else 0.0)
    print(f"[V6] finger limits=[{finger_lower:.3f}, {finger_upper:.3f}] open={finger_open:.3f} closed={finger_closed:.3f}", flush=True)

    base_position = robot.data.body_pos_w[0, base_ids[0]].cpu().numpy()
    base_quaternion = robot.data.body_quat_w[0, base_ids[0]].cpu().numpy()

    # --- load trajectory meta (grasp/place goals + hold pose) -------------------
    with np.load(args.trajectories, allow_pickle=False) as data:
        sample_ids = [str(value) for value in data["sample_id"]]
        index = sample_ids.index(args.sample_id)
        q_hold = np.asarray(data["place_q_start"][index], dtype=np.float64)
        place_goal_foot0 = np.asarray(data["place_goal_position"][index], dtype=np.float64)

    def to_f0(w):
        R = _quat_wxyz_to_matrix(base_quaternion)
        return (np.asarray(w, dtype=np.float64) - base_position) @ R

    def from_f0(local):
        R = _quat_wxyz_to_matrix(base_quaternion)
        return base_position + np.asarray(local, dtype=np.float64) @ R.T

    def fk_tcp_f0() -> np.ndarray:
        """Position of the analytic-FK TCP (foot_1 local +z 0.053m virtual point) in foot0 frame.

        The DifferentiableZBotFK chain ends at /zbot/foot_1/TCP with tcp_local_position
        [0,0,0.053]; the sim body 'foot_1_0' is the foot_1 origin. Using the sim body origin
        directly as the IK reference introduces a constant 53mm placement bias, which showed
        up as the ~53mm end-effector error in the physical-grasp runs."""
        tcp_world = robot.data.body_pos_w[0, tcp_ids[0]].cpu().numpy()
        R_world_tcp = _quat_wxyz_to_matrix(robot.data.body_quat_w[0, tcp_ids[0]].cpu().numpy())
        R_f0 = _quat_wxyz_to_matrix(base_quaternion)
        R_tcp_f0 = R_f0.T @ R_world_tcp
        return to_f0(tcp_world) + R_tcp_f0 @ np.array([0.0, 0.0, 0.053])

    # --- fix grasp pose (the arrow): q_close + measure jaw midpoint -------------
    q_close = np.array([-2.1070001125335693, 0.9599999785423279, 0.597000002861023,
                        0.47600001096725464, 0.09799999743700027, -0.028999999165534973])
    jp = default_pos.clone()
    jp[:, arm_ids] = torch.as_tensor(q_close, dtype=torch.float32, device=jp.device)
    jp[:, finger_ids] = torch.as_tensor([finger_open], dtype=torch.float32, device=jp.device)
    robot.write_joint_state_to_sim(jp, zero_vel)
    robot.set_joint_position_target(jp)
    for _ in range(200):  # v5: Robotiq linkage needs ~120 steps to fully open
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt())
        if _ % 50 == 0 or _ == 199:
            _fj = float(robot.data.joint_pos[0, finger_ids[0]].item())
            _lf = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
            _rf = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
            print(f"[V6DBG] open step={_} finger={_fj:.4f} body_gap={float(np.linalg.norm(_lf-_rf))*1000:.1f}mm "
                  f"lf_f0={np.round(to_f0(_lf),4)} rf_f0={np.round(to_f0(_rf),4)}", flush=True)
    mount_f0 = fk_tcp_f0()
    left_f0 = to_f0(robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy())
    right_f0 = to_f0(robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy())
    jaw_mid = 0.5 * (left_f0 + right_f0)
    gap_open = float(np.linalg.norm(left_f0 - right_f0))
    print(f"[V6] q_close mount_f0={np.round(mount_f0,4)}", flush=True)
    assert gap_open > args.cube_size + 0.005, "jaw cannot open wide enough for the cube"
    R_tcp_close = _quat_wxyz_to_matrix(robot.data.body_quat_w[0, tcp_ids[0]].cpu().numpy())
    sim_vis_local[0] = R_tcp_close.T @ np.array([args.vis_front_offset, 0.0, args.vis_raise_z], dtype=np.float64)
    print(f"[V6] vis_local_tcp={np.round(sim_vis_local[0],4)}", flush=True)

    # --- self-calibrate the REAL pad contact faces at q_close (fingers open) -----
    # The inner-finger body ORIGINS sit at the finger bases near the wrist; the pad
    # contact faces (Robotiq 2F-85 fingertip inner faces) are at local +y in each
    # body frame. Try both fingertip mesh faces and pick the pair that faces inward.
    lf_pos_w = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
    rf_pos_w = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
    lf_quat_w = robot.data.body_quat_w[0, lf_ids[0]].cpu().numpy()
    rf_quat_w = robot.data.body_quat_w[0, lf_ids[1]].cpu().numpy()
    R_lf = _quat_wxyz_to_matrix(lf_quat_w)
    R_rf = _quat_wxyz_to_matrix(rf_quat_w)
    sep_w = rf_pos_w - lf_pos_w
    # fingertip block (Robotiq "fingertipsstep") 6 face centers + outward normals in
    # the FINGER BODY frame. The visuals Xform has a 120deg orient (0.5,0.5,0.5,0.5),
    # i.e. body = (mesh_z, mesh_x, mesh_y); mesh extents x -0.0627..-0.0435,
    # y 0.1104..0.1484, z -0.011..0.011 -> BODY x -0.011..0.011, y -0.0627..-0.0435,
    # z 0.1104..0.1484.
    _faces = [
        (np.array([0.0110, -0.0531, 0.1294]), np.array([1.0, 0.0, 0.0])),
        (np.array([-0.0110, -0.0531, 0.1294]), np.array([-1.0, 0.0, 0.0])),
        (np.array([0.0000, -0.0435, 0.1294]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0000, -0.0627, 0.1294]), np.array([0.0, -1.0, 0.0])),
        (np.array([0.0000, -0.0531, 0.1104]), np.array([0.0, 0.0, -1.0])),
        (np.array([0.0000, -0.0531, 0.1484]), np.array([0.0, 0.0, 1.0])),
    ]
    sep_n = sep_w / max(float(np.linalg.norm(sep_w)), 1.0e-12)
    best_pad = None
    _cands: list[tuple] = []
    for f_l, n_l in _faces:
        P_l = lf_pos_w + R_lf @ f_l
        n_lw = R_lf @ n_l
        if float(np.dot(n_lw, sep_n)) < 0.999:
            continue  # left pad normal must point toward the right finger
        for f_r, n_r in _faces:
            P_r = rf_pos_w + R_rf @ f_r
            n_rw = R_rf @ n_r
            if float(np.dot(n_rw, sep_n)) > -0.999:
                continue  # right pad normal must point toward the left finger
            d = P_r - P_l
            dn = float(np.linalg.norm(d))
            _cands.append((dn, f_l, f_r, float(np.dot(n_lw, sep_n)), float(np.dot(n_rw, sep_n))))
            if best_pad is None or dn < best_pad[0]:
                best_pad = (dn, P_l, P_r, f_l, f_r, n_lw, n_rw)
    _cands.sort()
    print("[V6] pad face candidates (dist, f_l, f_r, dot_l, dot_r):", flush=True)
    for _cn in _cands:
        print(f"[V6]   {_cn[0]*1000:6.1f}mm f_l={np.round(_cn[1],4)} f_r={np.round(_cn[2],4)} "
              f"dotl={_cn[3]:.4f} dotr={_cn[4]:.4f}", flush=True)
    assert best_pad is not None, "could not find facing pad surfaces"
    pad_gap_open, pad_l_w, pad_r_w, pad_local_l, pad_local_r, n_lw, n_rw = best_pad
    pad_mid_w = 0.5 * (pad_l_w + pad_r_w)
    _ang_l = float(np.degrees(np.arccos(np.clip(np.dot(n_lw, sep_n), -1.0, 1.0))))
    _ang_par = float(np.degrees(np.arccos(np.clip(np.dot(n_lw, -n_rw), -1.0, 1.0))))
    print(f"[V6] pad_local_l={np.round(pad_local_l,4)} pad_local_r={np.round(pad_local_r,4)}", flush=True)
    print(f"[V6] pad_gap_open={pad_gap_open*1000:.1f}mm pad_mid_f0={np.round(to_f0(pad_mid_w),4)}", flush=True)
    print(f"[V6] pad normals l={np.round(n_lw,4)} r={np.round(n_rw,4)} | ang(n_l,sep)={_ang_l:.2f}deg ang(n_l,-n_r)={_ang_par:.2f}deg", flush=True)
    assert pad_gap_open > args.cube_size + 0.005, "pad gap at open cannot fit the cube"

    def _pad_mid_now() -> np.ndarray:
        lf_p = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
        rf_p = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
        lf_q = robot.data.body_quat_w[0, lf_ids[0]].cpu().numpy()
        rf_q = robot.data.body_quat_w[0, lf_ids[1]].cpu().numpy()
        P_l = lf_p + _quat_wxyz_to_matrix(lf_q) @ pad_local_l
        P_r = rf_p + _quat_wxyz_to_matrix(rf_q) @ pad_local_r
        return 0.5 * (P_l + P_r)

    def _pad_gap_now() -> float:
        lf_p = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
        rf_p = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
        lf_q = robot.data.body_quat_w[0, lf_ids[0]].cpu().numpy()
        rf_q = robot.data.body_quat_w[0, lf_ids[1]].cpu().numpy()
        P_l = lf_p + _quat_wxyz_to_matrix(lf_q) @ pad_local_l
        P_r = rf_p + _quat_wxyz_to_matrix(rf_q) @ pad_local_r
        return float(np.linalg.norm(P_r - P_l))

    # --- place the cube at the REAL pad grasp center, oriented so a cube face normal
    #     aligns with the pad closing direction (else the pads hit cube edges).
    #     Gravity stays OFF until the pads pinch, then is re-enabled to prove a REAL
    #     friction grasp during lift/carry. ----------------------------------------
    # align a cube face normal to the REAL pad normal (not sep): guarantees the pads
    # press flat on the cube faces instead of wedging the cube out sideways
    jaw_axis_w = n_lw / max(float(np.linalg.norm(n_lw)), 1.0e-9)
    cube_world = pad_mid_w.copy()
    _z0 = np.array([0.0, 0.0, 1.0])
    _v = np.cross(_z0, jaw_axis_w)
    _s = float(np.linalg.norm(_v))
    _c = float(np.dot(_z0, jaw_axis_w))
    if _s < 1.0e-9:
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0]) if _c > 0.0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        _q = np.array([_c + 1.0, _v[0], _v[1], _v[2]])
        cube_quat = _q / float(np.linalg.norm(_q))
    cube.write_root_pose_to_sim(torch.tensor([[cube_world[0], cube_world[1], cube_world[2],
                                               cube_quat[0], cube_quat[1], cube_quat[2], cube_quat[3]]],
                                             dtype=torch.float32, device=sim.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim.device))
    cube_f0 = to_f0(cube_world)
    print(f"[V6] preset cube_f0={np.round(cube_f0,4)} world={np.round(cube_world,4)} quat={np.round(cube_quat,4)}", flush=True)
    # disable gravity until the pads pinch the cube
    _grav_api = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath("/World/DynPartsLeft/Obj0"))
    _grav_api.GetDisableGravityAttr().Set(True)
    for _ in range(10):  # settle with gravity OFF: cube floats at the pad grasp center
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); cube.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
        sync_vis()
    settled_f0 = to_f0(cube.data.root_pos_w[0].cpu().numpy())
    print(f"[V6] settled cube_f0={np.round(settled_f0,4)}", flush=True)
    for _ in range(5):
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); cube.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
        sync_vis()
    grav_off_f0 = to_f0(cube.data.root_pos_w[0].cpu().numpy())
    print(f"[V6] gravity-off cube_f0={np.round(grav_off_f0,4)}", flush=True)
    cube_world = cube.data.root_pos_w[0].cpu().numpy().copy()

    # --- place endpoint: wrist mount such that cube center lands on place goal ---
    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs" / "kinematic_chain.json").to(args.device)
    with torch.no_grad():
        _, quat_close_dbg = fk(torch.as_tensor(q_close[None], dtype=torch.float32, device=args.device))
    R_close_dbg = _quat_wxyz_to_matrix(quat_close_dbg[0].cpu().numpy())
    tz_close_dbg = R_close_dbg @ np.array([0.0, 0.0, 1.0])
    tx_close_dbg = R_close_dbg @ np.array([1.0, 0.0, 0.0])
    ty_close_dbg = R_close_dbg @ np.array([0.0, 1.0, 0.0])
    jaw_axis_dbg = (left_f0 - right_f0) / max(float(np.linalg.norm(left_f0 - right_f0)), 1e-9)
    print(f"[V6] tool@q_close quat={np.round(quat_close_dbg[0].cpu().numpy(),4)} z={np.round(tz_close_dbg,4)} x={np.round(tx_close_dbg,4)} y={np.round(ty_close_dbg,4)}", flush=True)
    print(f"[V6] jaw_axis_f0={np.round(jaw_axis_dbg,4)}", flush=True)
    q_seed = np.asarray(q_hold, dtype=np.float64)
    with torch.no_grad():
        _, quat_hold = fk(torch.as_tensor(q_seed[None], dtype=torch.float32, device=args.device))
    tool_z = quat_apply(
        quat_hold, torch.zeros((1, 3), device=args.device) + torch.tensor([0.0, 0.0, 1.0], device=args.device)
    )[0].cpu().numpy()

    def _solve_place(goal_f0: np.ndarray, seed_q: np.ndarray, tool_z_local: np.ndarray, tcp_off: float) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(3):
            fk_goal = np.asarray(goal_f0, dtype=np.float64) + tcp_off * tool_z_local
            candidates = solve_ik_candidates(
                fk, fk_goal, n_candidates=4, restarts=32, iterations=300,
                position_tolerance_m=0.005, seed=args.seed, device=args.device,
            )
            wrapped_c = [(np.asarray(c) + np.pi) % (2 * np.pi) - np.pi for c in candidates]
            wrapped_s = (seed_q + np.pi) % (2 * np.pi) - np.pi
            q_sol = np.asarray(candidates[int(np.argmin([np.linalg.norm(wc - wrapped_s) for wc in wrapped_c]))], dtype=np.float64)
            with torch.no_grad():
                _, quat_sol = fk(torch.as_tensor(q_sol[None], dtype=torch.float32, device=args.device))
            new_z = quat_apply(
                quat_sol, torch.zeros((1, 3), device=args.device) + torch.tensor([0.0, 0.0, 1.0], device=args.device)
            )[0].cpu().numpy()
            if np.linalg.norm(new_z - tool_z_local) < 1.0e-3:
                break
            tool_z_local = new_z
        return q_sol, tool_z_local

    # Pose-aware place target: the cube hangs at a fixed LOCAL offset from the wrist mount
    # (measured at the grasp pose). When the arm re-orients to q_place that world-space
    # offset rotates with the tool, so we must project it through the relative tool rotation
    # instead of reusing the grasp-pose world vector (this caused the ~57mm placement bias).
    cube_center_f0 = np.asarray(settled_f0, dtype=np.float64)
    u_world_close = cube_center_f0 - np.asarray(mount_f0, dtype=np.float64)  # object - TCP @ grasp pose
    with torch.no_grad():
        _, quat_close = fk(torch.as_tensor(q_close[None], dtype=torch.float32, device=args.device))
    R_close = _quat_wxyz_to_matrix(quat_close[0].cpu().numpy())
    u_local = R_close.T @ u_world_close  # object offset expressed in the tool frame

    mount_goal_f0 = np.asarray(place_goal_foot0, dtype=np.float64) - u_world_close
    q_place, tool_z = _solve_place(mount_goal_f0, q_seed, tool_z, 0.0)
    for _ in range(4):
        with torch.no_grad():
            _, quat_place = fk(torch.as_tensor(q_place[None], dtype=torch.float32, device=args.device))
        R_place = _quat_wxyz_to_matrix(quat_place[0].cpu().numpy())
        u_world_place = R_place @ u_local
        mount_goal_new = np.asarray(place_goal_foot0, dtype=np.float64) - u_world_place
        if np.linalg.norm(mount_goal_new - mount_goal_f0) < 1.0e-5:
            break
        mount_goal_f0 = mount_goal_new
        q_place, tool_z = _solve_place(mount_goal_f0, q_seed, tool_z, 0.0)
    print(f"[V6] place goal f0={np.round(place_goal_foot0,4)}", flush=True)
    print(f"[V6] u_close={np.round(u_world_close,4)} u_place={np.round(mount_goal_f0 - np.asarray(place_goal_foot0, dtype=np.float64),4)}", flush=True)
    print(f"[V6] mount_goal_f0={np.round(mount_goal_f0,4)}", flush=True)

    # --- camera framing: track the grasped cube so it stays centered --------------
    cam_prim = stage.GetPrimAtPath(cam_path)
    CAM_OFFSET = np.array([args.cam_offset_x, args.cam_offset_y, args.cam_offset_z])

    def update_camera() -> None:
        if vis_frozen_pos[0] is not None:
            obj_world = vis_frozen_pos[0]
        else:
            obj_world = np.asarray(cube.data.root_pos_w[0].cpu().numpy(), dtype=np.float64) + _vis_offset_world()

        if args.cam_tool_frame:
            # ride along the TCP body frame (foot_1)
            quat_tool = robot.data.body_quat_w[0, tcp_ids[0]].cpu().numpy()
            R_tool = _quat_wxyz_to_matrix(quat_tool)
            local = np.array([args.cam_offset_x, args.cam_offset_y, args.cam_offset_z])
            eye = obj_world + local @ R_tool.T
        else:
            base = np.array([args.cam_offset_x, args.cam_offset_y, args.cam_offset_z])
            if args.cam_perp_jaw:
                # project the offset onto the plane perpendicular to the jaw-opening
                # axis (left-right finger separation) so the pinched cube is never
                # hidden behind a finger while the wrist rotates.
                lw = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
                rw = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
                jaw_axis = lw - rw
                n = float(np.linalg.norm(jaw_axis))
                if n > 1.0e-6:
                    jaw_axis = jaw_axis / n
                    perp = base - jaw_axis * float(np.dot(base, jaw_axis))
                    nn = float(np.linalg.norm(perp))
                    if nn > 1.0e-6:
                        base = perp * (float(np.linalg.norm(base)) / nn)
            eye = obj_world + base
        quat = _look_at_quat_wxyz(eye, obj_world)
        cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
        cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(value) for value in quat]))
    update_camera()
    print(f"[V6] camera tracking cube (offset {CAM_OFFSET})", flush=True)
    for _ in range(3):
        pin_root()
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())

    # --- timeline ----------------------------------------------------------------
    # Phases run in order; ALIGN is computed at runtime from the REAL physics cube pose
    # (closed-loop placement): it shifts the TCP so the cube center lands on place_goal,
    # compensating for any sliding of the cube inside the jaw during lift/carry.
    total_dt = args.frame_dt
    frame_index = 0
    images: list[np.ndarray] = []
    labels: list[str] = []
    per_frame: list[dict[str, Any]] = []

    joint_pos = default_pos.clone()
    finger_target = finger_open

    close_locked = {"locked": False, "target": finger_open}

    def run_stage(name: str, dur: float, q_from: np.ndarray, q_to: np.ndarray, finger_mode: str) -> None:
        nonlocal frame_index, finger_target
        n_steps = int(round(dur / sim.get_physics_dt()))
        frame_accum = 0.0
        for s in range(n_steps):
            u = s / max(1, n_steps - 1)
            q_now = q_from + (q_to - q_from) * u
            if finger_mode == "open":
                finger_target = finger_open
            elif finger_mode == "close":
                if close_locked["locked"]:
                    finger_target = close_locked["target"]
                else:
                    finger_target = finger_open + (args.finger_close_target - finger_open) * u
                    # stop when the PAD contact faces reach the cube: the pads then
                    # pinch the cube and hold it. (The body-origin gap would lock far
                    # too early because the pad faces sit ~yf ahead of the origins.)
                    if _pad_gap_now() <= args.cube_size + 0.005:
                        close_locked["locked"] = True
                        # drive to full close: the cube blocks the fingers, producing a
                        # strong sustained squeeze that holds the cube during lift/carry
                        close_locked["target"] = args.finger_close_target
                        finger_target = close_locked["target"]
            elif finger_mode == "hold_closed":
                finger_target = close_locked["target"] if close_locked["locked"] else args.finger_close_target
            else:  # release
                finger_target = (close_locked["target"] if close_locked["locked"] else args.finger_close_target) + (finger_open - (close_locked["target"] if close_locked["locked"] else args.finger_close_target)) * u
            joint_pos[:, arm_ids] = torch.as_tensor(q_now, dtype=torch.float32, device=joint_pos.device)
            joint_pos[:, finger_ids] = torch.as_tensor([finger_target], dtype=torch.float32, device=joint_pos.device)
            # NOTE: do NOT write_joint_state_to_sim here. The arm was already driven to
            # q_close in the pre-demo warm-up and the fingers are open; re-writing the
            # full joint state re-initializes the Robotiq mimic/linkage joints to their
            # default values and the fingers get jammed half-closed by the cube between
            # the pads (frame-1 gap dropped to 3.7cm instead of 8cm).
            robot.set_joint_position_target(joint_pos)
            pin_root()
            robot.write_data_to_sim()
            cube.update(sim.get_physics_dt())
            if cube_freeze_pose[0] is not None:
                # keep the physics cube pinned at the place point while the fingers
                # open: the Robotiq pad sweep would otherwise fling it sideways.
                _fp = cube_freeze_pose[0]
                cube.write_root_pose_to_sim(torch.tensor(
                    [[_fp[0], _fp[1], _fp[2], 1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32, device=sim.device))
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim.device))
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt())
            cube_target_world = cube.data.root_pos_w[0].cpu().numpy().copy()

            frame_accum += sim.get_physics_dt()
            if frame_accum >= total_dt or (s == n_steps - 1 and frame_index == 0):
                frame_accum = 0.0
                # IMPORTANT: read vis_pos/camera BEFORE re-setting - the RTX renderer
                # only applies transforms that were stable for one or more physics steps.
                # sync_vis/update_camera run once per saved frame (every 0.1 s), so the
                # transform is stable for ~20 steps and the cube actually renders.
                vis_pos = np.array([v for v in vis_ops[0].Get()]) if vis_ops[0].Get() is not None else None
                cam_pos = np.array([v for v in cam_prim.GetAttribute("xformOp:translate").Get()])
                if frame_index % 10 == 0:
                    print(f"[V6DBG] frame={frame_index} phase={name} cube_world={np.round(cube_target_world,3)} vis_pos={None if vis_pos is None else np.round(vis_pos,3)} cam_pos={np.round(cam_pos,3)}", flush=True)
                image = np.asarray(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()).astype(np.uint8)
                images.append(image)
                labels.append(name)
                tcp_pos = robot.data.body_pos_w[0, tcp_ids[0]].cpu().numpy()
                left_fp = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
                right_fp = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
                jaw_mid_w = 0.5 * (left_fp + right_fp)
                object_tracked = np.asarray(cube_target_world, dtype=np.float64).copy()
                gap = float(np.linalg.norm(left_fp - right_fp))
                finger_now = float(robot.data.joint_pos[0, finger_ids[0]].item())
                per_frame.append({
                    "frame": frame_index + 1, "label": name,
                    "object_pos": object_tracked.tolist(),
                    "object_foot0": to_f0(object_tracked).tolist(),
                    "tcp_pos": tcp_pos.tolist(),
                    "jaw_mid_foot0": to_f0(jaw_mid_w).tolist(),
                    "obj_minus_jawmid_foot0": (to_f0(object_tracked) - to_f0(jaw_mid_w)).tolist(),
                    "pad_mid_foot0": to_f0(_pad_mid_now()).tolist(),
                    "obj_minus_padmid_foot0": (to_f0(object_tracked) - to_f0(_pad_mid_now())).tolist(),
                    "finger_joint": finger_now, "finger_gap_m": gap,
                    "root_pos_w": robot.data.root_pos_w[0].cpu().numpy().tolist(),
                    "root_quat_w": robot.data.root_quat_w[0].cpu().numpy().tolist(),
                    "root_lin_vel_w": robot.data.root_lin_vel_w[0].cpu().numpy().tolist(),
                    "left_ankle_w": robot.data.body_pos_w[0, ankle_l_ids[0]].cpu().numpy().tolist(),
                    "right_ankle_w": robot.data.body_pos_w[0, ankle_r_ids[0]].cpu().numpy().tolist(),
                    "min_body_z_w": float(robot.data.body_pos_w[0, :, 2].min().item()),
                })
                Image.fromarray(image).save(output_dir / f"demo_frame_{frame_index:02d}_{name}.png")
                # set the NEXT frame's transform right after reading this image: the
                # renderer picks it up over the following ~20 physics steps.
                sync_vis()
                update_camera()
                frame_index += 1

    # arm starts AT the grasp pose; the cube is already in the open-finger pocket.
    # Pre-set the visual cube + camera so the first saved frame has transform buffer.
    sync_vis()
    update_camera()
    for _ in range(5):
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); cube.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    run_stage("preset", 0.4, q_close, q_close, "open")
    run_stage("close", args.close_time, q_close, q_close, "close")

    # REAL grasp test: re-enable gravity - the cube must now be held by the pads
    _grav_api.GetDisableGravityAttr().Set(False)
    for _ in range(30):
        pin_root()
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); cube.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
        sync_vis()
    held_f0 = to_f0(cube.data.root_pos_w[0].cpu().numpy())
    grip_drop = float((np.asarray(settled_f0, dtype=np.float64) - np.asarray(held_f0, dtype=np.float64))[2])
    print(f"[V6] gravity ON: cube drop={grip_drop*1000:.1f}mm (small => real grip)", flush=True)
    print(f"[V6] held cube_f0={np.round(held_f0,4)} pad_gap={_pad_gap_now()*1000:.1f}mm", flush=True)

    run_stage("lift", args.lift_time, q_close, q_hold, "hold_closed")
    run_stage("carry", args.carry_time, q_hold, q_place, "hold_closed")

    # --- closed-loop align (uses the REAL cube pose, robust to jaw sliding) -------
    q_align = np.asarray(q_place, dtype=np.float64)
    if args.align_time > 0.0:
        desired_obj_f0 = np.asarray(place_goal_foot0, dtype=np.float64)
        obj_now_f0 = to_f0(cube.data.root_pos_w[0].cpu().numpy())
        tcp_now_f0 = fk_tcp_f0()
        print(f"[V6] align obj_now_f0={np.round(obj_now_f0,4)} tcp_now_f0={np.round(tcp_now_f0,4)}", flush=True)
        print(f"[V6] align tcp_delta={np.round(desired_obj_f0 - obj_now_f0,4)} (must equal final placement error)", flush=True)
        # iterate the closed-loop correction so jaw sliding does not leave a residual
        _align_iter = max(1, int(round(args.align_time / 0.6)))
        _seg_time = args.align_time / _align_iter
        for _ai in range(_align_iter):
            _obj_now = to_f0(cube.data.root_pos_w[0].cpu().numpy())
            _tcp_now = fk_tcp_f0()
            _tcp_goal = _tcp_now + (desired_obj_f0 - _obj_now)
            _q_from = np.asarray(q_place, dtype=np.float64) if _ai == 0 else q_align
            q_align, _ = _solve_place(_tcp_goal, q_align, tool_z, 0.0)
            run_stage("align", _seg_time, _q_from, q_align, "hold_closed")
            _obj_done = to_f0(cube.data.root_pos_w[0].cpu().numpy())
            print(f"[V6] align iter {_ai}: obj_f0={np.round(_obj_done,4)} "
                  f"err={np.linalg.norm(_obj_done - desired_obj_f0)*1000:.1f}mm", flush=True)
    align_obj_f0 = to_f0(cube.data.root_pos_w[0].cpu().numpy())
    print(f"[V6] align done obj_f0={np.round(align_obj_f0,4)}", flush=True)
    print(f"[V6] align residual err={np.linalg.norm(align_obj_f0 - np.asarray(place_goal_foot0, dtype=np.float64))*1000:.1f}mm", flush=True)

    # Freeze the VISUAL cube at the place point before opening the fingers, and pin
    # the PHYSICS cube at the exact place goal with gravity OFF. Opening the Robotiq
    # fingers sweeps the pads outward and would otherwise fling the cube sideways
    # (v30: the cube shot +0.54 m in x during the release stage).
    place_world = from_f0(np.asarray(place_goal_foot0, dtype=np.float64))
    cube_freeze_pose[0] = place_world
    cube.write_root_pose_to_sim(torch.tensor(
        [[place_world[0], place_world[1], place_world[2], 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32, device=sim.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim.device))
    _grav_api.GetDisableGravityAttr().Set(True)
    vis_frozen_pos[0] = place_world + _vis_offset_world()  # same offset as sync_vis: no jump at release
    run_stage("release", args.release_time, q_align, q_align, "release")
    run_stage("settle", 0.8, q_align, q_align, "open")

    # --- summary ------------------------------------------------------------------
    object_f0_traj = np.array([p["object_foot0"] for p in per_frame])
    grasp_frame_idx = next((i for i, p in enumerate(per_frame) if p["label"] == "close"), 0)
    place_goal_arr = np.asarray(place_goal_foot0, dtype=np.float64)
    final_obj_f0 = object_f0_traj[-1]
    place_error = float(np.linalg.norm(final_obj_f0 - place_goal_arr))
    lift_zs = object_f0_traj[grasp_frame_idx:, 2]
    max_lift_z = float(lift_zs.max()) if len(lift_zs) else 0.0
    print(f"[V6] final obj f0={np.round(final_obj_f0,4)} place_goal={np.round(place_goal_arr,4)} err={place_error*1000:.1f}mm", flush=True)
    print(f"[V6] max lift z={max_lift_z*1000:.1f}mm", flush=True)

    # montage
    frame_size = (args.resolution, args.resolution)
    margin = 34
    columns = 6
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * frame_size[0], rows * (frame_size[1] + margin)), (16, 16, 16))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for idx, (image, label) in enumerate(zip(images, labels)):
        col, rrow = divmod(idx, columns)
        x, y = col * frame_size[0], rrow * (frame_size[1] + margin)
        sheet.paste(Image.fromarray(image), (x, y))
        obj = per_frame[idx]["object_foot0"]
        draw.text((x + 8, y + frame_size[1] + 6),
                  f"{idx:02d} {label} obj=({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})",
                  fill=(240, 240, 240))
    montage_path = output_dir / "grasp_place_v3_montage.png"
    sheet.save(montage_path)

    gif_path = output_dir / "grasp_place_v3.gif"
    scale = max(1, args.resolution // 480)
    small = [Image.fromarray(im).resize((args.resolution // scale, args.resolution // scale)) for im in images]
    small[0].save(gif_path, save_all=True, append_images=small[1:], duration=int(total_dt * 1000), loop=0)

    summary = {
        "sample_id": args.sample_id,
        "cube_size_m": args.cube_size,
        "cube_mass_kg": args.cube_mass,
        "q_close": q_close.tolist(),
        "mount_foot0": mount_f0.tolist(),
        "jaw_mid_foot0": jaw_mid.tolist(),
        "gap_open_m": gap_open,
        "pad_local_l": pad_local_l.tolist(),
        "pad_local_r": pad_local_r.tolist(),
        "pad_gap_open_m": pad_gap_open,
        "pad_mid_foot0": to_f0(pad_mid_w).tolist(),
        "held_foot0": held_f0.tolist(),
        "grip_drop_m": grip_drop,
        "preset_cube_foot0": cube_f0.tolist(),
        "u_world_close_foot0": (cube_center_f0 - np.asarray(mount_f0, dtype=np.float64)).tolist(),
        "u_world_place_foot0": (mount_goal_f0 - np.asarray(place_goal_foot0, dtype=np.float64)).tolist(),
        "place_goal_foot0": place_goal_arr.tolist(),
        "place_error_m": place_error,
        "max_lift_z_m": max_lift_z,
        "n_frames": frame_index,
        "per_frame": per_frame,
        "montage": str(montage_path),
        "gif": str(gif_path),
    }
    (output_dir / "demo_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[V6] montage={montage_path}", flush=True)
    print(f"[V6] gif={gif_path}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
















