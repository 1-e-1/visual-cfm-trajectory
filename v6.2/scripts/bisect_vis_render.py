# -*- coding: utf-8 -*-
"""Bisect: reproduce d63 visible case, then add v6.6 elements one by one."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _bootstrap  # noqa
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True; args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from h1_1111_grasp.env_cfg import H1_1111_CFG
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.utils import stage as stage_utils
import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdShade, Sdf

def look_at_quat(eye, target):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye; fwd = fwd/np.linalg.norm(fwd)
    right = np.cross(fwd, [0,0,1]); right = right/np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.stack([right, up, -fwd], axis=1)
    tr = rot[0,0]+rot[1,1]+rot[2,2]
    if tr > 0:
        s = np.sqrt(tr+1)*2; q = (0.25*s, (rot[2,1]-rot[1,2])/s, (rot[0,2]-rot[2,0])/s, (rot[1,0]-rot[0,1])/s)
    elif rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]:
        s = np.sqrt(1+rot[0,0]-rot[1,1]-rot[2,2])*2; q = ((rot[2,1]-rot[1,2])/s, 0.25*s, (rot[0,1]+rot[1,0])/s, (rot[0,2]+rot[2,0])/s)
    elif rot[1,1] > rot[2,2]:
        s = np.sqrt(1+rot[1,1]-rot[0,0]-rot[2,2])*2; q = ((rot[0,2]-rot[2,0])/s, (rot[0,1]+rot[1,0])/s, 0.25*s, (rot[1,2]+rot[2,1])/s)
    else:
        s = np.sqrt(1+rot[2,2]-rot[0,0]-rot[1,1])*2; q = ((rot[1,0]-rot[0,1])/s, (rot[0,2]+rot[2,0])/s, (rot[1,2]+rot[2,1])/s, 0.25*s)
    return q

def main():
    stage_utils.open_stage(str(Path(str(__import__('pathlib').Path(__file__).resolve().parents[1] / '1111_copy.usd'))))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))
    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath("/World/DynPartsLeft").IsValid():
        stage.RemovePrim("/World/DynPartsLeft")
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/DynPartsLeft/Obj0",
        spawn=sim_utils.CuboidCfg(
            size=(0.035,0.035,0.035), visual_material=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,0.5), rot=(1.0,0.0,0.0,0.0)),
    ))
    _pm = stage.GetPrimAtPath("/World/DynPartsLeft/Obj0/geometry/mesh")
    if _pm.IsValid(): _pm.GetAttribute("visibility").Set("invisible")
    _pr = stage.GetPrimAtPath("/World/DynPartsLeft/Obj0")
    if _pr.IsValid(): _pr.GetAttribute("visibility").Set("invisible")

    vis_path = "/World/CubeVis"
    if stage.GetPrimAtPath(vis_path).IsValid(): stage.RemovePrim(vis_path)
    vis_prim = stage.DefinePrim(vis_path, "Xform")
    vis_xform = UsdGeom.Xformable(vis_prim)
    vis_xform.AddTranslateOp(); vis_xform.AddOrientOp()
    vis_geom = UsdGeom.Cube.Define(stage, vis_path + "/Cube")
    vis_geom.GetSizeAttr().Set(0.035)
    vis_mat_prim = stage.DefinePrim(vis_path + "/Mat", "Material")
    vis_shader = UsdShade.Shader.Define(stage, vis_path + "/Mat/Shader")
    vis_shader.CreateIdAttr().Set("UsdPreviewSurface")
    vis_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 0.85, 0.05))
    vis_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    vis_mat = UsdShade.Material(vis_mat_prim)
    vis_mat.CreateSurfaceOutput().ConnectToSource(vis_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(vis_geom.GetPrim()).Bind(vis_mat)
    vis_ops = vis_xform.GetOrderedXformOps()

    cam_path = "/World/TestCam"
    if stage.GetPrimAtPath(cam_path).IsValid(): stage.RemovePrim(cam_path)
    sim_utils.spawn_camera(cam_path, sim_utils.PinholeCameraCfg(
        focal_length=28.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01,100.0)),
        translation=(0,0,0), orientation=(1,0,0,0))
    camera = Camera(CameraCfg(prim_path=cam_path, update_period=0.0, spawn=None, data_types=["rgb"], width=640, height=640))
    sim.reset(); robot.reset(); cube.reset(); camera.reset()

    arm_ids, _ = robot.find_joints([f"^joint{i}_0$" for i in range(1,7)], preserve_order=True)
    finger_ids, _ = robot.find_joints(["^finger_joint_0$"], preserve_order=True)
    lf_ids, _ = robot.find_bodies(["^left_inner_finger_0$","^right_inner_finger_0$"], preserve_order=True)
    base_ids, _ = robot.find_bodies(["^foot_0_0$"], preserve_order=True)
    q_close = np.array([-2.1070001125335693, 0.9599999785423279, 0.597000002861023, 0.47600001096725464, 0.09799999743700027, -0.028999999165534973])
    finger_open = 0.7792895138263702
    jp = robot.data.default_joint_pos.clone()
    jp[:, arm_ids] = torch.as_tensor([q_close], dtype=torch.float32, device=jp.device)
    jp[:, finger_ids] = torch.as_tensor([finger_open], dtype=torch.float32, device=jp.device)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.set_joint_position_target(jp)
    for _ in range(200):
        robot.write_data_to_sim(); sim.step(render=True); robot.update(sim.get_physics_dt())

    base_position = robot.data.body_pos_w[0, base_ids[0]].cpu().numpy()
    base_quaternion = robot.data.body_quat_w[0, base_ids[0]].cpu().numpy()
    def Rmat(q):
        q = np.asarray(q,float); q = q/np.linalg.norm(q); w,x,y,z = q
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    def to_f0_(w):
        R = Rmat(base_quaternion); return (np.asarray(w,float)-base_position) @ R
    def from_f0(local):
        R = Rmat(base_quaternion); return base_position + np.asarray(local,float) @ R.T

    # place cube at jaw midpoint, disable gravity, settle (like d63)
    lf0 = to_f0_(robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy())
    rf0 = to_f0_(robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy())
    cube_f0 = 0.5*(lf0+rf0)
    cube_world = from_f0(cube_f0)
    cube.write_root_pose_to_sim(torch.tensor([[cube_world[0],cube_world[1],cube_world[2],1,0,0,0]], dtype=torch.float32, device=sim.device))
    cube.write_root_velocity_to_sim(torch.zeros((1,6), dtype=torch.float32, device=sim.device))
    PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath("/World/DynPartsLeft/Obj0")).GetDisableGravityAttr().Set(True)
    for _ in range(15):
        robot.write_data_to_sim(); sim.step(render=True)
        robot.update(sim.get_physics_dt()); cube.update(sim.get_physics_dt())
    print(f"[BISECT] cube settled f0={np.round(to_f0_(cube.data.root_pos_w[0].cpu().numpy()),4)}", flush=True)

    lw = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
    rw = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
    mid = 0.5*(lw+rw)
    zoff = 0.035

    # SET A: like d63 - set vis once, look at vis cube
    vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
    vis_ops[1].Set(Gf.Quatf(1,0,0,0))
    cam_prim = stage.GetPrimAtPath(cam_path)
    eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
    qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
    cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
    from PIL import Image
    for _ in range(5):
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] A set-once: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    Image.fromarray(rgb).save(str(Path(__file__).resolve().parents[1] / "outputs" / "bisect_A.png"))

    # SET B: every step re-set vis (like v6.6 sync_vis), look at vis cube
    for _ in range(20):
        lw = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
        rw = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
        mid = 0.5*(lw+rw)
        vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
        vis_ops[1].Set(Gf.Quatf(1,0,0,0))
        eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
        qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
        cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
        cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] B every-step-sync: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    Image.fromarray(rgb).save(str(Path(__file__).resolve().parents[1] / "outputs" / "bisect_B.png"))

    # SET D: set once, then step 40 more (does repeated stepping kill it?)
    vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
    vis_ops[1].Set(Gf.Quatf(1,0,0,0))
    eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
    qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
    cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
    for _ in range(40):
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] D set-once-step40: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    Image.fromarray(rgb).save(str(Path(__file__).resolve().parents[1] / "outputs" / "bisect_D.png"))

    # SET F: set once, then render N steps (N=1,2,4,8,16), read each
    for nsteps in [1, 2, 4, 8, 16, 32]:
        vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
        vis_ops[1].Set(Gf.Quatf(1,0,0,0))
        eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
        qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
        cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
        cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
        for _ in range(nsteps):
            sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
        a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
        print(f"[BISECT] F n={nsteps}: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    # reset to a known good state
    vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
    vis_ops[1].Set(Gf.Quatf(1,0,0,0))
    eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
    qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
    cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
    for _ in range(10):
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())

    # SET E: set once, read immediately (frame lag test)
    vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
    vis_ops[1].Set(Gf.Quatf(1,0,0,0))
    eye = mid + np.array([0.0,0.0,zoff]) + np.array([0.55,0.0,0.25])
    qq = look_at_quat(eye, mid + np.array([0.0,0.0,zoff]))
    cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
    cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
    # warm up one frame, then read frame 1
    sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] E1 after-2-steps: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] E2 after-5-steps: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)

    # SET C: like B but look at the PHYS cube (v6.6 update_camera used phys cube before my fix)
    phys_w = cube.data.root_pos_w[0].cpu().numpy()
    for _ in range(20):
        lw = robot.data.body_pos_w[0, lf_ids[0]].cpu().numpy()
        rw = robot.data.body_pos_w[0, lf_ids[1]].cpu().numpy()
        mid = 0.5*(lw+rw)
        vis_ops[0].Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2]+zoff)))
        vis_ops[1].Set(Gf.Quatf(1,0,0,0))
        phys_w = cube.data.root_pos_w[0].cpu().numpy()
        eye = phys_w + np.array([0.55,0.0,0.25])
        qq = look_at_quat(eye, phys_w)
        cam_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*eye))
        cam_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*[float(v) for v in qq]))
        sim.step(render=True); robot.update(sim.get_physics_dt()); camera.update(sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0,...,:3].detach().cpu().numpy().astype(np.uint8)
    a = rgb.astype(int); r,g,b = a[...,0],a[...,1],a[...,2]
    print(f"[BISECT] C look-phys: yellow={int(((r>150)&(g>120)&(b<90)).sum())}", flush=True)
    Image.fromarray(rgb).save(str(Path(__file__).resolve().parents[1] / "outputs" / "bisect_C.png"))
    print("[BISECT] done", flush=True)
    simulation_app.close()

if __name__ == "__main__":
    main()
