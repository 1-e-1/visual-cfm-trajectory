"""Capture one five-camera sample from the current waist-mounted astronaut scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap  # noqa: F401,E402

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--usd",
    type=Path,
    default=PROJECT_ROOT / "1111_nasa_astronaut_waist_mounted_aux_arm.usd",
)
parser.add_argument(
    "--plan",
    type=Path,
    default=PROJECT_ROOT
    / "outputs/nasa_astronaut_aux_arm_20tasks/plans/task_01_center.npz",
)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "outputs/current_astronaut_dataset_sample",
)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--render_steps", type=int, default=8)
parser.add_argument(
    "--arm_pose", choices=("start", "pregrasp", "grasp"), default="pregrasp"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim.utils import stage as stage_utils  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSemantics, UsdShade  # noqa: E402

from h1_1111_grasp.env_cfg import H1_1111_CFG  # noqa: E402


CAMERAS = [
    ("wrist", "/World/UnitreeH1/ZBot6S_left/foot_1/Sensors/LeftWristCamera"),
    ("cube", "/World/UnitreeH1/Cube/Sensors/LeftCubeSideCamera"),
    ("chest", "/World/UnitreeH1/torso_link/Sensors/ChestLeftCamera"),
    ("abdomen", "/World/UnitreeH1/pelvis/Sensors/LeftHipCamera"),
    ("head", "/World/UnitreeH1/torso_link/Sensors/HeadTopCamera"),
]

CAMERA_ZH = {
    "wrist": "腕部相机",
    "cube": "安装座相机",
    "chest": "胸部相机",
    "abdomen": "腹部相机",
    "head": "头部相机",
}


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _bind_material(
    stage: Usd.Stage, prim: Usd.Prim, path: str, color: tuple[float, float, float]
) -> None:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _add_capture_task(
    stage: Usd.Stage, plan: dict[str, np.ndarray]
) -> tuple[np.ndarray, float]:
    foot0 = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/UnitreeH1/ZBot6S_left/foot_0")
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    center_foot0 = np.asarray(plan["grasp_center_position_foot0"], dtype=np.float64)
    center_world = np.asarray(
        foot0.Transform(Gf.Vec3d(*[float(value) for value in center_foot0]))
    )
    size = 0.10

    root_path = "/World/CurrentDatasetCapture"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    target = UsdGeom.Cube.Define(stage, f"{root_path}/TargetBlock")
    target.CreateSizeAttr(size)
    target.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center_world]))
    labels = UsdSemantics.LabelsAPI.Apply(target.GetPrim(), "class")
    labels.CreateLabelsAttr().Set(["target_object"])
    _bind_material(stage, target.GetPrim(), f"{root_path}/TargetMaterial", (0.95, 0.24, 0.08))

    support_height = max(0.08, float(center_world[2] - 0.5 * size))
    support = UsdGeom.Cylinder.Define(stage, f"{root_path}/PickupSupport")
    support.CreateRadiusAttr(0.065)
    support.CreateHeightAttr(support_height)
    support.AddTranslateOp().Set(
        Gf.Vec3d(float(center_world[0]), float(center_world[1]), 0.5 * support_height)
    )
    labels = UsdSemantics.LabelsAPI.Apply(support.GetPrim(), "class")
    labels.CreateLabelsAttr().Set(["task_support"])
    _bind_material(stage, support.GetPrim(), f"{root_path}/SupportMaterial", (0.45, 0.47, 0.50))
    return center_world, size


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(value) & (value > 0.02) & (value < 6.0)
    image = np.zeros(value.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.quantile(value[finite], [0.02, 0.98])
        normalized = 1.0 - np.clip((value - low) / max(high - low, 1.0e-6), 0.0, 1.0)
        image[finite] = (255.0 * normalized[finite]).astype(np.uint8)
    return image


def _target_ids(packet: dict) -> list[int]:
    result = []
    for semantic_id, labels in packet.get("info", {}).get("idToLabels", {}).items():
        if labels.get("class") == "target_object":
            result.append(int(semantic_id))
    return result


def _save_contact_sheet(
    images: list[tuple[str, np.ndarray, np.ndarray]], path: Path
) -> None:
    tile = 320
    title_height = 58
    sheet = Image.new("RGB", (len(images) * tile, tile + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    small_font = _font(17)
    for index, (name, rgb, target_mask) in enumerate(images):
        image = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.NEAREST)
        x = index * tile
        sheet.paste(image, (x, title_height))
        ys, xs = np.nonzero(target_mask)
        if len(xs):
            scale = tile / rgb.shape[1]
            box = (
                x + int(xs.min() * scale),
                title_height + int(ys.min() * scale),
                x + int((xs.max() + 1) * scale),
                title_height + int((ys.max() + 1) * scale),
            )
            draw.rectangle(box, outline=(255, 35, 20), width=4)
        draw.text((x + 8, 5), CAMERA_ZH[name], fill=(20, 20, 20), font=font)
        draw.text((x + 8, 32), name, fill=(80, 80, 80), font=small_font)
    sheet.save(path)


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.plan, allow_pickle=False) as data:
        plan = {key: np.array(data[key], copy=True) for key in data.files}

    stage_utils.open_stage(str(args.usd.resolve()))
    stage = omni.usd.get_context().get_stage()
    target_center_world, target_size = _add_capture_task(stage, plan)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=1))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))

    annotators: dict[str, dict[str, object]] = {}
    for name, camera_path in CAMERAS:
        if not stage.GetPrimAtPath(camera_path).IsValid():
            raise RuntimeError(f"missing camera: {camera_path}")
        product = rep.create.render_product(
            camera_path, resolution=(args.resolution, args.resolution)
        )
        views = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb", device="cpu"),
            "depth": rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane", device="cpu"
            ),
            "semantic": rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation",
                init_params={"semanticTypes": ["class"], "colorize": False},
                device="cpu",
            ),
        }
        for annotator in views.values():
            annotator.attach([product])
        annotators[name] = views

    sim.reset()
    robot.reset()
    arm_ids, arm_names = robot.find_joints(
        [f"^joint{i}_0$" for i in range(1, 7)], preserve_order=True
    )
    if len(arm_ids) != 6:
        raise RuntimeError(f"failed to resolve left arm joints: {arm_names}")
    pose_key = {"start": "q_start", "pregrasp": "q_pregrasp", "grasp": "q_grasp"}[
        args.arm_pose
    ]
    joint_position = robot.data.default_joint_pos.clone()
    joint_position[:, arm_ids] = torch.as_tensor(
        plan[pose_key], dtype=joint_position.dtype, device=robot.device
    )
    joint_velocity = torch.zeros_like(joint_position)
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    for _ in range(args.render_steps):
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())

    image_records: list[tuple[str, np.ndarray, np.ndarray]] = []
    report = {
        "source_usd": str(args.usd.resolve()),
        "source_plan": str(args.plan.resolve()),
        "sample_id": f"current_astronaut_{str(plan['sample_id'])}_{args.arm_pose}",
        "arm_pose": args.arm_pose,
        "resolution": [args.resolution, args.resolution],
        "target_center_world_m": target_center_world.tolist(),
        "target_size_m": target_size,
        "cameras": {},
    }
    for name, _ in CAMERAS:
        rgb = np.asarray(annotators[name]["rgb"].get_data())[..., :3].astype(np.uint8)
        depth = np.asarray(annotators[name]["depth"].get_data()).astype(np.float32)
        semantic_packet = annotators[name]["semantic"].get_data()
        semantic = np.asarray(semantic_packet["data"]).astype(np.int32)
        target_ids = _target_ids(semantic_packet)
        target_mask = np.isin(semantic, target_ids)

        Image.fromarray(rgb).save(args.output / f"{name}_rgb.png")
        Image.fromarray(_normalize_depth(depth)).save(args.output / f"{name}_depth.png")
        np.savez_compressed(
            args.output / f"{name}_raw.npz",
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            target_mask=target_mask,
        )
        image_records.append((name, rgb, target_mask))
        report["cameras"][name] = {
            "target_semantic_ids": target_ids,
            "target_visible_pixels": int(target_mask.sum()),
            "semantic_id_to_labels": semantic_packet.get("info", {}).get(
                "idToLabels", {}
            ),
        }

    _save_contact_sheet(image_records, args.output / "five_camera_rgb.png")
    visible_records = [record for record in image_records if bool(record[2].any())]
    _save_contact_sheet(visible_records, args.output / "visible_camera_rgb.png")
    report["visible_camera_names"] = [record[0] for record in visible_records]
    (args.output / "capture_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not any(entry["target_visible_pixels"] > 0 for entry in report["cameras"].values()):
        raise RuntimeError("target block is not visible in any camera")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=True)
