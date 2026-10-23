"""Capture and audit the four RGB-D views used by each mounted arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import h5py  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs/camera_audit"))
parser.add_argument("--side", choices=("left", "right"), default="left")
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--render_steps", type=int, default=4)
parser.add_argument("--no_semantic", action="store_true")
parser.add_argument(
    "--single_view", choices=("wrist", "cube", "chest", "abdomen", "head"), default=None
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.utils import stage as stage_utils
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom

from h1_1111_grasp.env_cfg import H1_1111_CFG


CAMERA_PATHS = {
    "left_wrist": "/World/UnitreeH1/ZBot6S_left/foot_1/Sensors/LeftWristCamera",
    "left_cube": "/World/UnitreeH1/Cube/Sensors/LeftCubeSideCamera",
    "left_chest": "/World/UnitreeH1/torso_link/Sensors/ChestLeftCamera",
    "left_abdomen": "/World/UnitreeH1/pelvis/Sensors/LeftHipCamera",
    "head_top": "/World/UnitreeH1/torso_link/Sensors/HeadTopCamera",
    "right_wrist": "/World/UnitreeH1/ZBot6S_right/foot_1/Sensors/RightWristCamera",
    "right_cube": "/World/UnitreeH1/Cube/Sensors/RightCubeSideCamera",
    "right_chest": "/World/UnitreeH1/torso_link/Sensors/ChestRightCamera",
    "right_abdomen": "/World/UnitreeH1/pelvis/Sensors/RightHipCamera",
}


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(value) & (value > 0.02) & (value < 4.0)
    image = np.zeros(value.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.quantile(value[finite], [0.02, 0.98])
        normalized = 1.0 - np.clip((value - low) / max(high - low, 1.0e-6), 0.0, 1.0)
        image[finite] = (255.0 * normalized[finite]).astype(np.uint8)
    return image


def _contact_sheet(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    width, height = 256, 256
    sheet = Image.new("RGB", (4 * width, 2 * (height + 28)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (name, array) in enumerate(images):
        row, column = divmod(index, 4)
        image = Image.fromarray(array).resize((width, height))
        x, y = column * width, row * (height + 28)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + height + 5), name, fill=(240, 240, 240))
    sheet.save(path)


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    scene_path = Path(__file__).resolve().parents[2] / "1111_copy.usd"
    print(f"[CAMERA] opening_stage={scene_path}", flush=True)
    stage_utils.open_stage(str(scene_path))
    print("[CAMERA] stage_opened", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=1))
    robot = Articulation(H1_1111_CFG.replace(prim_path="/World/UnitreeH1", spawn=None))
    print("[CAMERA] articulation_bound", flush=True)
    selected_names = [
        f"{args.side}_wrist",
        f"{args.side}_cube",
        f"{args.side}_chest",
        f"{args.side}_abdomen",
        "head_top",
    ]
    if args.single_view is not None:
        selected_names = ["head_top" if args.single_view == "head" else f"{args.side}_{args.single_view}"]
    render_products = {}
    annotators = {}
    for name, path in CAMERA_PATHS.items():
        if name not in selected_names:
            continue
        product = rep.create.render_product(path, resolution=(args.resolution, args.resolution))
        view_annotators = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb", device="cpu"),
            "distance_to_image_plane": rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane", device="cpu"
            ),
        }
        if not args.no_semantic:
            view_annotators["semantic_segmentation"] = rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation",
                init_params={"semanticTypes": ["class"], "colorize": False},
                device="cpu",
            )
        for annotator in view_annotators.values():
            annotator.attach([product])
        render_products[name] = product
        annotators[name] = view_annotators
    print(f"[CAMERA] render_products_bound={list(render_products)}", flush=True)
    sim.reset()
    print("[CAMERA] simulation_reset", flush=True)
    robot.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos)
    for _ in range(args.render_steps):
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
    print("[CAMERA] render_complete", flush=True)

    rgb_images = []
    depth_images = []
    report = {"side": args.side, "resolution": [args.resolution, args.resolution], "cameras": {}}
    stage = omni.usd.get_context().get_stage()
    for name, view_annotators in annotators.items():
        rgb = np.asarray(view_annotators["rgb"].get_data())[..., :3].astype(np.uint8)
        depth = np.asarray(view_annotators["distance_to_image_plane"].get_data()).astype(np.float32)
        semantic_packet = (
            view_annotators["semantic_segmentation"].get_data()
            if "semantic_segmentation" in view_annotators
            else None
        )
        semantic = (
            np.asarray(semantic_packet["data"]).astype(np.int32)
            if semantic_packet is not None
            else np.zeros(depth.shape, dtype=np.int32)
        )
        Image.fromarray(rgb).save(args.output / f"{name}_rgb.png")
        Image.fromarray(_normalize_depth(depth)).save(args.output / f"{name}_depth.png")
        np.savez_compressed(args.output / f"{name}_raw.npz", rgb=rgb, depth=depth, semantic=semantic)
        rgb_images.append((name, rgb))
        depth_images.append((name, np.repeat(_normalize_depth(depth)[..., None], 3, axis=-1)))
        finite = np.isfinite(depth)
        camera_prim = stage.GetPrimAtPath(CAMERA_PATHS[name])
        matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        report["cameras"][name] = {
            "prim_path": CAMERA_PATHS[name],
            "position_w_m": [float(value) for value in matrix.ExtractTranslation()],
            "quaternion_wxyz": [
                float(rotation.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            "finite_depth_ratio": float(finite.mean()),
            "depth_min_m": float(depth[finite].min()) if finite.any() else None,
            "depth_max_m": float(depth[finite].max()) if finite.any() else None,
            "semantic_ids": np.unique(semantic).astype(int).tolist(),
            "semantic_id_to_labels": (
                semantic_packet.get("info", {}).get("idToLabels", {})
                if semantic_packet is not None
                else {}
            ),
        }
    _contact_sheet(rgb_images, args.output / "contact_sheet_rgb.png")
    _contact_sheet(depth_images, args.output / "contact_sheet_depth.png")
    (args.output / "camera_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("[CAMERA] artifacts_saved", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(skip_cleanup=True)
