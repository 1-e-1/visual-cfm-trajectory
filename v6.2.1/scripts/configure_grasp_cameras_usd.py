"""Persistently aim the fixed RGB-D cameras at the left-arm grasp workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_TARGETS_W = {
    # The two endpoints are the validated contact centers transformed from foot_0.
    "/World/UnitreeH1/Cube/Sensors/LeftCubeSideCamera": (0.02, -0.77, 0.58),
    "/World/UnitreeH1/torso_link/Sensors/ChestLeftCamera": (-0.26, 0.05, 0.75),
    "/World/UnitreeH1/pelvis/Sensors/LeftHipCamera": (-0.12, -0.36, 0.67),
    "/World/UnitreeH1/torso_link/Sensors/HeadTopCamera": (-0.12, -0.36, 0.67),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=PROJECT_ROOT / "1111_copy.usd")
    parser.add_argument(
        "--backup",
        type=Path,
        default=PROJECT_ROOT / "1111_copy_pre_grasp_camera_fix.usd",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usd_path = args.usd.resolve()
    if not args.backup.exists():
        shutil.copy2(usd_path, args.backup)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"cannot open USD: {usd_path}")

    audit = []
    for camera_path, target_tuple in CAMERA_TARGETS_W.items():
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"missing camera prim: {camera_path}")
        parent_prim = camera_prim.GetParent().GetParent()
        camera_world_old = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
        eye = camera_world_old.ExtractTranslation()
        target = Gf.Vec3d(*target_tuple)

        # SetLookAt returns a world-to-camera view matrix.  Its inverse is the
        # camera-to-world transform expected by the USD Xformable.
        view = Gf.Matrix4d(1.0)
        view.SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0))
        desired_world = view.GetInverse()
        parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(0)
        desired_local = desired_world * parent_world.GetInverse()

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(desired_local)
        camera = UsdGeom.Camera(camera_prim)
        camera.GetFocalLengthAttr().Set(24.0)
        camera.GetHorizontalApertureAttr().Set(20.955)
        camera.GetVerticalApertureAttr().Set(20.955)

        forward = np.asarray(desired_world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)))
        expected = np.asarray(target) - np.asarray(eye)
        forward /= np.linalg.norm(forward)
        expected /= np.linalg.norm(expected)
        audit.append((camera_path, float(np.dot(forward, expected))))

    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(usd_path))
    for camera_path, expected_dot in audit:
        matrix = UsdGeom.Xformable(reopened.GetPrimAtPath(camera_path)).ComputeLocalToWorldTransform(0)
        print(
            f"camera={camera_path} position={list(matrix.ExtractTranslation())} "
            f"forward_alignment={expected_dot:.9f}"
        )
    print(f"saved={usd_path}")
    print(f"backup={args.backup.resolve()}")


if __name__ == "__main__":
    main()
