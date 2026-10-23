"""Inspect camera and grasp-object geometry authored in 1111_copy.usd."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "1111_copy.usd"
OUTPUT = ROOT / "outputs" / "scene_geometry_audit.json"


def _translation(prim: Usd.Prim) -> list[float]:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return [float(value) for value in matrix.ExtractTranslation()]


def main() -> None:
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"failed to open {SCENE}")
    cameras = []
    objects = []
    object_names = {"Box", "Cylinder", "Sphere", "Cone", "Capsule"}
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)
            cameras.append(
                {
                    "path": str(prim.GetPath()),
                    "world_position_m": _translation(prim),
                    "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
                    "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get()),
                    "clipping_range_m": [float(v) for v in camera.GetClippingRangeAttr().Get()],
                }
            )
        if prim.GetName().split("_")[0] in object_names and prim.IsA(UsdGeom.Xformable):
            objects.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "world_position_m": _translation(prim),
                }
            )
    report = {"scene": str(SCENE), "camera_count": len(cameras), "cameras": cameras, "objects": objects}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
