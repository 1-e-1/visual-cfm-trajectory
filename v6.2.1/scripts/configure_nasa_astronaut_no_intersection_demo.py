"""Create an isolated paper-demo scene with the auxiliary arm outside the suit silhouette."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "1111_nasa_astronaut_aux_arm_20tasks.usd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "1111_nasa_astronaut_no_intersection_demo.usd",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/nasa_astronaut_no_intersection_demo/scene_adjustment.json",
    )
    parser.add_argument(
        "--astronaut_world_y_offset_m",
        type=float,
        default=0.24,
        help="Move the render-only suit away from the mounted arm along world Y.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source.resolve(), args.output.resolve())
    stage = Usd.Stage.Open(str(args.output.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open copied USD: {args.output}")

    astronaut = UsdGeom.Xformable(stage.GetPrimAtPath("/World/NASA_Astronaut_ACES"))
    if not astronaut:
        raise RuntimeError("NASA astronaut root is missing")
    translate = astronaut.GetOrderedXformOps()[0]
    original = translate.Get()
    adjusted = Gf.Vec3d(original[0], original[1] + args.astronaut_world_y_offset_m, original[2])
    translate.Set(adjusted)

    # Extend the visual mounting plate across the new standoff distance.  The
    # validated internal FixedJoint and arm articulation are left untouched.
    mount = UsdGeom.Xformable(stage.GetPrimAtPath("/World/AuxArmWaistMount"))
    if mount:
        for op in mount.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale = op.Get()
                op.Set(Gf.Vec3d(scale[0] + args.astronaut_world_y_offset_m, scale[1], scale[2]))
                break

    stage.GetRootLayer().Save()
    report = {
        "source_scene": str(args.source.resolve()),
        "output_scene": str(args.output.resolve()),
        "astronaut_render_only": True,
        "astronaut_world_y_offset_m": float(args.astronaut_world_y_offset_m),
        "astronaut_original_translate": list(original),
        "astronaut_adjusted_translate": list(adjusted),
        "arm_articulation_modified": False,
        "fixed_joint_modified": False,
        "purpose": "paper visualization without suit-arm mesh intersection",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
