"""Create an isolated USD with mirrored Robotiq fingertip colliders.

The attachment-constraint baseline scene is never modified. The generated USD
is dedicated to native PhysX contact-grasp experiments.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_box(
    stage: Usd.Stage,
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: UsdShade.Material,
) -> None:
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        material, materialPurpose="physics"
    )


def set_sphere_pad(
    stage: Usd.Stage,
    path: str,
    center: tuple[float, float, float],
    radius: float,
    material: UsdShade.Material,
) -> None:
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*center))
    sphere.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
        material, materialPurpose="physics"
    )


def configure(source: Path, output: Path) -> dict[str, object]:
    if source.resolve() == output.resolve():
        raise ValueError("native-contact USD must not overwrite the baseline scene")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"cannot open copied stage: {output}")

    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/NativeContactGripperPad"
    )
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(2.5)
    physics.CreateDynamicFrictionAttr(2.0)
    physics.CreateRestitutionAttr(0.0)

    corrected = []
    for side in ("left", "right"):
        root = f"/World/UnitreeH1/ZBot6S_{side}/Robotiq2F85"
        for finger, sign in (("left_inner_finger", -1.0), ("right_inner_finger", 1.0)):
            finger_root = f"{root}/{finger}"
            if not stage.GetPrimAtPath(finger_root).IsValid():
                raise RuntimeError(f"missing finger link: {finger_root}")
            collision_root = f"{finger_root}/collisions"
            UsdGeom.Scope.Define(stage, collision_root)
            set_box(
                stage,
                f"{collision_root}/shaft",
                (0.0, sign * 0.0599, 0.1054),
                (0.027, 0.0299, 0.028),
                material,
            )
            # A thin, broad contact patch follows the inward-facing white pad.
            # Its inner surface is coincident with the measured visual face.
            pad_root = f"{collision_root}/pad"
            if stage.GetPrimAtPath(pad_root).IsValid():
                stage.RemovePrim(pad_root)
            UsdGeom.Scope.Define(stage, pad_root)
            pad_thickness = 0.006
            set_box(
                stage,
                f"{pad_root}/contact_face",
                (0.0, sign * (0.0435484238 + 0.5 * pad_thickness), 0.1294),
                (0.032, pad_thickness, 0.040),
                material,
            )
            corrected.append(finger_root)

    stage.GetRootLayer().Save()
    return {
        "source": str(source),
        "output": str(output),
        "corrected_finger_links": corrected,
        "static_friction": 2.5,
        "dynamic_friction": 2.0,
        "restitution": 0.0,
        "pad_shape": "thin_box",
        "pad_size_m": [0.032, 0.006, 0.040],
        "pad_center_z_m": 0.1294,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "1111_copy.usd")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "1111_copy_native_contact.usd"
    )
    args = parser.parse_args()
    report = configure(args.source.resolve(), args.output.resolve())
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
