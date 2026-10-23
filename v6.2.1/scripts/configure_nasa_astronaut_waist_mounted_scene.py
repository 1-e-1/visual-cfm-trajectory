"""Create a render-only astronaut scene with the auxiliary arm emerging from the left waist."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASTRONAUT_ROOT = "/World/NASA_Astronaut_ACES"
ACTIVE_ARM_ROOT = "/World/UnitreeH1/ZBot6S_left"
MOUNT_ROOT = "/World/AstronautWaistAuxArmMount"
MOUNT_CENTER = Gf.Vec3d(0.0735, -0.397, 1.1432)
PROTECTED_FIXED_JOINTS = {
    "/World/UnitreeH1/ZBot6S_right/GripperMountJoint",
    "/World/UnitreeH1/ZBot6S_left/GripperMountJoint",
    "/World/UnitreeH1/AssemblyJoints/CubeToTorso",
    "/World/UnitreeH1/AssemblyJoints/RightAuxArmToCube",
    "/World/UnitreeH1/AssemblyJoints/LeftAuxArmToCube",
}


def _material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    metallic: float,
    roughness: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mark_render_only(prim: Usd.Prim) -> None:
    prim.SetCustomDataByKey("astronaut:renderOnly", True)
    prim.SetCustomDataByKey("astronaut:collisionEnabled", False)


def _bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    _mark_render_only(prim)


def _world_range(stage: Usd.Stage, path: str) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


def _repath_local_assets(stage: Usd.Stage) -> dict[str, str]:
    assets = PROJECT_ROOT / "assets" / "source" / "assets"
    mapping = {
        "/World/UnitreeH1": assets / "unitree_h1.usd",
        "/World/UnitreeH1/ZBot6S_left": assets / "zbot_6s_new_configured.usd",
        "/World/UnitreeH1/ZBot6S_right": assets / "zbot_6s_new_configured.usd",
        "/World/UnitreeH1/ZBot6S_left/Robotiq2F85": assets / "robotiq_2f85.usd",
        "/World/UnitreeH1/ZBot6S_right/Robotiq2F85": assets / "robotiq_2f85.usd",
    }
    result = {}
    for prim_path, asset_path in mapping.items():
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"missing reference prim: {prim_path}")
        references = prim.GetReferences()
        references.ClearReferences()
        references.AddReference(str(asset_path.resolve()))
        result[prim_path] = str(asset_path.resolve())
    return result


def _move_astronaut_to_waist_mount(stage: Usd.Stage, lateral_offset_m: float) -> list[float]:
    astronaut = UsdGeom.Xformable(stage.GetPrimAtPath(ASTRONAUT_ROOT))
    if not astronaut:
        raise RuntimeError("NASA astronaut root is missing")
    translate_ops = [
        op for op in astronaut.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if not translate_ops:
        raise RuntimeError("NASA astronaut translate op is missing")
    value = translate_ops[0].Get()
    adjusted = Gf.Vec3d(value[0], value[1] + lateral_offset_m, value[2])
    translate_ops[0].Set(adjusted)
    return [float(component) for component in adjusted]


def _create_mount(stage: Usd.Stage) -> list[str]:
    old_mount = stage.GetPrimAtPath("/World/AuxArmWaistMount")
    if old_mount.IsValid():
        stage.RemovePrim(old_mount.GetPath())
    if stage.GetPrimAtPath(MOUNT_ROOT).IsValid():
        stage.RemovePrim(MOUNT_ROOT)

    root = UsdGeom.Xform.Define(stage, MOUNT_ROOT)
    _mark_render_only(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("astronaut:mountType", "left-waist auxiliary-arm interface")
    looks = UsdGeom.Scope.Define(stage, f"{MOUNT_ROOT}/Looks")
    suit_adapter = _material(stage, f"{looks.GetPath()}/SuitAdapter", (0.72, 0.76, 0.78), 0.15, 0.42)
    structural = _material(stage, f"{looks.GetPath()}/Structural", (0.045, 0.055, 0.065), 0.72, 0.22)
    accent = _material(stage, f"{looks.GetPath()}/Fastener", (0.62, 0.65, 0.67), 0.90, 0.14)

    created = []
    backing = UsdGeom.Cube.Define(stage, f"{MOUNT_ROOT}/SuitBackplate")
    backing.CreateSizeAttr(1.0)
    backing.AddTranslateOp().Set(Gf.Vec3d(MOUNT_CENTER[0], -0.350, MOUNT_CENTER[2]))
    backing.AddScaleOp().Set(Gf.Vec3d(0.205, 0.020, 0.245))
    _bind(backing.GetPrim(), suit_adapter)
    created.append(str(backing.GetPath()))

    flange = UsdGeom.Cylinder.Define(stage, f"{MOUNT_ROOT}/OuterFlange")
    flange.CreateAxisAttr(UsdGeom.Tokens.y)
    flange.CreateRadiusAttr(0.145)
    flange.CreateHeightAttr(0.070)
    flange.AddTranslateOp().Set(Gf.Vec3d(MOUNT_CENTER[0], -0.382, MOUNT_CENTER[2]))
    _bind(flange.GetPrim(), structural)
    created.append(str(flange.GetPath()))

    collar = UsdGeom.Cylinder.Define(stage, f"{MOUNT_ROOT}/ArmCollar")
    collar.CreateAxisAttr(UsdGeom.Tokens.y)
    collar.CreateRadiusAttr(0.105)
    collar.CreateHeightAttr(0.070)
    collar.AddTranslateOp().Set(Gf.Vec3d(MOUNT_CENTER[0], -0.427, MOUNT_CENTER[2]))
    _bind(collar.GetPrim(), structural)
    created.append(str(collar.GetPath()))

    for index, angle_deg in enumerate((0, 60, 120, 180, 240, 300)):
        angle = math.radians(angle_deg)
        x = MOUNT_CENTER[0] + 0.119 * math.cos(angle)
        z = MOUNT_CENTER[2] + 0.119 * math.sin(angle)
        bolt = UsdGeom.Cylinder.Define(stage, f"{MOUNT_ROOT}/Bolt_{index + 1:02d}")
        bolt.CreateAxisAttr(UsdGeom.Tokens.y)
        bolt.CreateRadiusAttr(0.010)
        bolt.CreateHeightAttr(0.010)
        bolt.AddTranslateOp().Set(Gf.Vec3d(x, -0.421, z))
        _bind(bolt.GetPrim(), accent)
        created.append(str(bolt.GetPath()))
    return created


def configure(source: Path, output: Path, report_path: Path, lateral_offset_m: float) -> dict[str, object]:
    if source.resolve() == output.resolve():
        raise ValueError("output USD must differ from source USD")
    shutil.copy2(source.resolve(), output.resolve())
    stage = Usd.Stage.Open(str(output.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open copied scene: {output}")

    fixed_joints_before = {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)
    }
    if not PROTECTED_FIXED_JOINTS.issubset(fixed_joints_before):
        raise RuntimeError(
            f"source scene is missing protected FixedJoints: "
            f"{sorted(PROTECTED_FIXED_JOINTS - fixed_joints_before)}"
        )
    references = _repath_local_assets(stage)
    astronaut_translation = _move_astronaut_to_waist_mount(stage, lateral_offset_m)
    mount_prims = _create_mount(stage)
    stage.GetRootLayer().Save()

    audit_stage = Usd.Stage.Open(str(output.resolve()))
    if audit_stage is None:
        raise RuntimeError(f"cannot reopen output scene: {output}")
    fixed_joints_after = {
        str(prim.GetPath()) for prim in audit_stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)
    }
    if not PROTECTED_FIXED_JOINTS.issubset(fixed_joints_after):
        raise RuntimeError(
            f"output scene is missing protected FixedJoints: "
            f"{sorted(PROTECTED_FIXED_JOINTS - fixed_joints_after)}"
        )

    astronaut_skeleton_prims = []
    astronaut_physics_prims = []
    for prim in Usd.PrimRange(audit_stage.GetPrimAtPath(ASTRONAUT_ROOT)):
        if prim.IsA(UsdSkel.Root) or prim.IsA(UsdSkel.Skeleton) or prim.IsA(UsdSkel.Animation):
            astronaut_skeleton_prims.append(str(prim.GetPath()))
        if any(
            prim.HasAPI(api)
            for api in (
                UsdPhysics.CollisionAPI,
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.MassAPI,
                UsdPhysics.ArticulationRootAPI,
            )
        ):
            astronaut_physics_prims.append(str(prim.GetPath()))
    mount_physics_prims = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(audit_stage.GetPrimAtPath(MOUNT_ROOT))
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if astronaut_physics_prims or mount_physics_prims:
        raise RuntimeError("render-only astronaut or mount unexpectedly has physics schemas")

    astronaut_range = _world_range(audit_stage, ASTRONAUT_ROOT)
    mount_range = _world_range(audit_stage, MOUNT_ROOT)
    arm_range = _world_range(audit_stage, ACTIVE_ARM_ROOT)
    result = {
        "source_scene": str(source.resolve()),
        "output_scene": str(output.resolve()),
        "design": "left-waist-mounted astronaut auxiliary arm",
        "astronaut_translation": astronaut_translation,
        "astronaut_lateral_offset_from_original_m": float(lateral_offset_m),
        "astronaut_world_bounds_m": [list(astronaut_range.GetMin()), list(astronaut_range.GetMax())],
        "mount_world_bounds_m": [list(mount_range.GetMin()), list(mount_range.GetMax())],
        "active_arm_world_bounds_m": [list(arm_range.GetMin()), list(arm_range.GetMax())],
        "mount_center_world_m": [float(value) for value in MOUNT_CENTER],
        "mount_prims": mount_prims,
        "protected_fixed_joints_preserved": True,
        "protected_fixed_joint_paths": sorted(PROTECTED_FIXED_JOINTS),
        "resolved_fixed_joint_paths": sorted(fixed_joints_after),
        "local_asset_references": references,
        "astronaut_skeleton_prims": astronaut_skeleton_prims,
        "astronaut_physics_prims": astronaut_physics_prims,
        "mount_physics_prims": mount_physics_prims,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


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
        default=PROJECT_ROOT / "1111_nasa_astronaut_waist_mounted_aux_arm.usd",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/nasa_astronaut_waist_mounted/scene_audit.json",
    )
    parser.add_argument(
        "--astronaut_lateral_offset_m",
        type=float,
        default=0.055,
        help="Small separation from the original suit pose; the mount collar bridges it.",
    )
    args = parser.parse_args()
    result = configure(args.source, args.output, args.report, args.astronaut_lateral_offset_m)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
