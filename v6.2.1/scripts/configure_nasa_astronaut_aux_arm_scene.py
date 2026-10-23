"""Build a visual NASA ACES astronaut scene with one native-contact auxiliary arm.

The existing H1 hierarchy is retained only as an invisible controller-compatible
carrier for the validated arm articulation.  The visible human is the complete
NASA ACES mesh.  It is static render geometry and has no collision, rigid-body,
mass, or articulation schemas.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H1_ROOT = "/World/UnitreeH1"
ACTIVE_ARM_ROOT = f"{H1_ROOT}/ZBot6S_left"
ASTRONAUT_ROOT = "/World/NASA_Astronaut_ACES"
ASTRONAUT_ASSET_ROOT = f"{ASTRONAUT_ROOT}/Asset"
ASTRONAUT_ASSET = PROJECT_ROOT / "assets/astronaut_nasa_aces/Advanced_Crew_Escape_Suit.usd"


def _set_invisible(prim: Usd.Prim) -> None:
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeInvisible()


def _remove_physics_schemas(prim: Usd.Prim) -> None:
    for api in (
        UsdPhysics.CollisionAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.ArticulationRootAPI,
    ):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)


def _make_material(
    stage: Usd.Stage, path: str, color: tuple[float, float, float], metallic: float, roughness: float
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _world_range(stage: Usd.Stage, prim_path: str) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()


def configure(source: Path, output: Path, report_path: Path) -> dict[str, object]:
    if source.resolve() == output.resolve():
        raise ValueError("the astronaut scene must not overwrite its source scene")
    if not ASTRONAUT_ASSET.is_file():
        raise FileNotFoundError(ASTRONAUT_ASSET)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"cannot open copied stage: {output}")

    # Remove the previous procedural astronaut appearance if the source ever
    # contains it, then hide every old robot visual except the active arm.
    old_visual = stage.GetPrimAtPath(f"{H1_ROOT}/torso_link/AstronautVisual")
    if old_visual.IsValid():
        stage.RemovePrim(old_visual.GetPath())
    hidden_gprims: list[str] = []
    hidden_legacy_roots: list[str] = []
    disabled_colliders: list[str] = []
    # Author visibility on body link roots as well as leaf meshes.  Fabric can
    # rebuild leaf render buckets during articulation initialization, while an
    # invisible ancestor remains authoritative.
    for child in stage.GetPrimAtPath(H1_ROOT).GetChildren():
        path = str(child.GetPath())
        if path == ACTIVE_ARM_ROOT:
            continue
        if path.startswith(f"{H1_ROOT}/ZBot6S_right"):
            _set_invisible(child)
            UsdGeom.Imageable(child).CreatePurposeAttr(UsdGeom.Tokens.guide)
            hidden_legacy_roots.append(path)
            continue
        if child.GetName() in {"Looks", "AssemblyJoints", "FixedCameras"}:
            continue
        _set_invisible(child)
        UsdGeom.Imageable(child).CreatePurposeAttr(UsdGeom.Tokens.guide)
        hidden_legacy_roots.append(path)
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not path.startswith(H1_ROOT):
            continue
        if prim.IsA(UsdGeom.Gprim) and not path.startswith(ACTIVE_ARM_ROOT):
            _set_invisible(prim)
            hidden_gprims.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not path.startswith(ACTIVE_ARM_ROOT):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled_colliders.append(path)

    if stage.GetPrimAtPath(ASTRONAUT_ROOT).IsValid():
        stage.RemovePrim(ASTRONAUT_ROOT)
    astronaut = UsdGeom.Xform.Define(stage, ASTRONAUT_ROOT)
    astronaut_asset = UsdGeom.Xform.Define(stage, ASTRONAUT_ASSET_ROOT)
    astronaut_asset.GetPrim().GetReferences().AddReference(str(ASTRONAUT_ASSET.resolve()))
    astronaut.GetPrim().SetCustomDataByKey("astronaut:source", "NASA Advanced Crew Escape Suit")
    astronaut.GetPrim().SetCustomDataByKey("astronaut:renderOnly", True)

    # Source GLB axes: X=left/right, Y=up, Z=depth.  A +120 degree rotation
    # around (1,1,1) maps them to USD Y=left/right, Z=up, X=depth.
    # The translation centers the 1.996 m source at world z=1.0 after 0.9 scale.
    astronaut.AddTranslateOp().Set(Gf.Vec3d(-4.1568, 1.0087, 0.1000))
    astronaut.AddOrientOp().Set(Gf.Quatf(0.5, Gf.Vec3f(0.5, 0.5, 0.5)))
    astronaut.AddScaleOp().Set(Gf.Vec3d(0.9, 0.9, 0.9))

    # Strip all physics schemas authored by the imported asset at every
    # composed descendant.  This creates stronger delete list-ops in this scene.
    render_only_paths: list[str] = []
    for prim in list(Usd.PrimRange(astronaut.GetPrim())):
        _remove_physics_schemas(prim)
        prim.SetCustomDataByKey("astronaut:renderOnly", True)
        render_only_paths.append(str(prim.GetPath()))

    # Isaac Sim 5.1's glTF converter maps this asset's materials to
    # transmission_factor=1, making the suit invisible while still casting a
    # shadow.  Override only transmission; preserve NASA's original colors.
    corrected_materials: list[str] = []
    for prim in Usd.PrimRange(astronaut.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        transmission = shader.GetInput("transmission_factor")
        if transmission:
            transmission.Set(0.0)
            corrected_materials.append(str(prim.GetPath()))

    # A visible waist-side mounting plate explains the rigid installation.  It
    # is appearance-only; the existing validated FixedJoint remains internal.
    looks = UsdGeom.Scope.Define(stage, "/World/AstronautDemoLooks")
    mount_material = _make_material(
        stage, f"{looks.GetPath()}/AuxArmMount", (0.055, 0.065, 0.075), 0.65, 0.24
    )
    mount = UsdGeom.Cube.Define(stage, "/World/AuxArmWaistMount")
    mount.CreateSizeAttr(1.0)
    mount.AddTranslateOp().Set(Gf.Vec3d(0.0735, -0.414, 1.1432))
    mount.AddRotateZOp().Set(90.0)
    mount.AddScaleOp().Set(Gf.Vec3d(0.19, 0.034, 0.22))
    UsdShade.MaterialBindingAPI.Apply(mount.GetPrim()).Bind(mount_material)
    mount.GetPrim().SetCustomDataByKey("astronaut:renderOnly", True)

    stage.GetRootLayer().Save()

    # Reopen to audit the final composed result, including referenced schemas.
    audit_stage = Usd.Stage.Open(str(output))
    physics_violations: list[dict[str, object]] = []
    for prim in Usd.PrimRange(audit_stage.GetPrimAtPath(ASTRONAUT_ROOT)):
        schemas = []
        for api in (
            UsdPhysics.CollisionAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI,
        ):
            if prim.HasAPI(api):
                schemas.append(api.__name__)
        if schemas:
            physics_violations.append({"path": str(prim.GetPath()), "schemas": schemas})
    if physics_violations:
        raise RuntimeError(f"astronaut visual physics audit failed: {physics_violations}")

    astronaut_range = _world_range(audit_stage, ASTRONAUT_ROOT)
    active_arm_range = _world_range(audit_stage, ACTIVE_ARM_ROOT)
    result = {
        "source_scene": str(source),
        "output_scene": str(output),
        "astronaut_asset": str(ASTRONAUT_ASSET),
        "astronaut_asset_source_url": (
            "https://science.nasa.gov/3d-resources/advanced-crew-escape-suit/"
        ),
        "astronaut_world_min_m": list(astronaut_range.GetMin()),
        "astronaut_world_max_m": list(astronaut_range.GetMax()),
        "astronaut_world_size_m": list(astronaut_range.GetSize()),
        "active_arm_root": ACTIVE_ARM_ROOT,
        "active_arm_world_min_m": list(active_arm_range.GetMin()),
        "active_arm_world_max_m": list(active_arm_range.GetMax()),
        "mount_center_world_m": [0.0735, -0.414, 1.1432],
        "mount_size_m": [0.19, 0.034, 0.22],
        "mount_rotation_world_z_deg": 90.0,
        "hidden_legacy_gprims": len(hidden_gprims),
        "hidden_legacy_roots": hidden_legacy_roots,
        "disabled_legacy_colliders": len(disabled_colliders),
        "astronaut_render_only_prim_count": len(render_only_paths),
        "corrected_transmission_materials": len(corrected_materials),
        "astronaut_physics_schema_violations": physics_violations,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "1111_copy_native_contact.usd"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "1111_nasa_astronaut_aux_arm_20tasks.usd",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/nasa_astronaut_aux_arm_20tasks/scene_audit.json",
    )
    args = parser.parse_args()
    result = configure(args.source.resolve(), args.output.resolve(), args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
