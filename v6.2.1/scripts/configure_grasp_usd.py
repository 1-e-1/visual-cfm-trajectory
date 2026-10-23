"""Persist Robotiq fingertip collision geometry in ``1111_copy.usd``.

The shipped gripper contains visual meshes but no usable collision geometry on
the inner fingers.  A physical pinch is impossible in that state.  This script
authors compact box colliders as children of the moving finger links and binds
a high-friction physics material.  It is idempotent and does not launch Isaac
Sim.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _add_box_collider(
    stage: Usd.Stage,
    prim_path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: UsdShade.Material,
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    cx, cy, cz = center
    hx, hy, hz = (0.5 * value for value in size)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(cx - hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy + hy, cz - hz),
            Gf.Vec3f(cx - hx, cy + hy, cz - hz),
            Gf.Vec3f(cx - hx, cy - hy, cz + hz),
            Gf.Vec3f(cx + hx, cy - hy, cz + hz),
            Gf.Vec3f(cx + hx, cy + hy, cz + hz),
            Gf.Vec3f(cx - hx, cy + hy, cz + hz),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([3] * 12)
    mesh.GetFaceVertexIndicesAttr().Set(
        [
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
            1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
        ]
    )
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material, materialPurpose="physics"
    )


def configure(scene_path: Path, backup_path: Path | None) -> dict[str, object]:
    if backup_path is not None and not backup_path.exists():
        shutil.copy2(scene_path, backup_path)

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Cannot open USD stage: {scene_path}")

    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/GripperPad")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.4)
    physics_material.CreateDynamicFrictionAttr(1.2)
    physics_material.CreateRestitutionAttr(0.0)
    finger_paths = []
    for side in ("left", "right"):
        root = f"/World/UnitreeH1/ZBot6S_{side}/Robotiq2F85"
        for finger in ("left_inner_finger", "right_inner_finger"):
            finger_path = f"{root}/{finger}"
            if not stage.GetPrimAtPath(finger_path).IsValid():
                raise RuntimeError(f"Missing gripper finger prim: {finger_path}")
            finger_paths.append(finger_path)
            collision_root = f"{finger_path}/collisions"
            UsdGeom.Scope.Define(stage, collision_root)
            _add_box_collider(
                stage,
                f"{collision_root}/shaft",
                (0.0, -0.0599, 0.1054),
                (0.027, 0.0299, 0.028),
                material,
            )
            _add_box_collider(
                stage,
                f"{collision_root}/pad",
                (0.0, -0.0531, 0.1294),
                (0.022, 0.0192, 0.038),
                material,
            )

    stage.GetRootLayer().Save()
    return {
        "scene": str(scene_path),
        "backup": None if backup_path is None else str(backup_path),
        "finger_links": finger_paths,
        "colliders": len(finger_paths) * 2,
        "static_friction": 1.4,
        "dynamic_friction": 1.2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PROJECT_ROOT / "1111_copy.usd")
    parser.add_argument(
        "--backup",
        type=Path,
        default=PROJECT_ROOT / "1111_copy_pre_grasp_pose.usd",
    )
    args = parser.parse_args()
    report = configure(args.scene.resolve(), args.backup.resolve())
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
