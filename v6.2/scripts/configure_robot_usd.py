"""Configure the ZBot USD for fixed-base kinematic trajectory generation and self-collision.

The script never edits the root-level source USD. It copies the source to
assets/zbot_6s_new_configured.usd, enables PhysX articulation self-collision,
and adds filtered pairs for directly connected links.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "assets" / "zbot_6s_new.usd"
DEFAULT_BACKUP = PROJECT_ROOT / "assets" / "zbot_6s_new_backup.usd"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "zbot_6s_new_configured.usd"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "logs" / "usd_configuration_report.json"


def applied_schema_names(prim) -> list[str]:
    try:
        return [str(s) for s in prim.GetAppliedSchemas()]
    except Exception:
        return []


def has_api_name(prim, token: str) -> bool:
    token_low = token.lower()
    return any(token_low in schema.lower() for schema in applied_schema_names(prim))


def relationship_targets(prim) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rel in prim.GetRelationships():
        targets = rel.GetTargets()
        if targets:
            out[rel.GetName()] = [str(t) for t in targets]
    return out


def value_to_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "pathString"):
        return value.pathString
    if isinstance(value, (list, tuple)):
        return [value_to_json(v) for v in value]
    try:
        return list(value)
    except Exception:
        return str(value)


def nearest_ancestor_with_api(stage, prim, api_token: str) -> str | None:
    path = prim.GetPath()
    while path != path.absoluteRootPath:
        path = path.GetParentPath()
        ancestor = stage.GetPrimAtPath(path)
        if ancestor and ancestor.IsValid() and has_api_name(ancestor, api_token):
            return str(path)
    return None


def configure_usd(source: Path, backup: Path, output: Path) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdPhysics

    if not source.exists():
        raise FileNotFoundError(source)
    backup.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(source, backup)
    shutil.copy2(source, output)

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {output}")

    articulation_roots = []
    collision_by_link: dict[str, list[str]] = {}
    joints = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or has_api_name(prim, "Articulation"):
            articulation_roots.append(prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI) or has_api_name(prim, "Collision"):
            parent_link = nearest_ancestor_with_api(stage, prim, "RigidBody")
            if parent_link:
                collision_by_link.setdefault(parent_link, []).append(str(prim.GetPath()))
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                else:
                    report_api = PhysxSchema.PhysxContactReportAPI(prim)
                report_api.CreateThresholdAttr().Set(0.0)

        type_name = prim.GetTypeName()
        if "Joint" in str(type_name) or any("Joint" in s for s in applied_schema_names(prim)):
            rels = relationship_targets(prim)
            body0 = rels.get("physics:body0", [None])[0]
            body1 = rels.get("physics:body1", [None])[0]
            if body0 and body1:
                joints.append({"path": str(prim.GetPath()), "type": str(type_name), "body0": body0, "body1": body1})

    changed_roots = []
    for prim in articulation_roots:
        physx_api = PhysxSchema.PhysxArticulationAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        physx_api.CreateEnabledSelfCollisionsAttr().Set(True)
        physx_api.CreateArticulationEnabledAttr().Set(True)
        changed_roots.append(str(prim.GetPath()))

    filtered_pairs = []
    for joint in joints:
        body0 = joint["body0"]
        body1 = joint["body1"]
        for collider0 in collision_by_link.get(body0, []):
            collider0_prim = stage.GetPrimAtPath(collider0)
            if not collider0_prim.IsValid():
                continue
            filtered_api = UsdPhysics.FilteredPairsAPI.Apply(collider0_prim)
            rel = filtered_api.CreateFilteredPairsRel()
            for collider1 in collision_by_link.get(body1, []):
                rel.AddTarget(collider1)
                filtered_pairs.append(
                    {
                        "joint": joint["path"],
                        "joint_type": joint["type"],
                        "body0": body0,
                        "body1": body1,
                        "collider0": collider0,
                        "collider1": collider1,
                    }
                )

    stage.GetRootLayer().Save()
    verify_stage = Usd.Stage.Open(str(output))
    verify_roots = []
    for prim in verify_stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or has_api_name(prim, "Articulation"):
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            verify_roots.append({"path": str(prim.GetPath()), "enabledSelfCollisions": value_to_json(attr.Get())})

    return {
        "source": str(source),
        "backup": str(backup),
        "output": str(output),
        "articulation_roots_configured": changed_roots,
        "self_collision_verification": verify_roots,
        "collision_link_count": len(collision_by_link),
        "filtered_pair_count": len(filtered_pairs),
        "filtered_pairs": filtered_pairs,
        "contact_report_threshold_set_on_colliders": sum(len(v) for v in collision_by_link.values()),
        "notes": [
            "Self-collision is enabled at the PhysX articulation root.",
            "FilteredPairsAPI was applied only to directly joint-connected link collider pairs.",
            "The source USD at the project root and assets backup are not edited.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    try:
        report = configure_usd(args.source, args.backup, args.output)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {args.output}", flush=True)
        print(f"[OK] wrote {args.report}", flush=True)
        print(f"[INFO] filtered_pair_count={report['filtered_pair_count']}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
