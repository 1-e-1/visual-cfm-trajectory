r"""Export the USD joint graph into a pure-Torch differentiable FK config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gf_vec(value: Any) -> list[float]:
    return [float(value[index]) for index in range(3)]


def gf_quat(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])]


def nearest_rigid_body(stage, prim) -> str | None:
    from pxr import UsdPhysics

    path = prim.GetPath().GetParentPath()
    while path != path.absoluteRootPath:
        ancestor = stage.GetPrimAtPath(path)
        if ancestor.IsValid() and ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(path)
        path = path.GetParentPath()
    return None


def extract(usd_path: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open {usd_path}")
    articulation_roots = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(articulation_roots) != 1:
        raise RuntimeError(f"Expected one articulation root, found {len(articulation_roots)}")
    root_body = str(articulation_roots[0].GetPath())

    tcp_candidates = [
        prim
        for prim in stage.Traverse()
        if prim.GetTypeName() == "Xform" and "tcp" in prim.GetName().lower()
    ]
    if len(tcp_candidates) != 1:
        raise RuntimeError(f"Expected one TCP Xform, found {[str(p.GetPath()) for p in tcp_candidates]}")
    tcp_prim = tcp_candidates[0]
    tcp_parent_body = nearest_rigid_body(stage, tcp_prim)
    if tcp_parent_body is None:
        raise RuntimeError("TCP does not have a rigid-body ancestor")

    edges = []
    for prim in stage.Traverse():
        if "Joint" not in prim.GetTypeName():
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            continue
        joint_type = "revolute" if "Revolute" in prim.GetTypeName() else "fixed" if "Fixed" in prim.GetTypeName() else None
        if joint_type is None:
            continue
        edge = {
            "name": prim.GetName(),
            "path": str(prim.GetPath()),
            "joint_type": joint_type,
            "body0": str(body0_targets[0]),
            "body1": str(body1_targets[0]),
            "local_pos0": gf_vec(joint.GetLocalPos0Attr().Get()),
            "local_quat0_wxyz": gf_quat(joint.GetLocalRot0Attr().Get()),
            "local_pos1": gf_vec(joint.GetLocalPos1Attr().Get()),
            "local_quat1_wxyz": gf_quat(joint.GetLocalRot1Attr().Get()),
            "axis": str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if joint_type == "revolute" else None,
        }
        edges.append(edge)

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(edge["body0"], []).append(edge)
        adjacency.setdefault(edge["body1"], []).append(edge)

    def find_path(current: str, target: str, used: set[str]) -> list[tuple[dict[str, Any], bool]] | None:
        if current == target:
            return []
        for edge in adjacency.get(current, []):
            if edge["path"] in used:
                continue
            forward = edge["body0"] == current
            next_body = edge["body1"] if forward else edge["body0"]
            result = find_path(next_body, target, used | {edge["path"]})
            if result is not None:
                return [(edge, forward)] + result
        return None

    graph_path = find_path(root_body, tcp_parent_body, set())
    if graph_path is None:
        raise RuntimeError(f"No joint path from {root_body} to {tcp_parent_body}")

    chain = []
    controllable_names = []
    for edge, forward in graph_path:
        if forward:
            item = {
                "name": edge["name"],
                "path": edge["path"],
                "joint_type": edge["joint_type"],
                "body_from": edge["body0"],
                "body_to": edge["body1"],
                "local_pos_from": edge["local_pos0"],
                "local_quat_from_wxyz": edge["local_quat0_wxyz"],
                "local_pos_to": edge["local_pos1"],
                "local_quat_to_wxyz": edge["local_quat1_wxyz"],
                "axis": edge["axis"],
                "angle_sign": 1.0,
            }
        else:
            item = {
                "name": edge["name"],
                "path": edge["path"],
                "joint_type": edge["joint_type"],
                "body_from": edge["body1"],
                "body_to": edge["body0"],
                "local_pos_from": edge["local_pos1"],
                "local_quat_from_wxyz": edge["local_quat1_wxyz"],
                "local_pos_to": edge["local_pos0"],
                "local_quat_to_wxyz": edge["local_quat0_wxyz"],
                "axis": edge["axis"],
                "angle_sign": -1.0,
            }
        if item["joint_type"] == "revolute":
            item["controlled_index"] = len(controllable_names)
            controllable_names.append(item["name"])
        else:
            item["controlled_index"] = -1
        chain.append(item)

    local_result = UsdGeom.Xformable(tcp_prim).GetLocalTransformation()
    tcp_matrix = local_result[0] if isinstance(local_result, tuple) else local_result
    tcp_translation = tcp_matrix.ExtractTranslation()
    tcp_rotation = tcp_matrix.ExtractRotationQuat()
    result = {
        "source_usd": str(usd_path),
        "articulation_root_body": root_body,
        "tcp_prim_path": str(tcp_prim.GetPath()),
        "tcp_parent_body": tcp_parent_body,
        "joint_names": controllable_names,
        "chain": chain,
        "tcp_local_position": gf_vec(tcp_translation),
        "tcp_local_quaternion_wxyz": gf_quat(tcp_rotation),
        "quaternion_convention": "wxyz",
        "joint_position_unit": "radian",
    }
    if len(controllable_names) != 6:
        raise RuntimeError(f"Expected 6 revolute joints in chain, got {controllable_names}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=PROJECT_ROOT / "assets" / "zbot_6s_new_configured.usd")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs" / "kinematic_chain.json")
    args = parser.parse_args()
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    try:
        result = extract(args.usd)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {args.output}", flush=True)
        print(f"[INFO] joint_names={result['joint_names']}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
