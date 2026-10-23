"""Persist balance and collision corrections directly into 1111_copy.usd."""

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = WORKSPACE_ROOT / "1111_copy.usd"
ROBOT_PATH = Sdf.Path("/World/UnitreeH1")

ARM_DRIVES = {
    "joint1": (800.0, 60.0, 300.0),
    "joint2": (800.0, 60.0, 300.0),
    "joint3": (700.0, 55.0, 300.0),
    "joint4": (450.0, 40.0, 220.0),
    "joint5": (350.0, 32.0, 160.0),
    "joint6": (250.0, 25.0, 100.0),
}

EXPECTED_MOUNTS = {
    "CubeToTorso": (
        "/World/UnitreeH1/torso_link",
        "/World/UnitreeH1/Cube",
    ),
    "RightAuxArmToCube": (
        "/World/UnitreeH1/Cube",
        "/World/UnitreeH1/ZBot6S_right/foot_0",
    ),
    "LeftAuxArmToCube": (
        "/World/UnitreeH1/Cube",
        "/World/UnitreeH1/ZBot6S_left/foot_0",
    ),
}


def _targets(prim: Usd.Prim, relationship: str) -> list[str]:
    return [str(path) for path in prim.GetRelationship(relationship).GetTargets()]


def _validate_mounts(stage: Usd.Stage) -> None:
    for name, (body0, body1) in EXPECTED_MOUNTS.items():
        prim = stage.GetPrimAtPath(ROBOT_PATH.AppendPath(f"AssemblyJoints/{name}"))
        if not prim.IsValid() or not prim.IsA(UsdPhysics.FixedJoint):
            raise RuntimeError(f"missing required FixedJoint: {name}")
        if _targets(prim, "physics:body0") != [body0]:
            raise RuntimeError(f"{name} body0 changed: {_targets(prim, 'physics:body0')}")
        if _targets(prim, "physics:body1") != [body1]:
            raise RuntimeError(f"{name} body1 changed: {_targets(prim, 'physics:body1')}")


def _filter_direct_mount_pair(stage: Usd.Stage, body0: str, body1: str) -> None:
    prim = stage.GetPrimAtPath(body0)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"mount body is not a rigid body: {body0}")
    api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    relationship = api.CreateFilteredPairsRel()
    targets = list(relationship.GetTargets())
    target = Sdf.Path(body1)
    if target not in targets:
        targets.append(target)
        relationship.SetTargets(targets)


def modify() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"failed to open {USD_PATH}")
    _validate_mounts(stage)

    for side in ("left", "right"):
        root_joint = ROBOT_PATH.AppendPath(f"ZBot6S_{side}/Robotiq2F85/root_joint")
        stage.OverridePrim(root_joint).SetActive(False)

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if prim.GetName() in ARM_DRIVES:
            stiffness, damping, max_force = ARM_DRIVES[prim.GetName()]
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(stiffness)
            drive.CreateDampingAttr(damping)
            drive.CreateMaxForceAttr(max_force)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
        elif prim.GetName() == "finger_joint":
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(40.0)
            drive.CreateDampingAttr(4.0)
            drive.CreateMaxForceAttr(80.0)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)

    pelvis = stage.GetPrimAtPath(ROBOT_PATH.AppendChild("pelvis"))
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    articulation.CreateEnabledSelfCollisionsAttr(True)

    # Directly constrained mounting faces must not generate persistent contact
    # impulses. All non-adjacent H1-arm and arm-arm collision pairs remain active.
    for body0, body1 in EXPECTED_MOUNTS.values():
        _filter_direct_mount_pair(stage, body0, body1)

    h1_bodies = []
    for prim in stage.TraverseAll():
        path = prim.GetPath()
        if not path.HasPrefix(ROBOT_PATH) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if path.HasPrefix(ROBOT_PATH.AppendChild("ZBot6S_left")):
            continue
        if path.HasPrefix(ROBOT_PATH.AppendChild("ZBot6S_right")):
            continue
        if prim.GetName() == "Cube":
            continue
        h1_bodies.append(path)
    for index, body_path in enumerate(h1_bodies):
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body_path))
        api.CreateFilteredPairsRel().SetTargets(h1_bodies[index + 1 :])

    layer_data = dict(stage.GetRootLayer().customLayerData)
    layer_data.update(
        {
            "directUsdConfiguration": "2026-08-01 balance and grasp safety",
            "mountTopology": "torso_link -> Cube -> left/right foot_0",
            "disabledWorldFixedJoints": "left/right Robotiq2F85/root_joint",
            "selfCollision": "enabled; stock H1-H1 and direct mount pairs filtered",
        }
    )
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    print(f"[OK] modified directly: {USD_PATH}", flush=True)
    print("[OK] preserved mounts: torso_link -> Cube -> left/right foot_0", flush=True)


if __name__ == "__main__":
    try:
        modify()
    finally:
        simulation_app.close()
