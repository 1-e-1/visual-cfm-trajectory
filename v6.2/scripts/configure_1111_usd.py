"""Author assembly, drive, and collision fixes directly into 1111_copy.usd."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = WORKSPACE_ROOT / "1111_copy.usd"
BACKUP_PATH = WORKSPACE_ROOT / "1111_copy_before_training_20260801.usd"
AUDIT_PATH = WORKSPACE_ROOT / "outputs" / "usd_audit.json"

ROBOT_PATH = Sdf.Path("/World/UnitreeH1")
PELVIS_PATH = ROBOT_PATH.AppendChild("pelvis")
GRIPPER_ROOT_JOINTS = [
    ROBOT_PATH.AppendPath("ZBot6S_left/Robotiq2F85/root_joint"),
    ROBOT_PATH.AppendPath("ZBot6S_right/Robotiq2F85/root_joint"),
]
ASSEMBLY_JOINTS = {
    ROBOT_PATH.AppendPath("AssemblyJoints/CubeToTorso"): (
        ROBOT_PATH.AppendChild("torso_link"),
        ROBOT_PATH.AppendChild("Cube"),
    ),
    ROBOT_PATH.AppendPath("AssemblyJoints/RightAuxArmToCube"): (
        ROBOT_PATH.AppendChild("Cube"),
        ROBOT_PATH.AppendPath("ZBot6S_right/foot_0"),
    ),
    ROBOT_PATH.AppendPath("AssemblyJoints/LeftAuxArmToCube"): (
        ROBOT_PATH.AppendChild("Cube"),
        ROBOT_PATH.AppendPath("ZBot6S_left/foot_0"),
    ),
}
ARM_DRIVES = {
    "joint1": (800.0, 60.0, 300.0),
    "joint2": (800.0, 60.0, 300.0),
    "joint3": (700.0, 55.0, 300.0),
    "joint4": (450.0, 40.0, 220.0),
    "joint5": (350.0, 32.0, 160.0),
    "joint6": (250.0, 25.0, 100.0),
}

SEMANTIC_LABELS = {
    ROBOT_PATH: "h1_body",
    ROBOT_PATH.AppendChild("Cube"): "back_mount",
    ROBOT_PATH.AppendChild("ZBot6S_left"): "aux_arm_left",
    ROBOT_PATH.AppendChild("ZBot6S_right"): "aux_arm_right",
    Sdf.Path("/World/AuxWorkspaces"): "workspace_obstacle",
}
for side in ("LeftParts", "RightParts"):
    for object_name in ("Box", "Cylinder", "Sphere", "Cone", "Capsule"):
        SEMANTIC_LABELS[
            Sdf.Path(f"/World/AuxWorkspaces/{side}/{object_name}")
        ] = f"graspable_{object_name.lower()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_assembly_joint(
    stage: Usd.Stage, joint_path: Sdf.Path, body0: Sdf.Path, body1: Sdf.Path
) -> dict[str, object]:
    joint = UsdPhysics.FixedJoint.Get(stage, joint_path)
    if not joint:
        raise RuntimeError(f"Missing required FixedJoint: {joint_path}")
    joint.GetBody0Rel().SetTargets([body0])
    joint.GetBody1Rel().SetTargets([body1])
    xform_cache = UsdGeom.XformCache()
    body0_pose = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0))
    body1_pose = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1))
    relative_pose = (body1_pose * body0_pose.GetInverse()).RemoveScaleShear()
    local_pos0 = Gf.Vec3f(relative_pose.ExtractTranslation())
    local_rot0 = Gf.Quatf(relative_pose.ExtractRotationQuat())
    joint.CreateLocalPos0Attr().Set(local_pos0)
    joint.CreateLocalRot0Attr().Set(local_rot0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return {
        "path": str(joint_path),
        "body0": str(body0),
        "body1": str(body1),
        "local_pos0": list(local_pos0),
        "local_rot0": [local_rot0.GetReal(), *local_rot0.GetImaginary()],
    }


def _configure_drives(stage: Usd.Stage) -> list[dict[str, object]]:
    configured = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        normalized_name = re.sub(r"_0$", "", prim.GetName())
        if normalized_name in ARM_DRIVES:
            stiffness, damping, max_force = ARM_DRIVES[normalized_name]
        elif normalized_name == "finger_joint":
            stiffness, damping, max_force = (40.0, 4.0, 80.0)
        else:
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force").Set("force")
        drive.CreateStiffnessAttr(stiffness).Set(stiffness)
        drive.CreateDampingAttr(damping).Set(damping)
        drive.CreateMaxForceAttr(max_force).Set(max_force)
        drive.CreateTargetPositionAttr(0.0).Set(0.0)
        drive.CreateTargetVelocityAttr(0.0).Set(0.0)
        configured.append(
            {
                "path": str(prim.GetPath()),
                "stiffness": stiffness,
                "damping": damping,
                "max_force": max_force,
            }
        )
    if len(configured) != 14:
        raise RuntimeError(f"Expected 14 arm/gripper drives, found {len(configured)}")
    return configured


def _disable_embedded_physics_scenes(stage: Usd.Stage) -> list[str]:
    disabled = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.Scene):
            disabled.append(str(prim.GetPath()))
            prim.SetActive(False)
    return disabled


def _disable_mounting_block_collision(stage: Usd.Stage) -> list[str]:
    cube_path = ROBOT_PATH.AppendChild("Cube")
    disabled = []
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(cube_path):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
            collision.CreateCollisionEnabledAttr(False).Set(False)
            disabled.append(str(prim.GetPath()))
    if not disabled:
        raise RuntimeError(f"No collision shape found under mounting block: {cube_path}")
    return disabled


def _remove_unsupported_collision_groups(stage: Usd.Stage) -> list[str]:
    """Remove redundant groups that PhysX GPU replication cannot clone."""
    removed = []
    for path in ("/World/H1InternalCollisionGroup", "/World/H1InternalCollisionGroups"):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
            removed.append(path)
    return removed


def _canonicalize_camera_xforms(stage: Usd.Stage) -> list[dict[str, object]]:
    """Convert matrix camera transforms to Isaac Lab's canonical TRS ops."""
    configured = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Camera):
            continue
        xformable = UsdGeom.Xformable(prim)
        matrix = xformable.GetLocalTransformation()
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        reset_stack = xformable.GetResetXformStack()
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
        translate_op.Set(Gf.Vec3d(*translation))
        orient_op.Set(
            Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
            )
        )
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        xformable.SetXformOpOrder([translate_op, orient_op, scale_op], reset_stack)
        configured.append(
            {
                "path": str(prim.GetPath()),
                "translation": [float(value) for value in translation],
                "quaternion_wxyz": [
                    float(rotation.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                ],
            }
        )
    if len(configured) != 11:
        raise RuntimeError(f"Expected 11 cameras, canonicalized {len(configured)}")
    return configured


def _configure_semantic_labels(stage: Usd.Stage) -> list[dict[str, str]]:
    """Persist class labels used by RGB-D reconstruction in the USD root layer."""
    configured = []
    for path, label in SEMANTIC_LABELS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing semantic-label target: {path}")
        if prim.HasAPI(UsdSemantics.LabelsAPI, "class"):
            labels_api = UsdSemantics.LabelsAPI(prim, "class")
        else:
            labels_api = UsdSemantics.LabelsAPI.Apply(prim, "class")
        labels_api.CreateLabelsAttr().Set([label])
        configured.append({"path": str(path), "class": label})
    return configured


def _enable_required_collisions(stage: Usd.Stage) -> tuple[int, int, list[str]]:
    pelvis = stage.GetPrimAtPath(PELVIS_PATH)
    if not pelvis.IsValid():
        raise RuntimeError(f"Missing articulation root: {PELVIS_PATH}")
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    articulation.CreateEnabledSelfCollisionsAttr(True).Set(True)
    articulation.CreateSolverPositionIterationCountAttr(8).Set(8)
    articulation.CreateSolverVelocityIterationCountAttr(4).Set(4)

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

    pair_count = 0
    for index, body_path in enumerate(h1_bodies):
        targets = h1_bodies[index + 1 :]
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body_path))
        filtered_pairs.CreateFilteredPairsRel().SetTargets(targets)
        pair_count += len(targets)

    cube = stage.GetPrimAtPath(ROBOT_PATH.AppendChild("Cube"))
    if not cube.IsValid():
        raise RuntimeError("Missing fixed mounting block: /World/UnitreeH1/Cube")
    mounting_targets = h1_bodies + [
        ROBOT_PATH.AppendPath("ZBot6S_left/foot_0"),
        ROBOT_PATH.AppendPath("ZBot6S_right/foot_0"),
    ]
    cube_pairs = UsdPhysics.FilteredPairsAPI.Apply(cube)
    cube_pairs.CreateFilteredPairsRel().SetTargets(mounting_targets)
    return len(h1_bodies), pair_count, [str(path) for path in mounting_targets]


def configure() -> None:
    print("[USD] configure start", flush=True)
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(SCENE_PATH)
    if not BACKUP_PATH.exists():
        shutil.copy2(SCENE_PATH, BACKUP_PATH)

    before_hash = _sha256(SCENE_PATH)
    stage = Usd.Stage.Open(str(SCENE_PATH))
    if stage is None:
        raise RuntimeError(f"Unable to open USD stage: {SCENE_PATH}")
    stage.SetEditTarget(stage.GetRootLayer())
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        raise RuntimeError("Missing required scene root: /World")
    stage.GetRootLayer().defaultPrim = "World"

    assembly = [
        _set_assembly_joint(stage, path, body0, body1)
        for path, (body0, body1) in ASSEMBLY_JOINTS.items()
    ]
    print("[USD] assembly validated", flush=True)
    for path in GRIPPER_ROOT_JOINTS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing gripper root joint: {path}")
        prim.SetActive(False)

    drives = _configure_drives(stage)
    print("[USD] drives configured", flush=True)
    disabled_physics_scenes = _disable_embedded_physics_scenes(stage)
    disabled_mount_collisions = _disable_mounting_block_collision(stage)
    print("[USD] embedded scene and mount collision configured", flush=True)
    removed_collision_groups = _remove_unsupported_collision_groups(stage)
    print("[USD] unsupported CollisionGroup hierarchy removed", flush=True)
    canonical_cameras = _canonicalize_camera_xforms(stage)
    print("[USD] camera transforms canonicalized", flush=True)
    semantic_labels = _configure_semantic_labels(stage)
    print("[USD] semantic labels configured", flush=True)
    h1_body_count, filtered_pair_count, mounting_filtered_targets = _enable_required_collisions(stage)

    layer_data = dict(stage.GetRootLayer().customLayerData)
    layer_data.update(
        {
            "h1_1111_configuration": "direct-root-layer-edit-v1",
            "assembly": "torso_link--Cube--left/right foot_0 FixedJoint chain",
            "collision_policy": "H1 internal pairs filtered; arm-arm and arm-H1 enabled",
            "semantic_policy": "class labels authored for H1, both auxiliary arms, mount, workspace, and graspables",
            "disabled_world_joints": "left/right Robotiq2F85/root_joint",
            "default_prim": "/World; Isaac Lab spawner references /World/UnitreeH1 directly",
        }
    )
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    if stage.GetRootLayer().defaultPrim != "World":
        raise RuntimeError("Failed to persist root-layer defaultPrim=World")

    audit = {
        "scene": str(SCENE_PATH),
        "default_prim": stage.GetRootLayer().defaultPrim,
        "backup": str(BACKUP_PATH),
        "sha256_before": before_hash,
        "sha256_after": _sha256(SCENE_PATH),
        "assembly_joints": assembly,
        "disabled_gripper_root_joints": [str(path) for path in GRIPPER_ROOT_JOINTS],
        "configured_drives": drives,
        "disabled_embedded_physics_scenes": disabled_physics_scenes,
        "disabled_mounting_block_collisions": disabled_mount_collisions,
        "removed_unsupported_collision_groups": removed_collision_groups,
        "canonical_camera_transforms": canonical_cameras,
        "semantic_labels": semantic_labels,
        "self_collisions": True,
        "h1_internal_filtered_body_count": h1_body_count,
        "h1_internal_filtered_pair_count": filtered_pair_count,
        "mounting_block_filtered_targets": mounting_filtered_targets,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    print("[USD] main", flush=True)
    try:
        configure()
    finally:
        simulation_app.close()
