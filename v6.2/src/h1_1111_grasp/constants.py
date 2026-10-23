"""Shared paths, joint names, and periodic dual-arm sampling bounds."""

import math

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
PRETRAINED_DIR = PACKAGE_ROOT / "pretrained"

SCENE_ASSET_PATH = WORKSPACE_ROOT / "1111_copy.usd"

H1_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow",
]

DUAL_ARM_JOINT_PATTERN = ["joint[1-6](_0)?"]
DUAL_GRIPPER_JOINT_PATTERN = ["finger_joint(_0)?"]
DUAL_ARM_DEFAULT_Q = [0.0] * 12
# Isaac Lab resolves the two identically named arm joints in interleaved order:
# joint1, joint1_0, joint2, joint2_0, ..., joint6, joint6_0.
DUAL_ARM_LOWER = [-math.pi] * 12
DUAL_ARM_UPPER = [math.pi] * 12

NON_FOOT_CONTACT_BODY_PATTERN = (
    "foot_0|b1|a2|b2|a3|b3|"
    "base|b4|a5|b5|a6|foot_1|base_link|.*knuckle|.*finger"
)
