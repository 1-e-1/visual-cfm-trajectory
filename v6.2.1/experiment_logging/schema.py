"""Canonical task-level schema for supplementary planner and PhysX experiments."""

from __future__ import annotations

from dataclasses import dataclass


FAILURE_REASONS = {
    "ENV_COLLISION",
    "SELF_COLLISION",
    "TERMINAL_POSITION",
    "TERMINAL_ORIENTATION",
    "DYNAMIC_LIMIT",
    "NO_IK_OR_UNREACHABLE",
    "NO_VALID_CANDIDATE",
    "TARGET_ASSOCIATION",
    "GRASP_CONTACT",
    "OBJECT_SLIP",
    "PLACE_ERROR",
    "EMERGENCY_STOP",
    "UNKNOWN",
}


@dataclass(frozen=True)
class Column:
    name: str
    unit: str
    required: bool = True


COLUMNS = (
    Column("experiment_id", "text"),
    Column("task_id", "text"),
    Column("scene_id", "text"),
    Column("method", "text"),
    Column("seed", "integer"),
    Column("q_start_id", "text"),
    Column("grasp_goal_id", "text"),
    Column("place_goal_id", "text"),
    Column("obstacle_count", "count"),
    Column("minimum_passage_clearance", "m"),
    Column("start_goal_joint_distance", "rad", False),
    Column("start_goal_tcp_distance", "m", False),
    Column("goal_local_obstacle_density", "1/m^3", False),
    Column("success_grasp", "bool"),
    Column("success_place", "bool"),
    Column("success_joint", "bool"),
    Column("full_task_success", "bool"),
    Column("planning_time_ms_grasp", "ms"),
    Column("planning_time_ms_place", "ms"),
    Column("path_length_grasp", "rad"),
    Column("path_length_place", "rad"),
    Column("min_env_clearance_grasp", "m"),
    Column("min_env_clearance_place", "m"),
    Column("min_self_clearance_grasp", "m"),
    Column("min_self_clearance_place", "m"),
    Column("terminal_pos_error_grasp", "m"),
    Column("terminal_pos_error_place", "m"),
    Column("terminal_rot_error_grasp", "rad"),
    Column("terminal_rot_error_place", "rad"),
    Column("acc_rms_grasp", "rad/s^2"),
    Column("acc_rms_place", "rad/s^2"),
    Column("jerk_rms_grasp", "rad/s^3"),
    Column("jerk_rms_place", "rad/s^3"),
    Column("duration_grasp", "s"),
    Column("duration_place", "s"),
    Column("grip_relative_pos_drift_rms", "m"),
    Column("grip_relative_rot_drift_rms", "rad"),
    Column("contact_force_peak", "N"),
    Column("contact_force_std", "N"),
    Column("object_angvel_peak", "rad/s"),
    Column("base_reaction_force_peak", "N", False),
    Column("base_reaction_torque_peak", "N*m", False),
    Column("emergency_stop", "bool"),
    Column("trajectory_file", "path"),
    Column("failure_reason", "enum"),
)

SUMMARY_COLUMNS = tuple(column.name for column in COLUMNS)
REQUIRED_COLUMNS = tuple(column.name for column in COLUMNS if column.required)
UNITS = {column.name: column.unit for column in COLUMNS}


TIMESERIES_FIELDS = {
    "t": "s",
    "q": "rad [T,6]",
    "qd": "rad/s [T,6]",
    "qdd": "rad/s^2 [T,6]",
    "qddd": "rad/s^3 [T,6]",
    "gripper_left_force": "N [T]",
    "gripper_right_force": "N [T]",
    "object_world_pos": "m [T,3]",
    "object_world_quat": "wxyz [T,4]",
    "gripper_world_pos": "m [T,3]",
    "gripper_world_quat": "wxyz [T,4]",
    "object_in_gripper_pos": "m [T,3]",
    "object_in_gripper_quat": "wxyz [T,4]",
    "object_linear_velocity": "m/s [T,3]",
    "object_angular_velocity": "rad/s [T,3]",
    "base_reaction_force": "N [T,3]",
    "base_reaction_torque": "N*m [T,3]",
    "state_machine_phase": "text [T]",
    "emergency_stop": "bool [T]",
}
