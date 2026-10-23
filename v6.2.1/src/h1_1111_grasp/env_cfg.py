"""Isaac Lab H1 balance environments for the corrected 1111 equipment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_assets import H1_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg

from . import mdp
from .usd_spawner import spawn_1111_robot
from .constants import (
    DUAL_ARM_DEFAULT_Q,
    DUAL_ARM_JOINT_PATTERN,
    DUAL_ARM_LOWER,
    DUAL_ARM_UPPER,
    DUAL_GRIPPER_JOINT_PATTERN,
    H1_JOINT_NAMES,
    NON_FOOT_CONTACT_BODY_PATTERN,
    SCENE_ASSET_PATH,
)


H1_1111_CFG = H1_CFG.copy()
H1_1111_CFG.spawn.usd_path = str(SCENE_ASSET_PATH)
H1_1111_CFG.spawn.func = spawn_1111_robot
H1_1111_CFG.spawn.activate_contact_sensors = True
H1_1111_CFG.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)
H1_1111_CFG.init_state.joint_pos.update(
    {"joint[1-6](_0)?": 0.0, "finger_joint(_0)?": 0.0}
)
H1_1111_CFG.actuators["dual_auxiliary_arms"] = ImplicitActuatorCfg(
    joint_names_expr=DUAL_ARM_JOINT_PATTERN,
    effort_limit_sim=300.0,
    velocity_limit_sim=2.0,
    stiffness={
        "joint1(_0)?": 800.0, "joint2(_0)?": 800.0, "joint3(_0)?": 700.0,
        "joint4(_0)?": 450.0, "joint5(_0)?": 350.0, "joint6(_0)?": 250.0,
    },
    damping={
        "joint1(_0)?": 60.0, "joint2(_0)?": 60.0, "joint3(_0)?": 55.0,
        "joint4(_0)?": 40.0, "joint5(_0)?": 32.0, "joint6(_0)?": 25.0,
    },
)
H1_1111_CFG.actuators["dual_grippers"] = ImplicitActuatorCfg(
    joint_names_expr=DUAL_GRIPPER_JOINT_PATTERN,
    effort_limit_sim=80.0,
    velocity_limit_sim=1.0,
    stiffness=40.0,
    damping=4.0,
)


@configclass
class H11111BalanceEnvCfg(H1FlatEnvCfg):
    """95-D stationary balance task with two independently moving payload arms."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 3.0
        self.scene.robot = H1_1111_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23

        h1 = SceneEntityCfg("robot", joint_names=H1_JOINT_NAMES, preserve_order=True)
        dual = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_PATTERN, preserve_order=True)
        self.actions.joint_pos.joint_names = H1_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {"asset_cfg": h1}
        self.observations.policy.joint_vel.params = {"asset_cfg": h1}
        self.observations.policy.base_xy_relative = ObsTerm(
            func=mdp.base_xy_relative,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        self.observations.policy.auxiliary_joint_pos = ObsTerm(
            func=locomotion_mdp.joint_pos_rel,
            params={"asset_cfg": dual},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        self.observations.policy.auxiliary_joint_vel = ObsTerm(
            func=locomotion_mdp.joint_vel_rel,
            params={"asset_cfg": dual},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_robot_joints.params["asset_cfg"] = h1
        self.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        for key in self.events.reset_base.params["velocity_range"]:
            self.events.reset_base.params["velocity_range"][key] = (-0.05, 0.05)
        self.events.reset_dual_aux_arms = EventTerm(
            func=mdp.reset_dual_aux_arms,
            mode="reset",
            params={
                "asset_cfg": dual,
                "default_positions": DUAL_ARM_DEFAULT_Q,
                "position_noise": 0.03,
                "lower_limits": DUAL_ARM_LOWER,
                "upper_limits": DUAL_ARM_UPPER,
            },
        )
        self.events.move_dual_aux_arms = EventTerm(
            func=mdp.nudge_dual_aux_arm_targets,
            mode="interval",
            interval_range_s=(0.6, 1.4),
            params={
                "asset_cfg": dual,
                "max_delta": 0.06,
                "lower_limits": DUAL_ARM_LOWER,
                "upper_limits": DUAL_ARM_UPPER,
            },
        )
        self.events.base_external_force_torque.params["force_range"] = (-15.0, 15.0)
        self.events.base_external_force_torque.params["torque_range"] = (-6.0, 6.0)

        self.rewards.alive = RewTerm(func=locomotion_mdp.is_alive, weight=2.0)
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_exp, weight=1.5, params={"target_height": 1.05, "std": 0.12}
        )
        self.rewards.base_xy_position = RewTerm(
            func=mdp.base_xy_position_exp, weight=2.0, params={"std": 0.12}
        )
        self.rewards.base_xy_position_l2 = RewTerm(func=mdp.base_xy_position_l2, weight=-25.0)
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.feet_air_time = None
        self.rewards.aux_arm_velocity = RewTerm(
            func=mdp.aux_arm_joint_velocity_l2, weight=-2.0e-3, params={"asset_cfg": dual}
        )
        self.rewards.aux_arm_acceleration = RewTerm(
            func=mdp.aux_arm_joint_acceleration_l2, weight=-1.0e-7, params={"asset_cfg": dual}
        )
        # Fixed-joint reaction forces from torso_link -> Cube -> arms are exposed
        # by ContactSensor as net forces, so they are not valid collision labels.
        # Manipulator collisions are checked by the visual/PhysX trajectory shield.
        self.rewards.auxiliary_collision = None
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.dof_acc_l2.weight = -3.0e-7
        self.rewards.dof_torques_l2.weight = -1.5e-6
        self.terminations.base_contact = None
        self.terminations.low_height = DoneTerm(
            func=locomotion_mdp.root_height_below_minimum, params={"minimum_height": 0.65}
        )
        self.terminations.excessive_tilt = DoneTerm(
            func=locomotion_mdp.bad_orientation, params={"limit_angle": 0.85}
        )
        self.terminations.base_xy_drift = DoneTerm(
            func=mdp.base_xy_drift_exceeded, params={"maximum_distance": 0.30}
        )


@configclass
class H11111ManipulationEnvCfg(H11111BalanceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.move_dual_aux_arms.interval_range_s = (0.30, 0.80)
        self.events.move_dual_aux_arms.params["max_delta"] = 0.12
        self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (-12.0, 12.0)
        self.rewards.base_xy_position.weight = 3.0
        self.rewards.base_xy_position_l2.weight = -40.0


@configclass
class H11111RobustEnvCfg(H11111ManipulationEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.move_dual_aux_arms.interval_range_s = (0.25, 0.65)
        self.events.move_dual_aux_arms.params["max_delta"] = 0.16
        self.events.base_external_force_torque.params["force_range"] = (-45.0, 45.0)
        self.events.base_external_force_torque.params["torque_range"] = (-20.0, 20.0)
        self.rewards.base_xy_position.weight = 4.0
        self.rewards.base_xy_position_l2.weight = -55.0
        self.events.push_robot = EventTerm(
            func=locomotion_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {"x": (-0.45, 0.45), "y": (-0.45, 0.45)}},
        )


@configclass
class H11111BalanceEnvCfgPlay(H11111RobustEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False


@configclass
class H11111PPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "h1_1111_balance"
        self.max_iterations = 700
        self.save_interval = 50
        self.seed = 20260801
        self.algorithm.learning_rate = 3.0e-4
