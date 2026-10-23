"""Fixed-base Isaac Lab scene for FK, Jacobian, and PhysX contact queries."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, sleep
from typing import Callable

import numpy as np
import torch


class FixedBaseKinematicScene:
    """One fixed robot instance used for deterministic kinematic data generation."""

    def __init__(
        self,
        usd_path: str | Path,
        joint_names: list[str],
        tcp_body_name: str = "foot_1",
        tcp_prim_name: str = "TCP",
        device: str = "cuda:0",
        physics_dt: float = 1.0 / 120.0,
        contact_threshold: float = 1.0e-3,
        pre_reset_setup: Callable[["FixedBaseKinematicScene"], None] | None = None,
    ) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self._math_utils = __import__("isaaclab.utils.math", fromlist=["combine_frame_transforms"])
        self.device_name = device
        self.contact_threshold = float(contact_threshold)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=device))
        self.robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(usd_path).resolve()),
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        fix_root_link=True,
                        enabled_self_collisions=True,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_linear_velocity=5.0,
                        max_angular_velocity=720.0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
                actuators={
                    "position_hold": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        effort_limit_sim=1.0e9,
                        velocity_limit_sim=100.0,
                        stiffness=1.0e6,
                        damping=1.0e4,
                    )
                },
            )
        )
        self.contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Robot/.*",
                update_period=0.0,
                history_length=1,
                force_threshold=self.contact_threshold,
            )
        )
        if pre_reset_setup is not None:
            pre_reset_setup(self)
        self.sim.reset()
        self.sim_dt = self.sim.get_physics_dt()
        self.robot.update(self.sim_dt)
        self.contact_sensor.update(self.sim_dt)
        self._update_extension_entities()

        self.joint_ids = []
        resolved_names = []
        for name in joint_names:
            ids, names = self.robot.find_joints([f"^{name}$"])
            if len(ids) != 1:
                raise RuntimeError(f"Expected one joint named {name}, got {names}")
            self.joint_ids.append(ids[0])
            resolved_names.append(names[0])
        self.joint_names = resolved_names
        self.tcp_body_ids, tcp_body_names = self.robot.find_bodies([f"^{tcp_body_name}$"])
        if len(self.tcp_body_ids) != 1:
            raise RuntimeError(f"Expected one TCP parent body named {tcp_body_name}, got {tcp_body_names}")
        self.tcp_body_id = self.tcp_body_ids[0]
        self.tcp_body_name = tcp_body_names[0]
        self.tcp_prim_path = f"/World/Robot/{tcp_body_name}/{tcp_prim_name}"
        self.tcp_offset_pos, self.tcp_offset_quat = self._read_tcp_local_pose(self.tcp_prim_path)

    def _reset_extension_sensors(self) -> None:
        """Reset sensors owned by a scene extension."""

    def _write_extension_data(self) -> None:
        """Write extension-owned assets before each simulation step."""

    def _update_extension_entities(self) -> None:
        """Refresh extension-owned assets and sensors after a simulation step."""

    def _close_extension_entities(self) -> None:
        """Release extension-owned references before SimulationContext teardown."""

    def _read_tcp_local_pose(self, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"TCP prim does not exist: {prim_path}")
        local_result = UsdGeom.Xformable(prim).GetLocalTransformation()
        matrix = local_result[0] if isinstance(local_result, tuple) else local_result
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        pos = torch.tensor(
            [[float(translation[0]), float(translation[1]), float(translation[2])]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        quat = torch.tensor(
            [[float(rotation.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-12)
        return pos, quat

    def _set_state_and_step(self, q: np.ndarray, settle_steps: int, render: bool = False) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float32)
        if q_np.shape != (6,) or not np.all(np.isfinite(q_np)):
            raise ValueError("q must be finite with shape (6,)")
        q_tensor = torch.as_tensor(q_np, dtype=torch.float32, device=self.sim.device).reshape(1, 6)
        zero_velocity = torch.zeros_like(q_tensor)

        self.robot.write_joint_state_to_sim(q_tensor, zero_velocity, joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(q_tensor, joint_ids=self.joint_ids)
        self.contact_sensor.reset()
        self._reset_extension_sensors()
        for _ in range(settle_steps):
            self.robot.set_joint_position_target(q_tensor, joint_ids=self.joint_ids)
            self.robot.write_data_to_sim()
            self._write_extension_data()
            self.sim.step(render=render)
            self.robot.update(self.sim_dt)
            self.contact_sensor.update(self.sim_dt)
            self._update_extension_entities()
        return q_np

    def _read_tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos = self.robot.data.body_pos_w[:, self.tcp_body_id]
        body_quat = self.robot.data.body_quat_w[:, self.tcp_body_id]
        tcp_pos, tcp_quat = self._math_utils.combine_frame_transforms(
            body_pos, body_quat, self.tcp_offset_pos, self.tcp_offset_quat
        )
        if tcp_quat[0, 0] < 0.0:
            tcp_quat = -tcp_quat
        return body_pos, body_quat, tcp_pos, tcp_quat

    def _read_contacts(self) -> tuple[bool, list[str], float]:
        forces = self.contact_sensor.data.net_forces_w
        force_norms = (
            torch.linalg.norm(forces[0], dim=-1)
            if forces is not None
            else torch.zeros(0, device=self.sim.device)
        )
        contact_indices = torch.nonzero(force_norms > self.contact_threshold).flatten().tolist()
        body_names = list(getattr(self.contact_sensor, "body_names", []))
        contact_bodies = [
            body_names[index] if index < len(body_names) else f"body_index_{index}"
            for index in contact_indices
        ]
        max_contact_force = float(force_norms.max().item()) if force_norms.numel() else 0.0
        return max_contact_force > self.contact_threshold, contact_bodies, max_contact_force

    def query(self, q: np.ndarray, settle_steps: int = 3) -> dict[str, object]:
        q_np = self._set_state_and_step(q, settle_steps)
        body_pos, body_quat, tcp_pos, tcp_quat = self._read_tcp_pose()
        jacobian = self._tcp_jacobian(body_pos, body_quat, tcp_pos)
        singular_values = torch.linalg.svdvals(jacobian[0]).detach().cpu().numpy()
        min_singular = float(np.min(singular_values))
        condition_number = float(np.max(singular_values) / max(min_singular, 1.0e-12))
        collision, contact_bodies, max_contact_force = self._read_contacts()
        root_pos = self.robot.data.root_pos_w[0].detach().cpu().numpy()

        return {
            "q": q_np.astype(np.float64),
            "tcp_position": tcp_pos[0].detach().cpu().numpy().astype(np.float64),
            "tcp_quaternion_wxyz": tcp_quat[0].detach().cpu().numpy().astype(np.float64),
            "jacobian": jacobian[0].detach().cpu().numpy().astype(np.float64),
            "jacobian_singular_values": singular_values.astype(np.float64),
            "min_jacobian_singular_value": min_singular,
            "jacobian_condition_number": condition_number,
            "collision": collision,
            "contact_bodies": contact_bodies,
            "max_contact_force": max_contact_force,
            "root_position": root_pos.astype(np.float64),
            "base_fixed": bool(self.robot.is_fixed_base and np.linalg.norm(root_pos) < 1.0e-5),
        }

    def _tcp_jacobian(
        self, body_pos: torch.Tensor, body_quat: torch.Tensor, tcp_pos: torch.Tensor
    ) -> torch.Tensor:
        del body_quat
        jacobian_body_index = self.tcp_body_id - 1 if self.robot.is_fixed_base else self.tcp_body_id
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        joint_indices = self.joint_ids if self.robot.is_fixed_base else [index + 6 for index in self.joint_ids]
        jacobian = all_jacobians[:, jacobian_body_index, :, joint_indices].clone()
        offset_world = tcp_pos - body_pos
        offset = offset_world[:, :, None].expand(-1, -1, jacobian.shape[-1])
        angular = jacobian[:, 3:6, :]
        jacobian[:, 0:3, :] += torch.cross(angular, offset, dim=1)
        return jacobian

    def validate_path(
        self,
        q_path: np.ndarray,
        settle_steps: int = 1,
        dense_substeps: int = 1,
        stop_on_collision: bool = True,
    ) -> dict[str, object]:
        path = np.asarray(q_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError("q_path must have shape (N, 6)")
        if dense_substeps < 1:
            raise ValueError("dense_substeps must be >= 1")
        if dense_substeps > 1:
            dense_points = []
            for segment in range(path.shape[0] - 1):
                alpha = np.arange(dense_substeps, dtype=np.float64) / dense_substeps
                dense_points.append(
                    path[segment][None, :] + alpha[:, None] * (path[segment + 1] - path[segment])[None, :]
                )
            dense_points.append(path[-1:])
            checked_path = np.vstack(dense_points)
        else:
            checked_path = path
        collision_indices = []
        max_contact_force = 0.0
        tcp_positions = []
        for index, q in enumerate(checked_path):
            self._set_state_and_step(q, settle_steps)
            _, _, tcp_pos, _ = self._read_tcp_pose()
            tcp_positions.append(tcp_pos[0].detach().cpu().numpy().astype(np.float64))
            collision, _, contact_force = self._read_contacts()
            max_contact_force = max(max_contact_force, contact_force)
            if collision:
                collision_indices.append(index)
                if stop_on_collision:
                    break
        tcp_positions_array = np.asarray(tcp_positions)
        tcp_path_length = float(np.sum(np.linalg.norm(np.diff(tcp_positions_array, axis=0), axis=1)))
        return {
            "trajectory_points": int(path.shape[0]),
            "dense_substeps_per_segment": int(dense_substeps),
            "planned_samples": int(checked_path.shape[0]),
            "samples": len(tcp_positions),
            "collision_free": not collision_indices,
            "collision_indices": collision_indices,
            "max_contact_force": max_contact_force,
            "tcp_path_length_m": tcp_path_length,
            "source": "PhysX contact reports via IsaacLab ContactSensor",
        }

    def playback_path(
        self,
        q_path: np.ndarray,
        render: bool = True,
        loops: int = 1,
        realtime_hz: float | None = None,
        ping_pong: bool = False,
    ) -> dict[str, object]:
        path = np.asarray(q_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError("q_path must have shape (N, 6)")
        if loops < 1:
            raise ValueError("loops must be >= 1")
        if realtime_hz is not None and realtime_hz <= 0.0:
            raise ValueError("realtime_hz must be positive")
        collision_indices = []
        max_contact_force = 0.0
        max_base_displacement = 0.0
        final_tcp_position = None
        final_tcp_quaternion = None
        global_index = 0
        frame_period = 1.0 / realtime_hz if realtime_hz is not None else None
        for loop_index in range(loops):
            loop_path = path[::-1] if ping_pong and loop_index % 2 else path
            for q in loop_path:
                frame_started = perf_counter()
                self._set_state_and_step(q, settle_steps=1, render=render)
                _, _, tcp_position, tcp_quaternion = self._read_tcp_pose()
                collision, _, contact_force = self._read_contacts()
                if collision:
                    collision_indices.append(global_index)
                max_contact_force = max(max_contact_force, contact_force)
                root_position = self.robot.data.root_pos_w[0].detach().cpu().numpy()
                max_base_displacement = max(max_base_displacement, float(np.linalg.norm(root_position)))
                final_tcp_position = tcp_position[0].detach().cpu().numpy().astype(np.float64)
                final_tcp_quaternion = tcp_quaternion[0].detach().cpu().numpy().astype(np.float64)
                if frame_period is not None:
                    sleep(max(0.0, frame_period - (perf_counter() - frame_started)))
                global_index += 1
        return {
            "samples": int(global_index),
            "loops": int(loops),
            "ping_pong": bool(ping_pong),
            "realtime_hz": realtime_hz,
            "collision_free": not collision_indices,
            "collision_indices": collision_indices,
            "max_contact_force": max_contact_force,
            "base_fixed": bool(self.robot.is_fixed_base and max_base_displacement < 1.0e-5),
            "max_base_displacement_m": max_base_displacement,
            "final_tcp_position": final_tcp_position,
            "final_tcp_quaternion_wxyz": final_tcp_quaternion,
        }

    def close(self) -> None:
        """Release Isaac Lab callbacks before SimulationApp shutdown."""
        if self.sim is not None:
            self._close_extension_entities()
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self.sim = None
