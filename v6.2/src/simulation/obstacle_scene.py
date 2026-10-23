"""Obstacle-aware Isaac Lab scene with wrist RGB-D and semantic sensing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.simulation.kinematic_scene import FixedBaseKinematicScene


class ObstacleAwareKinematicScene(FixedBaseKinematicScene):
    """Fixed-base robot scene extended with deterministic moving obstacles."""

    def __init__(
        self,
        usd_path: str | Path,
        joint_names: list[str],
        obstacle_cfg: dict[str, Any],
        tcp_body_name: str = "foot_1",
        tcp_prim_name: str = "TCP",
        device: str = "cuda:0",
    ) -> None:
        self.obstacle_cfg = obstacle_cfg
        self.all_obstacles: list[Any] = []
        self.static_obstacles: list[Any] = []
        self.dynamic_obstacles: list[Any] = []
        self.debris_obstacles: list[Any] = []
        self.obstacle_sensors: list[Any] = []
        self.obstacle_specs: list[dict[str, Any]] = []
        self.robot_rigid_body_paths: set[str] = set()
        self.robot_collision_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.wrist_camera: Any | None = None
        self.fixed_cameras: dict[str, Any] = {}
        self.obstacle_time_s = 0.0
        scene_cfg = obstacle_cfg["scene"]
        super().__init__(
            usd_path=usd_path,
            joint_names=joint_names,
            tcp_body_name=tcp_body_name,
            tcp_prim_name=tcp_prim_name,
            device=device,
            physics_dt=float(scene_cfg["physics_dt_s"]),
            contact_threshold=float(scene_cfg["contact_threshold_n"]),
            pre_reset_setup=self._setup_extension_scene,
        )
        self.set_obstacle_time(0.0)
        self._step_scene(render=True)

    def _setup_extension_scene(self, _: FixedBaseKinematicScene) -> None:
        import isaaclab.sim as sim_utils
        import omni.usd
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from pxr import UsdPhysics

        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)).func(
            "/World/ObstacleSceneLight", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
        )

        variants = self._shape_pool_variants()
        parking_position = np.asarray(
            self.obstacle_cfg.get("shape_pool", {}).get("parking_position_m", [8.0, 8.0, 8.0]),
            dtype=np.float64,
        )
        for index, variant in enumerate(variants):
            role = str(variant["role"])
            prim_path = f"/World/Obstacles/{role.capitalize()}_{index}"
            obstacle = self._make_shape(
                RigidObject,
                RigidObjectCfg,
                sim_utils,
                prim_path=prim_path,
                variant=variant,
                position=parking_position + np.asarray([0.0, 0.0, 0.25 * index]),
            )
            self.all_obstacles.append(obstacle)
            if role == "static":
                self.static_obstacles.append(obstacle)
            elif role == "dynamic":
                self.dynamic_obstacles.append(obstacle)
            elif role == "debris":
                self.debris_obstacles.append(obstacle)
            else:
                raise ValueError(f"Unsupported obstacle role: {role}")
            self.obstacle_specs.append(
                {
                    "name": str(variant["id"]),
                    "pool_id": str(variant["id"]),
                    "type": role,
                    "shape": str(variant["shape"]),
                    "size_m": np.asarray(variant["size_m"], dtype=np.float64),
                    "active": False,
                    "position_m": parking_position.copy(),
                    "center_m": parking_position.copy(),
                    "axis_xyz": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                    "amplitude_m": 0.0,
                    "frequency_hz": 0.0,
                    "corridor_half_length_m": 0.0,
                    "speed_m_s": 0.0,
                    "phase_rad": 0.0,
                    "motion_model": "static" if role == "static" else "sinusoidal",
                    "quaternion_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "parking_position_m": parking_position + np.asarray([0.0, 0.0, 0.25 * index]),
                    "prim_path": prim_path,
                }
            )

        self._configure_defaults()

        stage = omni.usd.get_context().get_stage()
        robot_body_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith("/World/Robot") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not robot_body_paths:
            raise RuntimeError("No robot rigid bodies found for filtered obstacle contact sensors")
        self.robot_rigid_body_paths = set(robot_body_paths)
        self.robot_collision_boxes = self._extract_robot_collision_boxes(stage, robot_body_paths)
        self.obstacle_sensors = [
            ContactSensor(
                ContactSensorCfg(
                    prim_path=spec["prim_path"],
                    update_period=0.0,
                    history_length=1,
                    force_threshold=self.contact_threshold,
                    filter_prim_paths_expr=robot_body_paths,
                    track_contact_points=False,
                    max_contact_data_count_per_prim=128,
                )
            )
            for spec in self.obstacle_specs
        ]

        camera_cfg = self.obstacle_cfg["camera"]
        pinhole_cfg = camera_cfg["pinhole"]
        offset_cfg = camera_cfg["offset"]
        self.wrist_camera = Camera(
            CameraCfg(
                prim_path=camera_cfg["prim_path"],
                update_period=float(camera_cfg["update_period_s"]),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(value) for value in offset_cfg["position_m"]),
                    rot=tuple(float(value) for value in offset_cfg["quaternion_wxyz"]),
                    convention=offset_cfg["convention"],
                ),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(pinhole_cfg["focal_length_mm"]),
                    focus_distance=float(pinhole_cfg["focus_distance_mm"]),
                    horizontal_aperture=float(pinhole_cfg["horizontal_aperture_mm"]),
                    clipping_range=tuple(float(value) for value in pinhole_cfg["clipping_range_m"]),
                ),
                data_types=list(camera_cfg["data_types"]),
                width=int(camera_cfg["width"]),
                height=int(camera_cfg["height"]),
                depth_clipping_behavior=camera_cfg["depth_clipping_behavior"],
                semantic_filter=["class"],
                colorize_semantic_segmentation=False,
            )
        )
        fixed_cfg = self.obstacle_cfg.get("fixed_cameras", {})
        if fixed_cfg.get("enabled", False):
            for name, item in fixed_cfg.get("cameras", {}).items():
                position = np.asarray(item["position_m"], dtype=np.float64)
                look_at = np.asarray(item["look_at_m"], dtype=np.float64)
                self.fixed_cameras[str(name)] = Camera(
                    CameraCfg(
                        prim_path=str(item["prim_path"]),
                        update_period=float(camera_cfg["update_period_s"]),
                        offset=CameraCfg.OffsetCfg(
                            pos=tuple(float(value) for value in position),
                            rot=tuple(float(value) for value in self._look_at_quaternion_wxyz(position, look_at)),
                            convention="ros",
                        ),
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=float(pinhole_cfg["focal_length_mm"]),
                            focus_distance=float(pinhole_cfg["focus_distance_mm"]),
                            horizontal_aperture=float(pinhole_cfg["horizontal_aperture_mm"]),
                            clipping_range=tuple(float(value) for value in pinhole_cfg["clipping_range_m"]),
                        ),
                        data_types=list(camera_cfg["data_types"]),
                        width=int(camera_cfg["width"]),
                        height=int(camera_cfg["height"]),
                        depth_clipping_behavior=camera_cfg["depth_clipping_behavior"],
                        semantic_filter=["class"],
                        colorize_semantic_segmentation=False,
                    )
                )

    def _shape_pool_variants(self) -> list[dict[str, Any]]:
        configured = self.obstacle_cfg.get("shape_pool", {}).get("variants")
        if configured:
            return [dict(item) for item in configured]
        static_cfg = self.obstacle_cfg["static_obstacles"]["default"]
        dynamic_cfg = self.obstacle_cfg["dynamic_obstacles"]["default"]
        return [
            {
                "id": static_cfg["name"],
                "role": "static",
                "shape": "cuboid",
                "size_m": static_cfg["size_m"],
                "color_rgb": [0.12, 0.42, 0.76],
            },
            {
                "id": dynamic_cfg["name"],
                "role": "dynamic",
                "shape": "cuboid",
                "size_m": dynamic_cfg["size_m"],
                "color_rgb": [0.88, 0.26, 0.16],
            },
        ]

    def _configure_defaults(self) -> None:
        static_cfg = self.obstacle_cfg["static_obstacles"]["default"]
        dynamic_cfg = self.obstacle_cfg["dynamic_obstacles"]["default"]
        static_index = next(index for index, item in enumerate(self.obstacle_specs) if item["type"] == "static")
        dynamic_index = next(index for index, item in enumerate(self.obstacle_specs) if item["type"] == "dynamic")
        static_spec = self.obstacle_specs[static_index]
        static_spec.update(
            active=True,
            position_m=np.asarray(static_cfg["position_m"], dtype=np.float64),
            quaternion_wxyz=np.asarray(static_cfg["quaternion_wxyz"], dtype=np.float64),
        )
        dynamic_spec = self.obstacle_specs[dynamic_index]
        dynamic_spec.update(
            active=True,
            center_m=np.asarray(dynamic_cfg["center_m"], dtype=np.float64),
            axis_xyz=np.asarray(dynamic_cfg["axis_xyz"], dtype=np.float64),
            amplitude_m=float(dynamic_cfg["amplitude_m"]),
            frequency_hz=float(dynamic_cfg["frequency_hz"]),
            phase_rad=float(dynamic_cfg["phase_rad"]),
            motion_model="sinusoidal",
            quaternion_wxyz=np.asarray(dynamic_cfg["quaternion_wxyz"], dtype=np.float64),
        )

    @staticmethod
    def _make_shape(
        rigid_object_type: Any,
        rigid_object_cfg_type: Any,
        sim_utils: Any,
        prim_path: str,
        variant: dict[str, Any],
        position: list[float] | np.ndarray,
    ) -> Any:
        size = np.asarray(variant["size_m"], dtype=np.float64)
        shape = str(variant["shape"])
        color = tuple(float(value) for value in variant["color_rgb"])
        role = str(variant["role"])
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            "collision_props": sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.82),
            "semantic_tags": [("class", f"{role}_obstacle")],
            "activate_contact_sensors": True,
        }
        if shape == "cuboid":
            spawn_cfg = sim_utils.CuboidCfg(size=tuple(float(value) for value in size), **common)
        elif shape == "sphere":
            spawn_cfg = sim_utils.SphereCfg(radius=float(size.max() * 0.5), **common)
        elif shape == "cylinder":
            spawn_cfg = sim_utils.CylinderCfg(
                radius=float(max(size[0], size[1]) * 0.5),
                height=float(size[2]),
                axis="Z",
                **common,
            )
        elif shape == "capsule":
            spawn_cfg = sim_utils.CapsuleCfg(
                radius=float(max(size[0], size[1]) * 0.5),
                height=float(size[2]),
                axis="Z",
                **common,
            )
        elif shape == "asteroid":
            spawn_cfg = sim_utils.MeshSphereCfg(radius=float(size.max() * 0.5), **common)
        else:
            raise ValueError(f"Unsupported obstacle shape: {shape}")
        obstacle = rigid_object_type(
            rigid_object_cfg_type(
                prim_path=prim_path,
                spawn=spawn_cfg,
                init_state=rigid_object_cfg_type.InitialStateCfg(
                    pos=tuple(float(value) for value in position),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
        )
        if shape == "asteroid":
            ObstacleAwareKinematicScene._roughen_asteroid_mesh(
                prim_path,
                int(variant.get("roughness_seed", 0)),
            )
        return obstacle

    @staticmethod
    def _roughen_asteroid_mesh(prim_path: str, seed: int) -> None:
        """Deform a sphere mesh before physics starts while retaining a convex collider."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        if not mesh_prim.IsValid():
            raise RuntimeError(f"Asteroid mesh not found: {prim_path}/geometry/mesh")
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get() or []
        deformed = []
        for point in points:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            factor = 1.0 + 0.16 * np.sin(37.0 * x + 29.0 * y + 41.0 * z + float(seed))
            deformed.append(Gf.Vec3f(x * factor, y * factor, z * factor))
        mesh.GetPointsAttr().Set(deformed)

    @staticmethod
    def _look_at_quaternion_wxyz(position: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return a ROS optical-frame quaternion whose +Z axis looks at target."""
        forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, world_up))) > 0.98:
            world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1.0e-12)
        down = np.cross(forward, right)
        down /= max(float(np.linalg.norm(down)), 1.0e-12)
        rotation = np.column_stack((right, down, forward))
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        quaternion = np.asarray([w, x, y, z], dtype=np.float64)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
        return quaternion

    @staticmethod
    def _dynamic_state(spec: dict[str, Any], time_s: float) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(spec["center_m"], dtype=np.float64)
        axis = np.asarray(spec["axis_xyz"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        amplitude = float(spec["amplitude_m"])
        omega = 2.0 * np.pi * float(spec["frequency_hz"])
        phase = omega * float(time_s) + float(spec["phase_rad"])
        position = center + axis * amplitude * np.sin(phase)
        velocity = axis * amplitude * omega * np.cos(phase)
        return position, velocity

    @staticmethod
    def _ping_pong_state(spec: dict[str, Any], time_s: float) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(spec["center_m"], dtype=np.float64)
        axis = np.asarray(spec["axis_xyz"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        half_length = max(float(spec["corridor_half_length_m"]), 1.0e-6)
        speed = max(float(spec["speed_m_s"]), 1.0e-6)
        period = 4.0 * half_length / speed
        phase_fraction = float(spec.get("phase_rad", 0.0)) / (2.0 * np.pi)
        cycle = (float(time_s) / period + phase_fraction) % 1.0
        if cycle < 0.25:
            displacement = 4.0 * cycle * half_length
            velocity_sign = 1.0
        elif cycle < 0.75:
            displacement = (2.0 - 4.0 * cycle) * half_length
            velocity_sign = -1.0
        else:
            displacement = (4.0 * cycle - 4.0) * half_length
            velocity_sign = 1.0
        return center + axis * displacement, axis * speed * velocity_sign

    @classmethod
    def obstacle_motion_state(cls, spec: dict[str, Any], time_s: float) -> tuple[np.ndarray, np.ndarray]:
        motion_model = spec.get("motion_model", "static")
        if motion_model == "static":
            return np.asarray(spec["position_m"], dtype=np.float64), np.zeros(3, dtype=np.float64)
        if motion_model == "sinusoidal":
            return cls._dynamic_state(spec, time_s)
        if motion_model == "linear_ping_pong":
            return cls._ping_pong_state(spec, time_s)
        raise ValueError(f"Unsupported obstacle motion model: {motion_model}")

    def _write_object_state(
        self,
        obstacle: Any,
        position: np.ndarray,
        quaternion: np.ndarray,
        linear_velocity: np.ndarray | None = None,
    ) -> None:
        state = torch.zeros((1, 13), dtype=torch.float32, device=self.sim.device)
        state[0, :3] = torch.as_tensor(position, dtype=torch.float32, device=self.sim.device)
        state[0, 3:7] = torch.as_tensor(quaternion, dtype=torch.float32, device=self.sim.device)
        if linear_velocity is not None:
            state[0, 7:10] = torch.as_tensor(linear_velocity, dtype=torch.float32, device=self.sim.device)
        obstacle.write_root_state_to_sim(state)

    def set_obstacle_time(self, time_s: float) -> None:
        """Synchronize dynamic obstacle pose and velocity to task time."""
        self.obstacle_time_s = float(time_s)
        for obstacle, spec in zip(self.all_obstacles, self.obstacle_specs, strict=True):
            if spec["active"]:
                position, velocity = self.obstacle_motion_state(spec, self.obstacle_time_s)
            else:
                position = np.asarray(spec["parking_position_m"], dtype=np.float64)
                velocity = np.zeros(3, dtype=np.float64)
            self._write_object_state(
                obstacle,
                position,
                np.asarray(spec["quaternion_wxyz"], dtype=np.float64),
                velocity,
            )

    def configure_obstacles(
        self,
        static_position_m: np.ndarray,
        dynamic_center_m: np.ndarray,
        dynamic_axis_xyz: np.ndarray,
        dynamic_amplitude_m: float,
        dynamic_frequency_hz: float,
        dynamic_phase_rad: float = 0.0,
    ) -> None:
        """Apply the legacy two-cuboid task contract."""
        axis = np.asarray(dynamic_axis_xyz, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        for spec in self.obstacle_specs:
            spec["active"] = False
        static_spec = next(item for item in self.obstacle_specs if item["type"] == "static")
        dynamic_spec = next(item for item in self.obstacle_specs if item["type"] == "dynamic")
        static_spec.update(
            active=True,
            position_m=np.asarray(static_position_m, dtype=np.float64),
            motion_model="static",
        )
        dynamic_spec.update(
            active=True,
            center_m=np.asarray(dynamic_center_m, dtype=np.float64),
            axis_xyz=axis,
            amplitude_m=float(dynamic_amplitude_m),
            frequency_hz=float(dynamic_frequency_hz),
            phase_rad=float(dynamic_phase_rad),
            motion_model="sinusoidal",
        )
        self.set_obstacle_time(0.0)

    def configure_randomized_obstacles(self, instances: list[dict[str, Any]]) -> None:
        """Activate a task-specific subset from the pre-spawned collision-shape pool."""
        if len(instances) > int(
            self.obstacle_cfg["observation_contract"]["privileged_obstacle_token"]["max_obstacles"]
        ):
            raise ValueError("Active obstacle count exceeds the observation token contract")
        for spec in self.obstacle_specs:
            spec["active"] = False
        activated: set[str] = set()
        for instance in instances:
            pool_id = str(instance["pool_id"])
            if pool_id in activated:
                raise ValueError(f"Obstacle pool variant cannot be activated twice: {pool_id}")
            activated.add(pool_id)
            spec = next((item for item in self.obstacle_specs if item["pool_id"] == pool_id), None)
            if spec is None:
                raise KeyError(f"Unknown obstacle pool variant: {pool_id}")
            spec["active"] = True
            for key in (
                "position_m",
                "center_m",
                "axis_xyz",
                "quaternion_wxyz",
            ):
                if key in instance:
                    spec[key] = np.asarray(instance[key], dtype=np.float64)
            for key in (
                "amplitude_m",
                "frequency_hz",
                "corridor_half_length_m",
                "speed_m_s",
                "phase_rad",
            ):
                if key in instance:
                    spec[key] = float(instance[key])
            spec["motion_model"] = str(instance.get("motion_model", spec["motion_model"]))
        self.set_obstacle_time(0.0)

    def active_obstacle_specs(self) -> list[dict[str, Any]]:
        """Return detached active specs for planners and dataset serialization."""
        output = []
        for spec in self.obstacle_specs:
            if not spec["active"]:
                continue
            output.append(
                {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in spec.items()
                    if key not in {"parking_position_m", "prim_path"}
                }
            )
        return output

    def set_dynamic_obstacle_pose(self, position: np.ndarray) -> None:
        """Override the dynamic obstacle pose for contact-sensor validation."""
        index = next(
            index
            for index, item in enumerate(self.obstacle_specs)
            if item["type"] == "dynamic" and item["active"]
        )
        spec = self.obstacle_specs[index]
        self._write_object_state(
            self.all_obstacles[index],
            np.asarray(position, dtype=np.float64),
            np.asarray(spec["quaternion_wxyz"], dtype=np.float64),
        )

    def _reset_extension_sensors(self) -> None:
        for sensor in self.obstacle_sensors:
            sensor.reset()
        if self.wrist_camera is not None:
            self.wrist_camera.reset()
        for camera in self.fixed_cameras.values():
            camera.reset()

    def _update_extension_entities(self) -> None:
        for obstacle in self.all_obstacles:
            obstacle.update(self.sim_dt)
        for sensor in self.obstacle_sensors:
            sensor.update(self.sim_dt)
        if self.wrist_camera is not None:
            self.wrist_camera.update(self.sim_dt)
        for camera in self.fixed_cameras.values():
            camera.update(self.sim_dt)

    def _step_scene(self, render: bool) -> None:
        self.robot.write_data_to_sim()
        self.sim.step(render=render)
        self.robot.update(self.sim_dt)
        self.contact_sensor.update(self.sim_dt)
        self._update_extension_entities()

    def read_obstacle_contacts(self) -> dict[str, Any]:
        """Return robot-obstacle contacts from per-obstacle filtered sensors."""
        collided: list[str] = []
        max_force = 0.0
        per_obstacle_force: dict[str, float] = {}
        overlap_result = self.read_collision_mesh_obb_overlaps()
        for spec, sensor in zip(self.obstacle_specs, self.obstacle_sensors, strict=True):
            if not spec["active"]:
                continue
            forces = sensor.data.force_matrix_w
            force = float(torch.linalg.norm(forces, dim=-1).max().item()) if forces is not None else 0.0
            per_obstacle_force[spec["name"]] = force
            max_force = max(max_force, force)
            if force > self.contact_threshold or overlap_result["per_obstacle"][spec["name"]]:
                collided.append(spec["name"])
        return {
            "collision": bool(collided),
            "obstacles": collided,
            "max_contact_force": max_force,
            "per_obstacle_force": per_obstacle_force,
            "overlap_robot_bodies": overlap_result["per_obstacle"],
            "overlap_diagnostics": overlap_result["diagnostics"],
            "detection_sources": [
                "PhysX ContactSensor",
                "USD collision-mesh OBB SAT broad phase",
            ],
        }

    @staticmethod
    def _extract_robot_collision_boxes(
        stage: Any,
        robot_body_paths: list[str],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        cache = UsdGeom.XformCache()
        boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_path in robot_body_paths:
            body_prim = stage.GetPrimAtPath(body_path)
            local_points: list[np.ndarray] = []
            for prim in Usd.PrimRange(body_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):
                    continue
                transform_result = cache.ComputeRelativeTransform(prim, body_prim)
                transform = transform_result[0] if isinstance(transform_result, tuple) else transform_result
                for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
                    transformed = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    local_points.append(np.asarray(transformed, dtype=np.float64))
            if local_points:
                points = np.stack(local_points)
                lower = points.min(axis=0)
                upper = points.max(axis=0)
                boxes[str(body_prim.GetName())] = ((lower + upper) * 0.5, (upper - lower) * 0.5)
        if not boxes:
            raise RuntimeError("Could not extract collision-mesh bounds from robot USD")
        return boxes

    @staticmethod
    def _quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
        w, x, y, z = quaternion
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _obb_overlap(
        center_a: np.ndarray,
        axes_a: np.ndarray,
        half_a: np.ndarray,
        center_b: np.ndarray,
        axes_b: np.ndarray,
        half_b: np.ndarray,
    ) -> bool:
        rotation = axes_a.T @ axes_b
        abs_rotation = np.abs(rotation) + 1.0e-9
        translation = axes_a.T @ (center_b - center_a)
        for index in range(3):
            radius_a = half_a[index]
            radius_b = float(half_b @ abs_rotation[index])
            if abs(translation[index]) > radius_a + radius_b:
                return False
        for index in range(3):
            radius_a = float(half_a @ abs_rotation[:, index])
            radius_b = half_b[index]
            if abs(float(translation @ rotation[:, index])) > radius_a + radius_b:
                return False
        for index_a in range(3):
            for index_b in range(3):
                radius_a = (
                    half_a[(index_a + 1) % 3] * abs_rotation[(index_a + 2) % 3, index_b]
                    + half_a[(index_a + 2) % 3] * abs_rotation[(index_a + 1) % 3, index_b]
                )
                radius_b = (
                    half_b[(index_b + 1) % 3] * abs_rotation[index_a, (index_b + 2) % 3]
                    + half_b[(index_b + 2) % 3] * abs_rotation[index_a, (index_b + 1) % 3]
                )
                separation = abs(
                    translation[(index_a + 2) % 3] * rotation[(index_a + 1) % 3, index_b]
                    - translation[(index_a + 1) % 3] * rotation[(index_a + 2) % 3, index_b]
                )
                if separation > radius_a + radius_b:
                    return False
        return True

    def read_collision_mesh_obb_overlaps(self, clearance_m: float = 0.0) -> dict[str, Any]:
        """Conservatively test obstacle OBBs against link collision-mesh OBBs."""
        per_obstacle: dict[str, list[str]] = {}
        diagnostics: dict[str, Any] = {}
        for spec, obstacle in zip(self.obstacle_specs, self.all_obstacles, strict=True):
            if not spec["active"]:
                continue
            obstacle_center = obstacle.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            obstacle_quaternion = obstacle.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            obstacle_axes = self._quaternion_to_matrix(obstacle_quaternion)
            obstacle_half = 0.5 * np.asarray(spec["size_m"], dtype=np.float64) + float(clearance_m)
            hits: list[str] = []
            for body_name, (local_center, local_half) in self.robot_collision_boxes.items():
                body_ids, _ = self.robot.find_bodies([f"^{body_name}$"])
                if len(body_ids) != 1:
                    continue
                body_id = body_ids[0]
                body_position = self.robot.data.body_pos_w[0, body_id].detach().cpu().numpy().astype(np.float64)
                body_quaternion = self.robot.data.body_quat_w[0, body_id].detach().cpu().numpy().astype(np.float64)
                body_axes = self._quaternion_to_matrix(body_quaternion)
                body_center = body_position + body_axes @ local_center
                overlaps = self._obb_overlap(
                    body_center,
                    body_axes,
                    local_half,
                    obstacle_center,
                    obstacle_axes,
                    obstacle_half,
                )
                if body_name == "foot_1":
                    diagnostics[spec["name"]] = {
                        "body_center": body_center.tolist(),
                        "obstacle_center": obstacle_center.tolist(),
                        "obstacle_quaternion": obstacle_quaternion.tolist(),
                        "overlaps": overlaps,
                    }
                if overlaps:
                    hits.append(f"/World/Robot/{body_name}")
            per_obstacle[spec["name"]] = sorted(hits)
        return {
            "collision": any(per_obstacle.values()),
            "per_obstacle": per_obstacle,
            "clearance_m": float(clearance_m),
            "diagnostics": diagnostics,
        }

    def capture_observation(
        self,
        q: np.ndarray,
        time_s: float = 0.0,
        render_steps: int = 3,
    ) -> dict[str, Any]:
        """Capture one synchronized robot state, obstacle state, and wrist image."""
        self.set_obstacle_time(time_s)
        self._set_state_and_step(q, settle_steps=max(1, render_steps), render=True)
        camera_data = self._read_camera_observation(self.wrist_camera)
        _, _, tcp_position, tcp_quaternion = self._read_tcp_pose()
        return {
            "q": np.asarray(q, dtype=np.float64),
            "time_s": float(time_s),
            "tcp_position": tcp_position[0].detach().cpu().numpy().astype(np.float64),
            "tcp_quaternion_wxyz": tcp_quaternion[0].detach().cpu().numpy().astype(np.float64),
            **camera_data,
            "contacts": self.read_obstacle_contacts(),
        }

    def _read_camera_observation(self, camera: Any) -> dict[str, Any]:
        output = camera.data.output
        rgb = output["rgb"][0, ..., :3].detach().cpu().numpy()
        depth = output["distance_to_image_plane"][0].detach().cpu().numpy()
        semantic = output["semantic_segmentation"][0].detach().cpu().numpy()
        if depth.ndim == 2:
            depth = depth[..., None]
        if semantic.ndim == 2:
            semantic = semantic[..., None]
        return {
            "rgb": rgb,
            "depth": depth,
            "semantic": semantic,
            "camera_info": camera.data.info,
            "camera_position_w": camera.data.pos_w[0].detach().cpu().numpy(),
            "camera_quaternion_w_ros": camera.data.quat_w_ros[0].detach().cpu().numpy(),
            "camera_quaternion_w_world": camera.data.quat_w_world[0].detach().cpu().numpy(),
        }

    def camera_view_names(self, include_wrist: bool = True) -> list[str]:
        configured = self.obstacle_cfg.get("fixed_cameras", {}).get("view_order", [])
        names = [str(name) for name in configured if str(name) == "wrist" or str(name) in self.fixed_cameras]
        if not names:
            names = ["wrist"] + sorted(self.fixed_cameras)
        if not include_wrist:
            names = [name for name in names if name != "wrist"]
        return names

    def capture_multiview_observation(
        self,
        q: np.ndarray,
        time_s: float = 0.0,
        render_steps: int = 3,
    ) -> dict[str, Any]:
        self.set_obstacle_time(time_s)
        self._set_state_and_step(q, settle_steps=max(1, render_steps), render=True)
        views: dict[str, dict[str, Any]] = {}
        for name in self.camera_view_names(include_wrist=True):
            camera = self.wrist_camera if name == "wrist" else self.fixed_cameras[name]
            views[name] = self._read_camera_observation(camera)
        _, _, tcp_position, tcp_quaternion = self._read_tcp_pose()
        return {
            "q": np.asarray(q, dtype=np.float64),
            "time_s": float(time_s),
            "tcp_position": tcp_position[0].detach().cpu().numpy().astype(np.float64),
            "tcp_quaternion_wxyz": tcp_quaternion[0].detach().cpu().numpy().astype(np.float64),
            "views": views,
            "contacts": self.read_obstacle_contacts(),
        }

    def validate_timed_path(
        self,
        q_path: np.ndarray,
        time_s: np.ndarray,
        settle_steps: int = 1,
        stop_on_collision: bool = True,
    ) -> dict[str, Any]:
        """Validate a time-parameterized path against synchronized moving obstacles."""
        path = np.asarray(q_path, dtype=np.float64)
        times = np.asarray(time_s, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError("q_path must have shape (N, 6)")
        if times.shape != (path.shape[0],) or np.any(np.diff(times) < 0.0):
            raise ValueError("time_s must be non-decreasing with shape (N,)")
        collision_indices: list[int] = []
        self_collision_indices: list[int] = []
        max_obstacle_force = 0.0
        max_any_force = 0.0
        tcp_positions: list[np.ndarray] = []
        for index, (q, task_time) in enumerate(zip(path, times, strict=True)):
            self.set_obstacle_time(float(task_time))
            self._set_state_and_step(q, settle_steps=settle_steps)
            _, _, tcp_position, _ = self._read_tcp_pose()
            tcp_positions.append(tcp_position[0].detach().cpu().numpy().astype(np.float64))
            obstacle_contact = self.read_obstacle_contacts()
            any_contact, _, any_force = self._read_contacts()
            max_obstacle_force = max(max_obstacle_force, float(obstacle_contact["max_contact_force"]))
            max_any_force = max(max_any_force, any_force)
            if obstacle_contact["collision"]:
                collision_indices.append(index)
            if any_contact and not obstacle_contact["collision"]:
                self_collision_indices.append(index)
            if stop_on_collision and (collision_indices or self_collision_indices):
                break
        tcp_array = np.asarray(tcp_positions)
        tcp_length = float(np.linalg.norm(np.diff(tcp_array, axis=0), axis=1).sum()) if len(tcp_array) > 1 else 0.0
        return {
            "trajectory_points": int(path.shape[0]),
            "samples": len(tcp_positions),
            "collision_free": not collision_indices and not self_collision_indices,
            "obstacle_collision_indices": collision_indices,
            "self_collision_indices": self_collision_indices,
            "max_obstacle_contact_force": max_obstacle_force,
            "max_any_robot_contact_force": max_any_force,
            "tcp_path_length_m": tcp_length,
            "source": "PhysX filtered ContactSensor with deterministic obstacle-time synchronization",
        }

    def obstacle_state_tokens(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Encode the privileged obstacle contract as fixed-size tokens and mask."""
        contract = self.obstacle_cfg["observation_contract"]["privileged_obstacle_token"]
        tokens = np.zeros(
            (int(contract["max_obstacles"]), int(contract["per_obstacle_dim"])),
            dtype=np.float32,
        )
        mask = np.zeros((int(contract["max_obstacles"]),), dtype=np.bool_)
        active_specs = [item for item in self.obstacle_specs if item["active"]]
        for index, spec in enumerate(active_specs[: len(tokens)]):
            position, velocity = self.obstacle_motion_state(spec, time_s)
            role_code = {"static": 0.0, "dynamic": 1.0, "debris": 2.0}[spec["type"]]
            tokens[index] = np.concatenate(
                [
                    [role_code],
                    np.asarray(spec["size_m"], dtype=np.float64),
                    position,
                    np.asarray(spec["quaternion_wxyz"], dtype=np.float64),
                    velocity,
                ]
            )
            mask[index] = True
        return tokens, mask

    def _close_extension_entities(self) -> None:
        self.wrist_camera = None
        self.fixed_cameras.clear()
        self.obstacle_sensors.clear()
        self.all_obstacles.clear()
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        self.debris_obstacles.clear()
