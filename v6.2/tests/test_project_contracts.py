from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.robot.torch_kinematics import DifferentiableZBotFK
from src.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_assets_and_fixed_base_introspection_contract() -> None:
    for relative_path in [
        "zbot_6s_new.usd",
        "assets/zbot_6s_new.usd",
        "assets/zbot_6s_new_backup.usd",
        "assets/zbot_6s_new_configured.usd",
    ]:
        assert (ROOT / relative_path).is_file()
    report = json.loads((ROOT / "outputs/logs/robot_introspection_configured.json").read_text(encoding="utf-8"))
    assert len(report["controllable_revolute_joints"]) == 6
    assert report["selected_tcp"]["path"].endswith("/TCP")
    runtime_report = json.loads((ROOT / "outputs/logs/single_task_report.json").read_text(encoding="utf-8"))
    assert runtime_report["task"]["start"]["base_fixed"]
    assert runtime_report["task"]["goal"]["base_fixed"]


def test_unbounded_joint_contract_and_sampling_domain() -> None:
    config = load_yaml(ROOT / "configs/robot.yaml")
    assert config["joint_space"]["physical_position_limits"] is None
    lower = np.asarray(config["joint_space"]["sampling_lower"])
    upper = np.asarray(config["joint_space"]["sampling_upper"])
    np.testing.assert_allclose(lower, -np.pi)
    np.testing.assert_allclose(upper, np.pi)


def test_debug_dataset_schema_endpoints_and_condition() -> None:
    shard = np.load(ROOT / "data/debug/train_shard_00000.npz", allow_pickle=False)
    required = load_yaml(ROOT / "configs/dataset.yaml")["required_fields"]
    assert set(required).issubset(shard.files)
    assert shard["q"].shape[1:] == (64, 6)
    assert shard["condition"].shape[1:] == (13,)
    np.testing.assert_allclose(shard["q"][:, 0], shard["q_start"], atol=1.0e-6)
    np.testing.assert_allclose(shard["q"][:, -1], shard["q_goal"], atol=1.0e-6)
    np.testing.assert_allclose(shard["condition"][:, :6], shard["q_start"], atol=1.0e-6)
    np.testing.assert_allclose(shard["condition"][:, 6:13], shard["goal_tcp_pose"], atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(shard["goal_tcp_pose"][:, 3:7], axis=1), 1.0, atol=1.0e-6)
    assert np.all(shard["collision_free"])


def test_differentiable_fk_changes_with_joint_state() -> None:
    fk = DifferentiableZBotFK(ROOT / "configs/kinematic_chain.json")
    q = torch.zeros(2, 6)
    q[1, 1] = 0.4
    position, quaternion = fk(q)
    assert not torch.allclose(position[0], position[1])
    assert not torch.allclose(quaternion[0], quaternion[1])


def test_saved_fk_validation_is_within_tolerance() -> None:
    report = json.loads((ROOT / "outputs/logs/torch_fk_validation.json").read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["max_position_error_m"] <= report["position_tolerance_m"]
    assert report["max_orientation_error_rad"] <= report["orientation_tolerance_rad"]


def test_saved_physx_self_collision_validation_contract() -> None:
    report = json.loads((ROOT / "outputs/logs/self_collision_validation.json").read_text(encoding="utf-8"))
    assert report["base_fixed_by_runtime_fix_root_link"]
    assert report["random_pose_collision_count"] >= 1
    zero_pose = report["manual_collision_candidate_results"][0]
    assert np.allclose(zero_pose["q"], 0.0)
    assert not zero_pose["collision"]
    configuration = json.loads((ROOT / "outputs/logs/usd_configuration_report.json").read_text(encoding="utf-8"))
    assert configuration["self_collision_verification"][0]["enabledSelfCollisions"]
    assert configuration["filtered_pair_count"] == 11
