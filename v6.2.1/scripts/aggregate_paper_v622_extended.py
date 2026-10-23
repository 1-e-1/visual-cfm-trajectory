"""Aggregate extended v6.2.2 experiments into one machine-readable report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_v622"


def read(relative: str) -> dict:
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def metrics(relative: str) -> dict:
    report = read(relative)
    return {
        "source": relative,
        "tasks": report["tasks"],
        "grasp_successes": report["grasp"]["successes"],
        "place_successes": report["place"]["successes"],
        "joint_successes": report["two_stage"]["successes"],
        "joint_rate": report["two_stage"]["success_rate"],
        "joint_wilson_95": report["two_stage"]["success_rate_wilson_95"],
        "grasp_raw_error_mean_m": report["grasp"]["mean_raw_terminal_error_all"],
        "place_raw_error_mean_m": report["place"]["mean_raw_terminal_error_all"],
        "target_component_accuracy": report.get("target_component_accuracy"),
        "target_center_mean_error_m": report.get("target_center_mean_error_m"),
        "target_type_accuracy": report.get("target_type_accuracy"),
        "elapsed_s": report["elapsed_s"],
    }


def main() -> None:
    payload = {
        "protocol": "pose_visual_cfm_v622_paper_20260812",
        "model_baselines_k16": {
            "visual_cfm": metrics("inference_sweeps/candidates_k16_steps32.json"),
            "condition_cfm": metrics("baselines/condition_cfm_eval_k16.json"),
            "condition_cfm_capacity_matched": metrics(
                "baselines/condition_cfm_capacity_matched_eval_k16.json"
            ),
            "condition_ddim": metrics("baselines/condition_ddim_eval_k16.json"),
        },
        "loss_ablations_k8": {
            name: metrics(f"loss_ablations/{name}_eval_k8.json")
            for name in (
                "none",
                "no_terminal",
                "no_environment_collision",
                "no_self_collision",
                "no_dynamics",
                "no_auxiliary",
            )
        },
        "loss_ablations_fulltrain_k8": {
            name: metrics(f"loss_ablations_fulltrain/{name}_eval_k8.json")
            for name in (
                "no_terminal",
                "no_environment_collision",
                "no_self_collision",
                "no_dynamics",
                "no_auxiliary",
            )
        },
        "data_camera_ablations_k8": {
            "data100": metrics("loss_ablations/none_eval_k8.json"),
            "data50": metrics("data_camera_ablations/data50_eval_k8.json"),
            "data25": metrics("data_camera_ablations/data25_eval_k8.json"),
            "single_wrist": metrics(
                "data_camera_ablations/single_wrist_eval_k8.json"
            ),
        },
        "pointcloud_screen_k8": {
            "points2048": metrics("inference_sweeps/candidates_k8_steps32.json"),
            "points512": metrics("pointcloud_ablations/scene_points_512_k8.json"),
            "points128": metrics("pointcloud_ablations/scene_points_128_k8.json"),
        },
        "structural_ablations_k8": {
            name: metrics(f"structural_ablations/{name}_eval_k8.json")
            for name in (
                "skip_privileged",
                "skip_finetune",
                "from_scratch",
                "no_geometry",
                "no_film",
                "single_expert",
                "large_encoder",
            )
        },
        "not_executed_by_design": {
            "target_point_mask_ablation": (
                "not applicable: target_point_mask is stored but never consumed by the model"
            ),
            "high_resolution_pose_five_camera": (
                "no matched five-camera pose dataset above 128x128; interpolation is not a sensor-resolution experiment"
            ),
            "random_native_contact_physx_grid": (
                "invalid because the independent nominal rerun failed before finger closure"
            ),
        },
        "physx_reproducibility": {
            "historical_nominal_report": "outputs/native_contact/native_contact_return_place_validation.json",
            "historical_overall_success": True,
            "historical_contact_sensor_bilateral": True,
            "historical_event_callback_pad_counts": {"left": 0, "right": 0},
            "independent_nominal_rerun_success": False,
            "independent_rerun_failure": (
                "object displaced before finger closure; emergency-stop centering audit failed"
            ),
            "random_mass_size_friction_grid_valid": False,
        },
    }
    path = OUT / "extended_experiments_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
