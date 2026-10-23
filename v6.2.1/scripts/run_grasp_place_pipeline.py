"""Run the packaged Visual CFM pose grasp-place pipeline from one config."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _append_vector(command: list[str], name: str, values: list[float]) -> None:
    command.extend((name, *(str(float(value)) for value in values)))


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _run(command: list[str], dry_run: bool) -> None:
    print("[PIPELINE] " + subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/grasp_place_pipeline.yaml",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--regenerate_plan", action="store_true")
    parser.add_argument("--mode", choices=("lift", "full"), default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    _require(config_path, "pipeline config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planning = config["planning"]
    execution = config["execution"]

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "outputs" / "pipeline_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    scene_usd = _resolve(config["scene"]["usd"])
    frozen_plan = _resolve(planning["frozen_plan"])
    mode = args.mode or execution["default_mode"]
    headless = execution["headless"] if args.headless is None else args.headless
    record = execution["record"] if args.record is None else args.record
    _require(scene_usd, "scene USD")

    if args.regenerate_plan:
        plan_path = run_dir / "trajectory_plan.npz"
        plan_report = run_dir / "trajectory_plan_report.json"
        planner_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/plan_pose_grasp_place.py"),
            "--data_dir", str(_resolve(planning["data_dir"])),
            "--checkpoint", str(_resolve(planning["checkpoint"])),
            "--model_config", str(_resolve(planning["model_config"])),
            "--task_config", str(_resolve(planning["task_config"])),
            "--output", str(plan_path),
            "--report", str(plan_report),
            "--candidates", str(int(planning["candidates"])),
            "--max_rows", str(int(planning["max_rows"])),
            "--seed", str(int(config["seed"])),
            "--device", str(config["device"]),
        ]
        for key in ("data_dir", "checkpoint", "model_config", "task_config"):
            _require(_resolve(planning[key]), f"planning.{key}")
        _run(planner_command, args.dry_run)
    else:
        plan_path = frozen_plan
        plan_report = None
        _require(plan_path, "frozen trajectory plan")

    execution_report = run_dir / "physx_execution_report.json"
    executor_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/execute_pose_grasp_place_physx.py"),
        "--usd", str(scene_usd),
        "--plan", str(plan_path),
        "--output", str(execution_report),
        "--grasp_mode", str(execution["grasp_mode"]),
        "--cube_size", str(float(execution["cube_size_m"])),
        "--cube_mass", str(float(execution["cube_mass_kg"])),
        "--physics_dt", str(float(execution["physics_dt_s"])),
        "--motion_time_scale", str(float(execution["motion_time_scale"])),
    ]
    _append_vector(executor_command, "--pickup_center_offset", execution["pickup_center_offset_m"])
    _append_vector(executor_command, "--place_center_offset", execution["place_center_offset_m"])
    _append_vector(
        executor_command,
        "--grasp_point_offset_object",
        execution["grasp_point_offset_object_m"],
    )
    if mode == "lift":
        executor_command.append("--stop_after_lift")
    if headless:
        executor_command.append("--headless")
    if record:
        executor_command.extend((
            "--record_dir", str(run_dir / "media"),
            "--record_fps", str(float(execution["record_fps"])),
        ))
        if execution.get("camera_track_object", False):
            executor_command.append("--camera_track_object")
        _append_vector(executor_command, "--camera_eye", execution["camera_eye_w"])
        _append_vector(executor_command, "--camera_target", execution["camera_target_w"])
    _run(executor_command, args.dry_run)

    manifest = {
        "pipeline_version": int(config["version"]),
        "run_name": run_name,
        "config": str(config_path),
        "scene_usd": str(scene_usd),
        "plan": str(plan_path),
        "plan_regenerated": bool(args.regenerate_plan),
        "plan_report": None if plan_report is None else str(plan_report),
        "mode": mode,
        "headless": bool(headless),
        "record": bool(record),
        "execution_report": str(execution_report),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        result = json.loads(execution_report.read_text(encoding="utf-8"))
        manifest["result"] = {
            key: result.get(key)
            for key in ("grasp_success", "place_success", "overall_success", "lift_m", "place_error_m")
            if key in result
        }
        manifest["collision_audit"] = result.get("collision_audit")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PIPELINE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
