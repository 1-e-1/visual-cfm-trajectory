"""Record the 20 NASA astronaut auxiliary-arm native-contact tasks.

Each task runs in a fresh Isaac Sim process.  Successful outputs are reused on
resume, and a compact batch summary is updated after every task.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/nasa_astronaut_aux_arm_20tasks"
MANIFEST = OUTPUT_ROOT / "task_manifest.json"
EXECUTOR = PROJECT_ROOT / "scripts/execute_pose_grasp_place_native_contact.py"


def _successful(report_path: Path, video_path: Path, gif_path: Path) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in (report_path, video_path, gif_path)):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(report.get("overall_success"))


def _summary_entry(name: str, report_path: Path, elapsed_s: float, reused: bool) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sensor = report["native_contact_audit"]["contact_force_sensor"]
    return {
        "name": name,
        "reused": reused,
        "elapsed_s": elapsed_s,
        "overall_success": bool(report["overall_success"]),
        "grasp_success": bool(report["grasp_success"]),
        "place_success": bool(report["place_success"]),
        "lift_mm": 1000.0 * float(report["lift_m"]),
        "place_error_mm": 1000.0 * float(report["place_error_m"]),
        "bilateral_contact": bool(sensor["bilateral_contact_observed"]),
        "peak_force_n": sensor["peak_force_n"],
        "collision_free": bool(report["collision_audit"]["collision_free"]),
        "recorded_frames": int(report["recording"]["frames"]),
        "video": report["recording"]["video"],
        "gif": report["recording"]["gif"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1, help="One-based first task index.")
    parser.add_argument("--end", type=int, default=20, help="Inclusive one-based last task index.")
    parser.add_argument("--force", action="store_true", help="Overwrite successful recordings.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not (1 <= args.start <= args.end <= len(tasks)):
        raise ValueError(f"invalid task range {args.start}..{args.end} for {len(tasks)} tasks")

    scene = Path(manifest["scene_usd"])
    camera = manifest["camera"]
    reports_dir = OUTPUT_ROOT / "reports"
    media_dir = OUTPUT_ROOT / "media"
    logs_dir = OUTPUT_ROOT / "logs"
    for directory in (reports_dir, media_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_ROOT / "recording_summary.json"
    existing: dict[str, dict[str, object]] = {}
    if summary_path.is_file():
        for item in json.loads(summary_path.read_text(encoding="utf-8")).get("tasks", []):
            existing[item["name"]] = item

    for index, task in enumerate(tasks, start=1):
        if index < args.start or index > args.end:
            continue
        name = task["name"]
        report_path = reports_dir / f"{name}.json"
        task_media = media_dir / name
        video_path = task_media / "pose_grasp_place_final.mp4"
        gif_path = task_media / "pose_grasp_place_final.gif"
        reused = not args.force and _successful(report_path, video_path, gif_path)
        if reused:
            entry = _summary_entry(name, report_path, 0.0, True)
            print(f"[{index:02d}/{len(tasks):02d}] reuse {name}", flush=True)
        else:
            task_media.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(EXECUTOR),
                "--usd", str(scene),
                "--plan", task["output_plan"],
                "--output", str(report_path),
                "--record_dir", str(task_media),
                "--cube_size", "0.03",
                "--cube_mass", "0.03",
                "--cube_angular_damping", "8.0",
                "--close_duration", "6.0",
                "--contact_hold_duration", "3.0",
                "--place_hold_duration", "6.0",
                "--release_duration", "6.0",
                "--settle_duration", "2.0",
                "--approach_stop_fraction", "1.0",
                "--preload_gap", "0.003",
                "--grasp_finger_target", "0.5395",
                "--skip_gap_servo",
                "--gripper_effort_limit", "40",
                "--gripper_velocity_limit", "0.15",
                "--gripper_stiffness", "80",
                "--gripper_damping", "20",
                "--motion_time_scale", "3.0",
                "--lift_time_scale", "5.0",
                "--maximum_cube_linear_speed", "2.0",
                "--maximum_cube_angular_speed", "30.0",
                "--maximum_target_contact_impulse", "0.2",
                "--place_center_offset", "-0.0018", "-0.0022", "0.0009",
                "--record_fps", str(camera["fps"]),
                "--record_width", str(camera["resolution"][0]),
                "--record_height", str(camera["resolution"][1]),
                "--camera_eye", *[str(value) for value in camera["eye_w"]],
                "--camera_target", *[str(value) for value in camera["target_w"]],
                "--headless",
            ]
            log_path = logs_dir / f"{name}.log"
            print(f"[{index:02d}/{len(tasks):02d}] record {name}", flush=True)
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            elapsed_s = time.perf_counter() - started
            if result.returncode != 0:
                raise RuntimeError(f"task {name} failed with code {result.returncode}; see {log_path}")
            if not _successful(report_path, video_path, gif_path):
                raise RuntimeError(f"task {name} did not pass validation; see {report_path}")
            entry = _summary_entry(name, report_path, elapsed_s, False)

        existing[name] = entry
        ordered = [existing[item["name"]] for item in tasks if item["name"] in existing]
        payload = {
            "scene_usd": str(scene),
            "camera": camera,
            "completed": len(ordered),
            "successful": sum(bool(item["overall_success"]) for item in ordered),
            "tasks": ordered,
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{index:02d}/{len(tasks):02d}] success {name}: "
            f"lift={entry['lift_mm']:.2f} mm, error={entry['place_error_mm']:.2f} mm",
            flush=True,
        )

    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
