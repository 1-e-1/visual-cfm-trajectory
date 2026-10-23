"""Low-cost raw CFM screen for stage-wise and final checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "published_global_best": "outputs/checkpoints/visual_cfm_v622_pose.pt",
    "privileged_aux_best": "outputs/paper_v622/training/visual_cfm_v622_stagewise_privileged_aux.pt",
    "predicted_target_best": "outputs/paper_v622/training/visual_cfm_v622_stagewise_visual_distill.pt",
    "finetune_best": "outputs/paper_v622/training/visual_cfm_v622_stagewise_visual_finetune.pt",
    "finetune_final": "outputs/paper_v622/training/visual_cfm_v622_stagewise_final.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--only", default="")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    unknown = selected - set(CHECKPOINTS)
    if unknown:
        raise ValueError(f"unknown checkpoints: {sorted(unknown)}")
    checkpoints = {
        name: path
        for name, path in CHECKPOINTS.items()
        if not selected or name in selected
    }
    mode = "full_screen" if args.postprocess else "raw_screen"
    output_root = ROOT / "outputs/paper_v622/training" / mode
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, checkpoint_name in checkpoints.items():
        checkpoint = ROOT / checkpoint_name
        output = output_root / f"{name}.json"
        log = output.with_suffix(".log")
        command = [
            str(ROOT.parent.parent / "conda_envs/env_isaaclab/python.exe"),
            str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
            "--data_dir", str(ROOT / "data/visual_v622_pose/left"),
            "--checkpoint", str(checkpoint),
            "--config", str(ROOT / "configs/visual_cfm_v622_pose.yaml"),
            "--output", str(output),
            "--candidates", str(args.candidates),
            "--ode_steps", str(args.ode_steps),
            "--max_tasks", "49",
            "--seed", str(args.seed),
            "--independent_stage_generators",
        ]
        # ROOT is on D:, so construct the fixed environment path explicitly.
        command[0] = __import__('sys').executable
        if args.postprocess:
            command.extend(
                [
                    "--postprocess",
                    "--postprocess_mode", "reference_spline",
                    "--path_residual_weight", "0.15",
                    "--timing_points", "256",
                    "--unclipped_duration",
                    "--nominal_max_duration_s", "5.0",
                    "--max_output_duration_s", "30.0",
                    "--exclude_carried_target_from_cloud",
                    "--carried_target_exclusion_padding_m", "0.015",
                ]
            )
        if output.exists() and not args.force:
            status, elapsed = "reused", 0.0
        else:
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = time.perf_counter() - started
            status = "completed" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                raise RuntimeError(f"{name} failed; see {log}")
        report = json.loads(output.read_text(encoding="utf-8"))
        runs.append(
            {
                "name": name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "status": status,
                "elapsed_s": elapsed,
                "grasp_mean_raw_terminal_error_m": report["grasp"][
                    "mean_raw_terminal_error_all"
                ],
                "grasp_mean_raw_orientation_error_rad": report["grasp"][
                    "mean_raw_orientation_error_all_rad"
                ],
                "place_mean_raw_terminal_error_m": report["place"][
                    "mean_raw_terminal_error_all"
                ],
                "place_mean_raw_orientation_error_rad": report["place"][
                    "mean_raw_orientation_error_all_rad"
                ],
                "target_component_accuracy": report["target_component_accuracy"],
                "target_center_mean_error_m": report["target_center_mean_error_m"],
                "target_type_accuracy": report["target_type_accuracy"],
            }
        )
        print(f"[{status}] {name} elapsed={elapsed:.1f}s", flush=True)
    payload = {
        "screen_only": True,
        "postprocess": args.postprocess,
        "candidates": args.candidates,
        "ode_steps": args.ode_steps,
        "runs": runs,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
