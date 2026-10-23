"""Measure synchronized inference-stage runtime for K=1/8/16/64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--candidate_counts", default="1,8,16,64")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max_tasks", type=int, default=49)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    output_root = ROOT / config["output_root"] / "runtime_breakdown"
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for count in [int(value) for value in args.candidate_counts.split(",")]:
        output = output_root / f"k{count}.json"
        log = output.with_suffix(".log")
        command = [
            str(config["python"]),
            str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
            "--data_dir", str(ROOT / config["dataset"]),
            "--checkpoint", str(ROOT / config["checkpoint"]),
            "--config", str(ROOT / config["model_config"]),
            "--output", str(output),
            "--candidates", str(count),
            "--ode_steps", str(evaluation["ode_steps"]),
            "--max_tasks", str(args.max_tasks),
            "--seed", str(args.seed),
            "--position_tolerance_m", str(evaluation["position_tolerance_m"]),
            "--orientation_tolerance_rad", str(evaluation["orientation_tolerance_rad"]),
            "--required_clearance_m", str(evaluation["required_clearance_m"]),
            "--self_capsule_radius_m", str(evaluation["self_capsule_radius_m"]),
            "--independent_stage_generators",
            "--exclude_carried_target_from_cloud",
            "--carried_target_exclusion_padding_m",
            str(evaluation["carried_target_exclusion_padding_m"]),
            "--postprocess",
            "--postprocess_mode", str(evaluation["postprocess_mode"]),
            "--path_residual_weight", "0.15",
            "--timing_points", str(evaluation["timing_points"]),
            "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
            "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
            "--unclipped_duration",
        ]
        if output.is_file() and not args.force:
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
                raise RuntimeError(f"K={count} failed; see {log}")
        runs.append(
            {
                "candidates": count,
                "tasks": args.max_tasks,
                "seed": args.seed,
                "status": status,
                "elapsed_s": elapsed,
                "output": str(output.relative_to(ROOT)),
            }
        )
        print(f"[{status}] K={count} elapsed={elapsed:.1f}s", flush=True)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "cuda_synchronized_timing": True,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
