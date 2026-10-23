"""Run the paired residual-weight mechanism sweep with a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00)


def weight_tag(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--weights", default=",".join(str(value) for value in WEIGHTS))
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max_tasks", type=int, default=49)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]
    output_root = ROOT / config["output_root"] / "residual_weight_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []

    for weight in weights:
        tag = weight_tag(weight)
        output = output_root / f"weight_{tag}.json"
        log = output.with_suffix(".log")
        command = [
            str(config["python"]),
            str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
            "--data_dir", str(ROOT / config["dataset"]),
            "--checkpoint", str(ROOT / config["checkpoint"]),
            "--config", str(ROOT / config["model_config"]),
            "--output", str(output),
            "--candidates", str(args.candidates),
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
            "--postprocess_mode", "reference_spline",
            "--path_residual_weight", str(weight),
            "--timing_points", str(evaluation["timing_points"]),
            "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
            "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
            "--unclipped_duration",
        ]
        if output.exists() and not args.force:
            status = "reused"
            elapsed = 0.0
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
                raise RuntimeError(f"weight={weight:.2f} failed; see {log}")
        runs.append(
            {
                "residual_weight": weight,
                "candidates": args.candidates,
                "tasks": args.max_tasks,
                "seed": args.seed,
                "status": status,
                "elapsed_s": elapsed,
                "output": str(output.relative_to(ROOT)),
                "log": str(log.relative_to(ROOT)),
            }
        )
        print(f"[{status}] residual_weight={weight:.2f} elapsed={elapsed:.1f}s", flush=True)

    manifest = {
        "protocol_id": config["protocol_id"],
        "paired_task_set": True,
        "frozen_checkpoint": str(config["checkpoint"]),
        "runs": runs,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
