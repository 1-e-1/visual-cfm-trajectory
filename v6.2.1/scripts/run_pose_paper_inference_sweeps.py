"""Run candidate-count and Heun-step inference sweeps for the frozen model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def evaluate_command(
    config: dict, *, candidates: int, ode_steps: int, seed: int, output: Path
) -> list[str]:
    evaluation = config["evaluation"]
    command = [
        str(config["python"]),
        str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
        "--data_dir", str(ROOT / config["dataset"]),
        "--checkpoint", str(ROOT / config["checkpoint"]),
        "--config", str(ROOT / config["model_config"]),
        "--output", str(output),
        "--candidates", str(candidates),
        "--ode_steps", str(ode_steps),
        "--max_tasks", str(evaluation["tasks"]),
        "--seed", str(seed),
        "--position_tolerance_m", str(evaluation["position_tolerance_m"]),
        "--orientation_tolerance_rad", str(evaluation["orientation_tolerance_rad"]),
        "--required_clearance_m", str(evaluation["required_clearance_m"]),
        "--self_capsule_radius_m", str(evaluation["self_capsule_radius_m"]),
        "--independent_stage_generators",
        "--postprocess",
        "--postprocess_mode", evaluation["postprocess_mode"],
        "--geometry_control_points", str(evaluation["geometry_control_points"]),
        "--timing_points", str(evaluation["timing_points"]),
        "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
        "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
    ]
    if evaluation["unclipped_duration"]:
        command.append("--unclipped_duration")
    if evaluation.get("exclude_carried_target_from_cloud", False):
        command.extend(
            [
                "--exclude_carried_target_from_cloud",
                "--carried_target_exclusion_padding_m",
                str(evaluation["carried_target_exclusion_padding_m"]),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--candidate_counts", default="1,8,16,64")
    parser.add_argument("--ode_steps", default="8,16,32,64")
    parser.add_argument("--ode_candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference_steps = int(config["evaluation"]["ode_steps"])
    candidate_counts = [
        int(item) for item in args.candidate_counts.split(",") if item.strip()
    ]
    ode_steps = [int(item) for item in args.ode_steps.split(",") if item.strip()]
    output_root = ROOT / config["output_root"] / "inference_sweeps"
    output_root.mkdir(parents=True, exist_ok=True)

    specifications = []
    specifications.extend(
        ("candidates", value, reference_steps) for value in candidate_counts
    )
    specifications.extend(
        ("ode_steps", args.ode_candidates, value)
        for value in ode_steps
    )
    runs = []
    for sweep, candidates, steps in specifications:
        output = output_root / f"{sweep}_k{candidates}_steps{steps}.json"
        log = output.with_suffix(".log")
        command = evaluate_command(
            config,
            candidates=candidates,
            ode_steps=steps,
            seed=args.seed,
            output=output,
        )
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
                raise RuntimeError(f"sweep failed; see {log}")
        runs.append(
            {
                "sweep": sweep,
                "candidates": candidates,
                "ode_steps": steps,
                "seed": args.seed,
                "status": status,
                "elapsed_s": elapsed,
                "output": str(output.relative_to(ROOT)),
            }
        )
        print(
            f"[{status}] {sweep} K={candidates} steps={steps} elapsed={elapsed:.1f}s",
            flush=True,
        )

    manifest = {
        "protocol_id": config["protocol_id"],
        "note": "Single fixed sampling seed; efficiency/sensitivity sweep, not an independent-task CI.",
        "runs": runs,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
