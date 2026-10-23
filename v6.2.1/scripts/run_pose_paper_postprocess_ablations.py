"""Run post-processing ablations that match the implemented inference chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def base_command(config: dict, output: Path, seed: int, candidates: int) -> list[str]:
    evaluation = config["evaluation"]
    command = [
        str(config["python"]),
        str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
        "--data_dir", str(ROOT / config["dataset"]),
        "--checkpoint", str(ROOT / config["checkpoint"]),
        "--config", str(ROOT / config["model_config"]),
        "--output", str(output),
        "--candidates", str(candidates),
        "--ode_steps", str(evaluation["ode_steps"]),
        "--max_tasks", str(evaluation["tasks"]),
        "--seed", str(seed),
        "--position_tolerance_m", str(evaluation["position_tolerance_m"]),
        "--orientation_tolerance_rad", str(evaluation["orientation_tolerance_rad"]),
        "--required_clearance_m", str(evaluation["required_clearance_m"]),
        "--self_capsule_radius_m", str(evaluation["self_capsule_radius_m"]),
        "--independent_stage_generators",
    ]
    if evaluation.get("exclude_carried_target_from_cloud", False):
        command.extend(
            [
                "--exclude_carried_target_from_cloud",
                "--carried_target_exclusion_padding_m",
                str(evaluation["carried_target_exclusion_padding_m"]),
            ]
        )
    return command


def matrix(config: dict) -> list[tuple[str, int, list[str], str]]:
    evaluation = config["evaluation"]
    full_k = int(evaluation["candidates"])
    common = [
        "--postprocess",
        "--postprocess_mode", "reference_spline",
        "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
        "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
    ]
    return [
        ("raw", full_k, [], "Raw 64-point CFM samples without projection or retiming."),
        (
            "reference_residual_0",
            1,
            common
            + [
                "--path_residual_weight", "0",
                "--timing_points", "256",
                "--unclipped_duration",
            ],
            "Minimum-jerk reference geometry with no CFM path residual.",
        ),
        (
            "reference_residual_0p15",
            full_k,
            common
            + [
                "--path_residual_weight", "0.15",
                "--timing_points", "256",
                "--unclipped_duration",
            ],
            "Current complete post-processing configuration.",
        ),
        (
            "reference_residual_1",
            full_k,
            common
            + [
                "--path_residual_weight", "1.0",
                "--timing_points", "256",
                "--unclipped_duration",
            ],
            "Retain the full CFM residual around the reference curve.",
        ),
        (
            "geometry_preserving",
            full_k,
            [
                "--postprocess",
                "--postprocess_mode", "geometry_preserving",
                "--geometry_control_points", "16",
                "--timing_points", "256",
                "--unclipped_duration",
                "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
                "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
            ],
            "Periodic geometry-preserving endpoint projection.",
        ),
        (
            "reference_timing_64",
            full_k,
            common
            + [
                "--path_residual_weight", "0.15",
                "--timing_points", "64",
                "--unclipped_duration",
            ],
            "Current geometry with 64-point dense retiming/checking.",
        ),
        (
            "reference_hard_5s",
            full_k,
            common
            + [
                "--path_residual_weight", "0.15",
                "--timing_points", "64",
            ],
            "Legacy 5-second hard-clipped duration behavior.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--candidate_cap", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiments = matrix(config)
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    if selected:
        unknown = selected - {name for name, _, _, _ in experiments}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        experiments = [item for item in experiments if item[0] in selected]
    output_root = ROOT / config["output_root"] / "postprocess_ablations"
    output_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for name, candidates, extra, note in experiments:
        if args.candidate_cap:
            candidates = min(candidates, args.candidate_cap)
        output = output_root / f"{name}.json"
        log = output.with_suffix(".log")
        command = base_command(config, output, args.seed, candidates) + extra
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
        runs.append(
            {
                "name": name,
                "candidates": candidates,
                "note": note,
                "status": status,
                "elapsed_s": elapsed,
                "output": str(output.relative_to(ROOT)),
            }
        )
        print(f"[{status}] {name} elapsed={elapsed:.1f}s", flush=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"protocol_id": config["protocol_id"], "runs": runs}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
