"""Run deterministic input-modality and sensor-perturbation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def base_command(
    config: dict, output: Path, seed: int, candidates: int
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
        "--ode_steps", str(evaluation["ode_steps"]),
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


def experiment_matrix() -> list[tuple[str, list[str], str]]:
    experiments = [
        ("clean", [], "Frozen clean reference with independent stage RNG streams."),
        ("rgb_std_5", ["--rgb_noise_std", "5"], "Gaussian RGB noise, std=5 pixel levels."),
        ("rgb_std_15", ["--rgb_noise_std", "15"], "Gaussian RGB noise, std=15 pixel levels."),
        ("depth_std_0p005", ["--depth_noise_std", "0.005"], "Gaussian depth noise, std=5 mm."),
        ("depth_std_0p02", ["--depth_noise_std", "0.02"], "Gaussian depth noise, std=20 mm."),
        (
            "combined_rgb15_depth20mm",
            ["--rgb_noise_std", "15", "--depth_noise_std", "0.02"],
            "Combined RGB and depth noise.",
        ),
        ("semantic_zero", ["--semantic_mode", "zero"], "Remove semantic images."),
        (
            "geometry_tokens_zero",
            ["--zero_geometry_tokens"],
            "Remove geometric values from obstacle tokens; mask structure remains.",
        ),
        (
            "generator_visual_only",
            ["--zero_geometry_tokens", "--semantic_mode", "zero"],
            "RGB-D-only generation input; selector still uses stored scene_points.",
        ),
        (
            "visual_blackout_all",
            ["--camera_blackout_indices", "0,1,2,3,4", "--semantic_mode", "zero"],
            "Remove all camera observations while retaining geometry tokens.",
        ),
    ]
    view_names = ("wrist", "cube", "chest", "abdomen", "head")
    experiments.extend(
        (
            f"blackout_{name}",
            ["--camera_blackout_indices", str(index)],
            f"Black out the {name} camera while preserving tensor shape.",
        )
        for index, name in enumerate(view_names)
    )
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"] / "robustness"
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    matrix = experiment_matrix()
    if selected:
        unknown = selected - {name for name, _, _ in matrix}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        matrix = [item for item in matrix if item[0] in selected]

    runs = []
    for name, extra_args, note in matrix:
        output = output_root / f"{name}.json"
        log = output.with_suffix(".log")
        command = base_command(config, output, args.seed, args.candidates) + extra_args
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
                raise RuntimeError(f"{name} failed; see {log}")
        runs.append(
            {
                "name": name,
                "note": note,
                "status": status,
                "elapsed_s": elapsed,
                "output": str(output.relative_to(ROOT)),
                "command": command,
            }
        )
        print(f"[{status}] {name} elapsed={elapsed:.1f}s", flush=True)

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "seed": args.seed,
                "candidates": args.candidates,
                "paired_tasks": True,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
