"""Run the frozen raw/full Pose Visual CFM paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def command_for(config: dict, group: str, seed: int, output: Path) -> list[str]:
    evaluation = config["evaluation"]
    command = [
        str(config["python"]),
        str(ROOT / "scripts/evaluate_visual_cfm_v621.py"),
        "--data_dir", str(ROOT / config["dataset"]),
        "--checkpoint", str(ROOT / config["checkpoint"]),
        "--config", str(ROOT / config["model_config"]),
        "--output", str(output),
        "--candidates", str(evaluation["candidates"]),
        "--ode_steps", str(evaluation["ode_steps"]),
        "--max_tasks", str(evaluation["tasks"]),
        "--seed", str(seed),
        "--position_tolerance_m", str(evaluation["position_tolerance_m"]),
        "--orientation_tolerance_rad", str(evaluation["orientation_tolerance_rad"]),
        "--required_clearance_m", str(evaluation["required_clearance_m"]),
        "--self_capsule_radius_m", str(evaluation["self_capsule_radius_m"]),
    ]
    if evaluation.get("independent_stage_generators", False):
        command.append("--independent_stage_generators")
    if evaluation.get("exclude_carried_target_from_cloud", False):
        command.extend(
            [
                "--exclude_carried_target_from_cloud",
                "--carried_target_exclusion_padding_m",
                str(evaluation["carried_target_exclusion_padding_m"]),
            ]
        )
    if config["main_groups"][group]["postprocess"]:
        command.extend(
            [
                "--postprocess",
                "--postprocess_mode", evaluation["postprocess_mode"],
                "--geometry_control_points", str(evaluation["geometry_control_points"]),
                "--timing_points", str(evaluation["timing_points"]),
                "--nominal_max_duration_s", str(evaluation["nominal_max_duration_s"]),
                "--max_output_duration_s", str(evaluation["max_output_duration_s"]),
            ]
        )
        if evaluation["unclipped_duration"]:
            command.append("--unclipped_duration")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/paper_v622_pose_experiments.yaml",
    )
    parser.add_argument("--groups", default="raw,full")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"] / "main"
    output_root.mkdir(parents=True, exist_ok=True)
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in config["evaluation"]["sampling_seeds"]]
    )
    unknown = sorted(set(groups) - set(config["main_groups"]))
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")

    runs = []
    for group in groups:
        for seed in seeds:
            output = output_root / f"{group}_seed{seed}.json"
            log = output.with_suffix(".log")
            command = command_for(config, group, int(seed), output)
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
                    raise RuntimeError(f"{group} seed {seed} failed; see {log}")
            runs.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "status": status,
                    "elapsed_s": elapsed,
                    "output": str(output.relative_to(ROOT)),
                    "log": str(log.relative_to(ROOT)),
                    "command": command,
                }
            )
            print(f"[{status}] {group} seed={seed} elapsed={elapsed:.1f}s", flush=True)

    manifest = {
        "protocol_id": config["protocol_id"],
        "config": str(args.config.relative_to(ROOT)),
        "runs": runs,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
