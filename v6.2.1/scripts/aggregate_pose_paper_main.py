"""Aggregate repeated sampling seeds without treating repeats as new tasks."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z / denominator * sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    return [center - radius, center + radius]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, default=ROOT / "outputs/paper_v622/main"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/paper_v622/main_summary.json"
    )
    args = parser.parse_args()

    summary: dict[str, object] = {"groups": {}}
    for group in ("raw", "full"):
        paths = sorted(args.input_dir.glob(f"{group}_seed*.json"))
        if not paths:
            continue
        loaded = [
            (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
        ]
        loaded = [item for item in loaded if "seed" in item[1] and "records" in item[1]]
        paths = [item[0] for item in loaded]
        reports = [item[1] for item in loaded]
        group_summary = {"seeds": [], "sampling_seed_statistics": {}}
        for path, report in zip(paths, reports):
            group_summary["seeds"].append(
                {
                    "seed": report["seed"],
                    "tasks": report["tasks"],
                    "grasp_successes": report["grasp"]["successes"],
                    "place_successes": report["place"]["successes"],
                    "two_stage_successes": report["two_stage"]["successes"],
                    "two_stage_wilson_95": report["two_stage"]["success_rate_wilson_95"],
                    "elapsed_s": report["elapsed_s"],
                    "source": str(path.relative_to(ROOT)),
                }
            )
        for stage in ("grasp", "place", "two_stage"):
            rates = np.asarray([report[stage]["success_rate"] for report in reports])
            group_summary["sampling_seed_statistics"][stage] = {
                "mean_success_rate": float(rates.mean()),
                "sample_std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
                "minimum": float(rates.min()),
                "maximum": float(rates.max()),
            }
        reference = reports[0]
        successes = int(reference["two_stage"]["successes"])
        tasks = int(reference["tasks"])
        group_summary["reference_task_wilson_95"] = wilson(successes, tasks)
        summary["groups"][group] = group_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
