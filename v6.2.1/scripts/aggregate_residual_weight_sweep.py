"""Aggregate paired residual-weight runs into a traceable paper CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def weight_from_path(path: Path) -> float:
    return float(path.stem.removeprefix("weight_").replace("p", "."))


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=ROOT / "outputs/paper_v622/residual_weight_sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_figures/data_cache/residual_weight_sweep.csv",
    )
    args = parser.parse_args()

    rows = []
    expected_ids: list[str] | None = None
    for path in sorted(args.input_dir.glob("weight_*.json"), key=weight_from_path):
        report = json.loads(path.read_text(encoding="utf-8"))
        records = report["records"]
        sample_ids = [str(record["sample_id"]) for record in records]
        if expected_ids is None:
            expected_ids = sample_ids
        elif sample_ids != expected_ids:
            raise RuntimeError(f"unpaired task order in {path}")
        joint_success = [record for record in records if record["two_stage_success"]]

        def both_fail(key: str) -> int:
            return sum(
                not (record["grasp"][key] and record["place"][key])
                for record in records
            )

        path_lengths = [
            0.5
            * (
                record["grasp"]["joint_path_length_rad"]
                + record["place"]["joint_path_length_rad"]
            )
            for record in joint_success
            if "joint_path_length_rad" in record["grasp"]
        ]
        rows.append(
            {
                "residual_weight": weight_from_path(path),
                "tasks": len(records),
                "candidates": int(report["candidates"]),
                "seed": int(report["seed"]),
                "grasp_successes": sum(record["grasp"]["success"] for record in records),
                "place_successes": sum(record["place"]["success"] for record in records),
                "joint_successes": len(joint_success),
                "joint_success_rate": len(joint_success) / len(records),
                "environment_failure_tasks": both_fail("environment_ok"),
                "self_collision_failure_tasks": both_fail("self_clearance_ok"),
                "dynamic_failure_tasks": both_fail("dynamics_ok"),
                "mean_acc_rms_common_success": mean_or_nan(
                    [
                        0.5
                        * (
                            record["grasp"]["acceleration_rms_rad_s2"]
                            + record["place"]["acceleration_rms_rad_s2"]
                        )
                        for record in joint_success
                    ]
                ),
                "mean_jerk_rms_common_success": mean_or_nan(
                    [
                        0.5
                        * (
                            record["grasp"]["jerk_rms_rad_s3"]
                            + record["place"]["jerk_rms_rad_s3"]
                        )
                        for record in joint_success
                    ]
                ),
                "mean_path_length_rad_common_success": mean_or_nan(path_lengths),
                "source_file": str(path.relative_to(ROOT)),
            }
        )

    if not rows:
        raise FileNotFoundError(f"no weight_*.json under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired settings to {args.output}")


if __name__ == "__main__":
    main()
