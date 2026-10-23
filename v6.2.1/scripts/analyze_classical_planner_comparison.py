"""Aggregate paired Visual CFM, RRT-Connect, and STOMP evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    difference = first - second
    if len(difference) == 0:
        return {"tasks": 0, "mean_difference": float("nan"), "ci95": [float("nan")] * 2}
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    estimates = difference[indices].mean(axis=1)
    return {
        "tasks": int(len(difference)),
        "ours_mean": float(first.mean()),
        "baseline_mean": float(second.mean()),
        "mean_difference": float(difference.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
    }


def flatten_ours(record: dict) -> dict:
    row = {
        "task_id": record["sample_id"],
        "method": "Visual CFM (Ours)",
        "success_grasp": int(record["grasp"]["success"]),
        "success_place": int(record["place"]["success"]),
        "success_joint": int(record["two_stage_success"]),
        "failure_reason": "NONE" if record["two_stage_success"] else "UNSAFE_CANDIDATE",
    }
    fields = {
        "planning_time_ms": ("timing_breakdown_ms", "total_stage_ms"),
        "joint_path_length_rad": (None, "joint_path_length_rad"),
        "environment_clearance_m": (None, "environment_clearance_m"),
        "self_clearance_m": (None, "self_clearance_m"),
        "acceleration_rms_rad_s2": (None, "acceleration_rms_rad_s2"),
        "jerk_rms_rad_s3": (None, "jerk_rms_rad_s3"),
        "duration_s": (None, "duration_s"),
        "terminal_position_error_m": (None, "terminal_error_m"),
        "terminal_orientation_error_rad": (None, "orientation_error_rad"),
    }
    for stage in ("grasp", "place"):
        for output, (parent, source) in fields.items():
            if parent:
                value = record[parent][stage][source]
            else:
                value = record[stage][source]
            row[f"{output}_{stage}"] = value
    return row


def flatten_classical(record: dict) -> dict:
    row = {
        "task_id": record["task_id"],
        "method": {"rrt_connect": "RRT-Connect", "stomp": "STOMP"}[record["method"]],
        "success_grasp": int(record["success_grasp"]),
        "success_place": int(record["success_place"]),
        "success_joint": int(record["success_joint"]),
        "failure_reason": record["failure_reason"],
    }
    for stage in ("grasp", "place"):
        for key, value in record[stage].items():
            if key not in {"success", "failure_reason"}:
                row[f"{key}_{stage}"] = value
    return row


def mean_stage(row: dict, field: str) -> float:
    return float(np.nanmean([row[f"{field}_grasp"], row[f"{field}_place"]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_c/full_k16_seed20260815.json",
    )
    parser.add_argument(
        "--classical",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_a/classical_planners_seed20260815.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/supplementary_experiments/experiment_a/experiment_a_summary.json",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    ours_raw = json.loads(args.ours.read_text(encoding="utf-8"))
    classical_raw = json.loads(args.classical.read_text(encoding="utf-8"))
    rows = [flatten_ours(record) for record in ours_raw["records"]]
    rows.extend(flatten_classical(record) for record in classical_raw["records"])
    methods = ["Visual CFM (Ours)", "RRT-Connect", "STOMP"]
    by_method = {
        method: {row["task_id"]: row for row in rows if row["method"] == method}
        for method in methods
    }
    common_ids = sorted(set.intersection(*(set(values) for values in by_method.values())))
    if len(common_ids) != len(ours_raw["records"]):
        raise RuntimeError(
            f"paired task mismatch: {len(common_ids)} common of {len(ours_raw['records'])} ours"
        )

    metrics = [
        "planning_time_ms",
        "joint_path_length_rad",
        "environment_clearance_m",
        "self_clearance_m",
        "acceleration_rms_rad_s2",
        "jerk_rms_rad_s3",
        "duration_s",
    ]
    summary = {}
    for method in methods:
        selected = [by_method[method][task_id] for task_id in common_ids]
        method_summary = {"tasks": len(selected)}
        for success_field in ("success_grasp", "success_place", "success_joint"):
            successes = sum(row[success_field] for row in selected)
            method_summary[success_field] = {
                "successes": int(successes),
                "rate": successes / len(selected),
                "wilson_95": wilson(successes, len(selected)),
            }
        successful = [row for row in selected if row["success_joint"]]
        method_summary["successful_task_metrics"] = {
            metric: {
                "mean": float(np.nanmean([mean_stage(row, metric) for row in successful])),
                "median": float(np.nanmedian([mean_stage(row, metric) for row in successful])),
            }
            for metric in metrics
        }
        summary[method] = method_summary

    comparisons = {}
    rng = np.random.default_rng(args.seed)
    ours = by_method["Visual CFM (Ours)"]
    for baseline_name in ("RRT-Connect", "STOMP"):
        baseline = by_method[baseline_name]
        ours_only = sum(
            ours[task_id]["success_joint"] and not baseline[task_id]["success_joint"]
            for task_id in common_ids
        )
        baseline_only = sum(
            baseline[task_id]["success_joint"] and not ours[task_id]["success_joint"]
            for task_id in common_ids
        )
        discordant = ours_only + baseline_only
        pvalue = (
            float(binomtest(min(ours_only, baseline_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        shared = [
            task_id
            for task_id in common_ids
            if ours[task_id]["success_joint"] and baseline[task_id]["success_joint"]
        ]
        comparisons[baseline_name] = {
            "mcnemar_exact": {
                "ours_only_success": int(ours_only),
                "baseline_only_success": int(baseline_only),
                "discordant": int(discordant),
                "p_value": pvalue,
            },
            "common_success_tasks": len(shared),
            "paired_bootstrap_ours_minus_baseline": {
                metric: bootstrap_difference(
                    np.asarray([mean_stage(ours[task_id], metric) for task_id in shared]),
                    np.asarray([mean_stage(baseline[task_id], metric) for task_id in shared]),
                    rng,
                    args.bootstrap_samples,
                )
                for metric in metrics
            },
        }

    report = {
        "protocol": {
            "paired_tasks": len(common_ids),
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "continuous_metrics_population": "tasks where both compared methods succeed in grasp and place",
            "success_test": "two-sided exact McNemar (binomial form)",
            "curobo_status": classical_raw["protocol"]["curobo_status"],
        },
        "methods": summary,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_name("planner_comparison_per_task.csv")
    ordered_rows = [by_method[method][task_id] for task_id in common_ids for method in methods]
    fieldnames = list(ordered_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
