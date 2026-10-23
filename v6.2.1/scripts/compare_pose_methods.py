"""Paired success and continuous-metric comparison for two pose reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "environment_clearance_m",
    "self_clearance_m",
    "velocity_rms_rad_s",
    "acceleration_rms_rad_s2",
    "jerk_rms_rad_s3",
    "duration_s",
)


def paired_bootstrap_ci(
    differences: np.ndarray, generator: np.random.Generator, draws: int
) -> list[float]:
    if len(differences) == 0:
        return [float("nan"), float("nan")]
    indices = generator.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare_stage(
    left: list[dict],
    right: list[dict],
    stage: str,
    generator: np.random.Generator,
    bootstrap_draws: int,
) -> dict[str, object]:
    left_success = np.asarray([record[stage]["success"] for record in left], dtype=np.bool_)
    right_success = np.asarray([record[stage]["success"] for record in right], dtype=np.bool_)
    left_only = int(np.sum(left_success & ~right_success))
    right_only = int(np.sum(~left_success & right_success))
    discordant = left_only + right_only
    common = np.flatnonzero(left_success & right_success)
    metrics = {}
    for metric in METRICS:
        left_values = np.asarray([left[index][stage][metric] for index in common])
        right_values = np.asarray([right[index][stage][metric] for index in common])
        differences = left_values - right_values
        try:
            wilcoxon_p = float(wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        metrics[metric] = {
            "common_success_tasks": int(len(common)),
            "left_mean": float(left_values.mean()) if len(common) else float("nan"),
            "right_mean": float(right_values.mean()) if len(common) else float("nan"),
            "paired_mean_difference_left_minus_right": (
                float(differences.mean()) if len(common) else float("nan")
            ),
            "paired_bootstrap_95": paired_bootstrap_ci(
                differences, generator, bootstrap_draws
            ),
            "wilcoxon_two_sided_p": wilcoxon_p,
        }
    return {
        "left_successes": int(left_success.sum()),
        "right_successes": int(right_success.sum()),
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "mcnemar_exact_two_sided_p": (
            float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "metrics_on_common_successes": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    left_report = json.loads(args.left.read_text(encoding="utf-8"))
    right_report = json.loads(args.right.read_text(encoding="utf-8"))
    left_by_id = {record["sample_id"]: record for record in left_report["records"]}
    right_by_id = {record["sample_id"]: record for record in right_report["records"]}
    sample_ids = sorted(set(left_by_id) & set(right_by_id))
    if len(sample_ids) != left_report["tasks"] or len(sample_ids) != right_report["tasks"]:
        raise ValueError("reports must contain the same unique sample_id set")
    left = [left_by_id[sample_id] for sample_id in sample_ids]
    right = [right_by_id[sample_id] for sample_id in sample_ids]
    generator = np.random.default_rng(args.seed)
    stages = {
        stage: compare_stage(
            left, right, stage, generator, args.bootstrap_draws
        )
        for stage in ("grasp", "place")
    }
    left_joint = np.asarray([record["two_stage_success"] for record in left], dtype=np.bool_)
    right_joint = np.asarray([record["two_stage_success"] for record in right], dtype=np.bool_)
    left_only = int(np.sum(left_joint & ~right_joint))
    right_only = int(np.sum(~left_joint & right_joint))
    discordant = left_only + right_only
    payload = {
        "left": args.left_name,
        "right": args.right_name,
        "paired_tasks": len(sample_ids),
        "difference_convention": "left_minus_right",
        "stages": stages,
        "two_stage": {
            "left_successes": int(left_joint.sum()),
            "right_successes": int(right_joint.sum()),
            "left_only_successes": left_only,
            "right_only_successes": right_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            ),
        },
    }
    output = args.output or ROOT / "outputs/paper_v622/comparisons" / (
        f"{args.left.stem}_vs_{args.right.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
