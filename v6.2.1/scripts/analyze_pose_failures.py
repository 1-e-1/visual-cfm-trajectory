"""Create criterion-level failure attribution from a pose evaluation report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRITERIA = (
    "terminal_ok",
    "orientation_ok",
    "environment_ok",
    "self_clearance_ok",
    "dynamics_ok",
    "duration_ok",
)


def stage_summary(records: list[dict], stage: str) -> dict[str, object]:
    failures = []
    for record in records:
        metric = record[stage]
        if metric["success"]:
            continue
        failed = tuple(name for name in CRITERIA if not metric[name])
        failures.append(failed)
    criterion_counts = {
        criterion: sum(criterion in failure for failure in failures)
        for criterion in CRITERIA
    }
    combination_counts = Counter(
        "+".join(failure) if failure else "unclassified" for failure in failures
    )
    candidate_availability = {
        criterion: sum(
            bool(record[f"{stage}_has_passing_candidate"][criterion])
            for record in records
        )
        for criterion in (*CRITERIA, "success")
    }
    return {
        "failures": len(failures),
        "criterion_failure_counts_nonexclusive": criterion_counts,
        "failure_combination_counts": dict(combination_counts.most_common()),
        "tasks_with_at_least_one_passing_candidate": candidate_availability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    records = report["records"]
    joint = Counter()
    for record in records:
        grasp = bool(record["grasp"]["success"])
        place = bool(record["place"]["success"])
        if grasp and place:
            joint["both_success"] += 1
        elif not grasp and place:
            joint["grasp_only_failure"] += 1
        elif grasp and not place:
            joint["place_only_failure"] += 1
        else:
            joint["both_failure"] += 1
    payload = {
        "source": str(args.report),
        "tasks": len(records),
        "grasp": stage_summary(records, "grasp"),
        "place": stage_summary(records, "place"),
        "joint_outcomes": dict(joint),
    }
    output = args.output or args.report.with_name(args.report.stem + "_failures.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
