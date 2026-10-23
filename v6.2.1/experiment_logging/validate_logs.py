"""Validate task pairing, failure enums, units, durations and trajectory files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from schema import FAILURE_REASONS, REQUIRED_COLUMNS


def _missing(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    try:
        return math.isnan(float(value))
    except ValueError:
        return False


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def validate(summary_path: Path) -> dict[str, object]:
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(set(REQUIRED_COLUMNS) - columns)
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
    methods_by_task: dict[str, set[str]] = defaultdict(set)
    method_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=2):
        prefix = f"row {index}"
        methods_by_task[row.get("task_id", "")].add(row.get("method", ""))
        method_counts[row.get("method", "")] += 1
        reason = row.get("failure_reason", "")
        success = _as_bool(row.get("success_joint", "0"))
        if not success and reason not in FAILURE_REASONS:
            errors.append(f"{prefix}: invalid failure_reason={reason!r}")
        for key in ("duration_grasp", "duration_place"):
            value = row.get(key)
            if _missing(value):
                warnings.append(f"{prefix}: {key} is NaN")
            elif float(value) <= 0.0:
                errors.append(f"{prefix}: {key} must be > 0")
        trajectory = row.get("trajectory_file", "")
        if success and (_missing(trajectory) or not Path(trajectory).is_file()):
            errors.append(f"{prefix}: successful row lacks trajectory file: {trajectory}")
        for key in REQUIRED_COLUMNS:
            if key in row and _missing(row[key]):
                warnings.append(f"{prefix}: required field {key} is NaN")
    expected_methods = set(method_counts)
    for task_id, methods in methods_by_task.items():
        if methods != expected_methods:
            errors.append(
                f"task {task_id!r} is not paired: has {sorted(methods)}, "
                f"expected {sorted(expected_methods)}"
            )
    return {
        "summary": str(summary_path.resolve()),
        "rows": len(rows),
        "tasks": len(methods_by_task),
        "methods": dict(method_counts),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.summary)
    output = args.output or args.summary.with_name("validation_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
