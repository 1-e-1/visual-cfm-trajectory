"""Aggregate synchronized runtime reports into a plotting CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=ROOT / "outputs/paper_v622/runtime_breakdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_figures/data_cache/runtime_breakdown.csv",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("k*.json"), key=lambda p: int(p.stem[1:])):
        report = json.loads(path.read_text(encoding="utf-8"))
        timing = report.get("mean_timing_breakdown_ms")
        if timing is None:
            raise RuntimeError(f"timing fields missing from {path}")
        rows.append(
            {
                "candidates": int(report["candidates"]),
                "tasks": int(report["tasks"]),
                "seed": int(report["seed"]),
                **timing,
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no k*.json under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runtime settings to {args.output}")


if __name__ == "__main__":
    main()
