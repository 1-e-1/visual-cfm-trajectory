"""Aggregate the frozen valid candidate-count and Heun-step sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def row(path: Path, sweep: str) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sweep": sweep,
        "source": str(path.relative_to(ROOT)),
        "candidates": report["candidates"],
        "ode_steps": report["ode_steps"],
        "tasks": report["tasks"],
        "grasp_successes": report["grasp"]["successes"],
        "place_successes": report["place"]["successes"],
        "two_stage_successes": report["two_stage"]["successes"],
        "two_stage_success_rate": report["two_stage"]["success_rate"],
        "two_stage_wilson_95": report["two_stage"]["success_rate_wilson_95"],
        "grasp_mean_raw_terminal_error_m": report["grasp"][
            "mean_raw_terminal_error_all"
        ],
        "place_mean_raw_terminal_error_m": report["place"][
            "mean_raw_terminal_error_all"
        ],
        "elapsed_s": report["elapsed_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=ROOT / "outputs/paper_v622/inference_sweeps",
    )
    parser.add_argument(
        "--main_reference",
        type=Path,
        default=ROOT / "outputs/paper_v622/main/full_seed20260814.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_v622/inference_sweeps_summary.json",
    )
    args = parser.parse_args()

    candidate_paths = [
        args.input_dir / f"candidates_k{count}_steps32.json"
        for count in (1, 8, 16)
    ]
    ode_paths = [
        args.input_dir / f"ode_steps_k8_steps{steps}.json"
        for steps in (8, 16, 32, 64)
    ]
    required = [*candidate_paths, *ode_paths, args.main_reference]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen sweep reports: {missing}")
    candidate_rows = [row(path, "candidate_count") for path in candidate_paths]
    candidate_rows.append(row(args.main_reference, "candidate_count"))
    ode_rows = [row(path, "heun_steps") for path in ode_paths]
    payload = {
        "protocol": {
            "paired_tasks": 49,
            "sampling_seed": 20260814,
            "candidate_sweep_heun_steps": 32,
            "heun_sweep_candidates": 8,
            "independent_stage_generators": True,
            "postprocess": "reference_spline + 256-point unclipped retiming",
        },
        "candidate_count_rows": candidate_rows,
        "heun_step_rows": ode_rows,
        "interpretation_boundary": (
            "Rows share the same tasks and sampling seed. Compare paired outcomes and "
            "latency; do not treat rows as independent binomial samples."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
