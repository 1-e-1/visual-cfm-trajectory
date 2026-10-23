"""Aggregate modality and robustness reports into a compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=Path, default=ROOT / "outputs/paper_v622/robustness"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/paper_v622/robustness_summary.json",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": path.stem,
                "tasks": report["tasks"],
                "grasp_success_rate": report["grasp"]["success_rate"],
                "place_success_rate": report["place"]["success_rate"],
                "two_stage_success_rate": report["two_stage"]["success_rate"],
                "two_stage_wilson_95": report["two_stage"]["success_rate_wilson_95"],
                "target_component_accuracy": report["target_component_accuracy"],
                "target_center_mean_error_m": report["target_center_mean_error_m"],
                "target_type_accuracy": report["target_type_accuracy"],
                "elapsed_s": report["elapsed_s"],
            }
        )
    payload = {
        "boundary": (
            "All rows use stored scene_points for final safety selection. The generator_visual_only "
            "row is therefore a generator-input ablation, not an end-to-end pure-vision result."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
