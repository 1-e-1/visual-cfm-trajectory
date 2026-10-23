"""Copy validated source figures into their final manuscript order."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
FORMATS = ("png", "pdf", "svg")

# (paper number, source stem, paper slug, manuscript section)
ORDER = (
    (1, "fig01_task_overview", "task_overview", "3.1"),
    (2, "fig02_system_pipeline", "system_pipeline", "3.3"),
    (3, "fig03_multiview_perception", "multiview_perception", "4.1"),
    (4, "fig04_candidate_geometry", "candidate_geometry", "4.2"),
    (5, "fig05_residual_reconstruction", "residual_reconstruction", "4.4"),
    (6, "fig12_planner_pareto", "classical_planner_comparison", "6.1"),
    (7, "fig06_unseen_task_feasibility", "unseen_task_feasibility", "6.2"),
    (8, "fig15_learning_baselines", "learning_baselines", "6.3"),
    (9, "fig07_low_impact_tradeoff", "low_impact_tradeoff", "6.3"),
    (10, "fig08_candidate_scaling", "candidate_scaling", "6.4"),
    (11, "fig17_runtime_breakdown", "runtime_breakdown", "6.4"),
    (12, "fig09_executability_ablation", "executability_ablation", "6.5"),
    (13, "fig16_residual_landscape", "residual_landscape", "6.5"),
    (14, "fig10_vision_ablation", "vision_ablation", "6.6"),
    (15, "fig14_failure_modes", "failure_modes", "6.7"),
    (16, "fig11_physx_sequence", "physx_sequence", "6.8"),
)


def main() -> None:
    rows = []
    for extension in FORMATS:
        (ROOT / "manuscript" / extension).mkdir(parents=True, exist_ok=True)
    for paper_number, source_stem, paper_slug, section in ORDER:
        target_stem = f"fig{paper_number:02d}_{paper_slug}"
        for extension in FORMATS:
            source = ROOT / extension / f"{source_stem}.{extension}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, ROOT / "manuscript" / extension / f"{target_stem}.{extension}")
        rows.append(
            {
                "paper_figure": f"Fig.{paper_number}",
                "paper_file": target_stem,
                "source_file": source_stem,
                "section": section,
                "status": "validated",
            }
        )
    with (ROOT / "manuscript" / "figure_mapping.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} manuscript figures under {ROOT / 'manuscript'}")


if __name__ == "__main__":
    main()
