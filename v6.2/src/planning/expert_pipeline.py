"""PhysX hard-rejection stage for optimized expert candidates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expert_optimizer import ExpertCandidate


def select_physx_valid_expert(
    candidates: list[ExpertCandidate],
    scene,
    tcp_path_weight: float,
    dense_substeps: int,
    tcp_path_evaluator: Callable[[Any], float],
) -> tuple[ExpertCandidate, dict[str, Any], list[dict[str, Any]]]:
    reports = []
    eligible = []
    for candidate in candidates:
        report: dict[str, Any] = {
            "initialization_index": candidate.initialization_index,
            "optimizer_success": candidate.optimizer_success,
            "optimizer_status": candidate.optimizer_status,
            "optimizer_message": candidate.optimizer_message,
            "iterations": candidate.iterations,
            "planning_time_s": candidate.planning_time_s,
            "kinematic_objective": candidate.objective,
            "soft_limits_satisfied": candidate.soft_limits_satisfied,
        }
        if not candidate.optimizer_success or not candidate.soft_limits_satisfied:
            report["rejection_reason"] = "optimizer_failed_or_planning_soft_limits_exceeded"
            reports.append(report)
            continue
        tcp_path_length = tcp_path_evaluator(candidate.trajectory.q)
        score = candidate.objective + tcp_path_weight * tcp_path_length
        report["torch_fk_tcp_path_length_m"] = tcp_path_length
        report["selection_score_with_tcp_path"] = score
        eligible.append((score, candidate, report))
        reports.append(report)
    eligible.sort(key=lambda item: item[0])
    selected = None
    selected_validation = None
    for _, candidate, report in eligible:
        validation = scene.validate_path(
            candidate.trajectory.q,
            settle_steps=1,
            dense_substeps=dense_substeps,
            stop_on_collision=True,
        )
        report["physx_path_validation"] = validation
        if validation["collision_free"]:
            selected = candidate
            selected_validation = validation
            break
        report["rejection_reason"] = "physx_self_collision"
    if selected is None or selected_validation is None:
        raise RuntimeError("No optimized candidate passed numerical constraints and dense PhysX validation")
    for _, candidate, report in eligible:
        if candidate is not selected and "physx_path_validation" not in report:
            report["physx_status"] = "not_evaluated_after_better_candidate_passed"
    return selected, selected_validation, reports
