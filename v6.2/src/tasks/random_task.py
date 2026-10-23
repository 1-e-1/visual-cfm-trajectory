"""Random start/goal task generation from two independent uniform joint samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomTaskConfig:
    sampling_lower: np.ndarray
    sampling_upper: np.ndarray
    min_joint_distance: float
    min_tcp_distance_m: float
    max_resample_attempts: int
    reject_singular: bool
    min_jacobian_singular_value: float
    max_condition_number: float


def generate_random_task(scene, config: RandomTaskConfig, rng: np.random.Generator) -> dict[str, object]:
    lower = np.asarray(config.sampling_lower, dtype=np.float64)
    upper = np.asarray(config.sampling_upper, dtype=np.float64)
    if lower.shape != (6,) or upper.shape != (6,) or np.any(lower >= upper):
        raise ValueError("Sampling bounds must have shape (6,) with lower < upper")

    failure_counts: Counter[str] = Counter()
    failure_examples: list[dict[str, object]] = []
    for attempt in range(1, config.max_resample_attempts + 1):
        q_start = rng.uniform(lower, upper)
        q_goal = rng.uniform(lower, upper)
        if np.linalg.norm(q_goal - q_start) < config.min_joint_distance:
            failure_counts["joint_distance_too_small"] += 1
            continue

        start = scene.query(q_start)
        goal = scene.query(q_goal)
        reasons = []
        if start["collision"]:
            reasons.append("start_collision")
        if goal["collision"]:
            reasons.append("goal_collision")
        if not start["base_fixed"] or not goal["base_fixed"]:
            reasons.append("base_not_fixed")
        tcp_distance = float(np.linalg.norm(goal["tcp_position"] - start["tcp_position"]))
        if tcp_distance < config.min_tcp_distance_m:
            reasons.append("tcp_distance_too_small")
        if config.reject_singular:
            for prefix, state in (("start", start), ("goal", goal)):
                if state["min_jacobian_singular_value"] < config.min_jacobian_singular_value:
                    reasons.append(f"{prefix}_near_singular")
                if state["jacobian_condition_number"] > config.max_condition_number:
                    reasons.append(f"{prefix}_condition_number_high")

        if reasons:
            failure_counts.update(reasons)
            if len(failure_examples) < 20:
                failure_examples.append({"attempt": attempt, "reasons": reasons})
            continue

        goal_quat = np.asarray(goal["tcp_quaternion_wxyz"], dtype=np.float64)
        condition = np.concatenate((q_start, goal["tcp_position"], goal_quat))
        return {
            "q_start": q_start,
            "q_goal": q_goal,
            "start": start,
            "goal": goal,
            "condition": condition,
            "condition_dim": 13,
            "attempts": attempt,
            "tcp_distance_m": tcp_distance,
            "rejection_counts": dict(failure_counts),
            "rejection_examples": failure_examples,
        }

    raise RuntimeError(
        f"Could not generate a valid task in {config.max_resample_attempts} attempts; "
        f"rejections={dict(failure_counts)}"
    )
