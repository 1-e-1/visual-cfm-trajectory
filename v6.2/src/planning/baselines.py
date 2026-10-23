"""Public baseline trajectory API."""

from .trajectory_math import JointTrajectory, linear_trajectory, minimum_jerk_duration, minimum_jerk_trajectory

__all__ = ["JointTrajectory", "linear_trajectory", "minimum_jerk_duration", "minimum_jerk_trajectory"]
