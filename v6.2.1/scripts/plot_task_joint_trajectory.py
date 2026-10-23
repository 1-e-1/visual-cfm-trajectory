"""Plot one generated grasp-place task trajectory with Matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEGMENTS = (
    ("transfer_to_pregrasp_q", "transfer_to_pregrasp_duration", "生成至预抓取"),
    ("approach_q", "approach_duration", "接近物块"),
    ("lift_q", "lift_duration", "夹取后抬升"),
    ("transfer_to_preplace_q", "transfer_to_preplace_duration", "搬运至预放置"),
    ("descend_q", "descend_duration", "下降放置"),
)

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#4D4D4D")
PHASE_COLORS = ("#EAF3F8", "#FFF3E7", "#EAF6F0", "#F5ECF3", "#F4F1E7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/nasa_astronaut_aux_arm_20tasks/plans/task_01_center.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/task_trajectory_matplotlib",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def load_timed_trajectory(
    plan_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str | int]]]:
    times: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    phases: list[dict[str, float | str | int]] = []
    elapsed = 0.0

    with np.load(plan_path, allow_pickle=False) as plan:
        for segment_index, (q_key, duration_key, label) in enumerate(SEGMENTS):
            q = np.asarray(plan[q_key], dtype=np.float64)
            duration = float(plan[duration_key])
            if q.ndim != 2 or q.shape[1] != 6 or q.shape[0] < 2:
                raise ValueError(f"invalid trajectory tensor {q_key}: {q.shape}")
            if not np.isfinite(q).all() or not np.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"invalid values in {q_key}/{duration_key}")

            segment_time = np.linspace(elapsed, elapsed + duration, q.shape[0])
            if segment_index:
                segment_time = segment_time[1:]
                q = q[1:]
            times.append(segment_time)
            trajectories.append(q)
            phases.append(
                {
                    "label": label,
                    "start_s": elapsed,
                    "end_s": elapsed + duration,
                    "duration_s": duration,
                    "points": int(q.shape[0] + (1 if segment_index else 0)),
                }
            )
            elapsed += duration

    time = np.concatenate(times)
    q_wrapped = np.concatenate(trajectories, axis=0)
    q_continuous = np.unwrap(q_wrapped, axis=0, period=2.0 * np.pi)
    return time, q_continuous, phases


def add_phase_background(axis: plt.Axes, phases: list[dict[str, float | str | int]]) -> None:
    for index, phase in enumerate(phases):
        start = float(phase["start_s"])
        end = float(phase["end_s"])
        axis.axvspan(start, end, color=PHASE_COLORS[index], alpha=0.55, zorder=0)
        if index:
            axis.axvline(start, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)


def plot_trajectory(
    time: np.ndarray,
    q: np.ndarray,
    phases: list[dict[str, float | str | int]],
    output_dir: Path,
    task_name: str,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12.4, 7.4), sharex=True)
    groups = ((0, 1, 2), (3, 4, 5))
    legend_handles = []
    legend_labels = []
    for axis, joint_indices in zip(axes, groups):
        add_phase_background(axis, phases)
        for joint_index in joint_indices:
            line = axis.plot(
                time,
                q[:, joint_index],
                color=COLORS[joint_index],
                linewidth=1.55,
                label=rf"$q_{joint_index + 1}$",
            )[0]
            legend_handles.append(line)
            legend_labels.append(line.get_label())
        axis.set_ylabel("关节角 / rad")
        axis.grid(True, color="#B8B8B8", linewidth=0.55, alpha=0.55)
        axis.set_xlim(float(time[0]), float(time[-1]))

    axes[0].set_title("关节 1-3")
    axes[1].set_title("关节 4-6")
    axes[1].set_xlabel("物理时间 / s")

    y_max = axes[0].get_ylim()[1]
    y_min = axes[0].get_ylim()[0]
    label_y = y_max - 0.035 * (y_max - y_min)
    for phase in phases:
        center = 0.5 * (float(phase["start_s"]) + float(phase["end_s"]))
        axes[0].text(
            center,
            label_y,
            str(phase["label"]),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333333",
        )

    total_duration = float(time[-1] - time[0])
    figure.suptitle(
        f"一次抓取-放置任务的生成关节轨迹\n{task_name} | 总时长 {total_duration:.2f} s",
        fontsize=14,
        fontweight="bold",
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=True,
        framealpha=0.95,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{task_name}_joint_trajectory.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{task_name}_joint_trajectory.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    time, q, phases = load_timed_trajectory(args.plan.resolve())
    task_name = args.plan.stem
    plot_trajectory(time, q, phases, args.output.resolve(), task_name)
    print(
        f"task={task_name} points={len(time)} duration_s={time[-1]:.6f} "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
