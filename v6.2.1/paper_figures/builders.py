"""Figure builders backed only by existing project artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch

from common import (
    annotate_panel_labels,
    bootstrap_ci,
    configure_paper_rcparams,
    data_required,
    image_or_placeholder,
    load_csv_or_json,
    paired_bootstrap_diff,
    save_figure_all_formats,
    wilson_ci,
    write_csv,
)
from constants import DOUBLE_COLUMN_IN, METHOD_STYLES, STAGE_COLORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "paper_figures"
sys.path.insert(0, str(PROJECT_ROOT))


def _save(fig: plt.Figure, number: int, slug: str) -> list[str]:
    fig.tight_layout()
    return save_figure_all_formats(fig, FIGURE_ROOT, f"fig{number:02d}_{slug}")


def _box(axis: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.015", fc=color, ec="#444444", lw=0.7)
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7)


def fig01(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    import imageio.v3 as iio

    video = PROJECT_ROOT / "outputs/nasa_astronaut_no_intersection_demo/media_final/pose_grasp_place_final.mp4"
    if not video.is_file():
        return _missing_figure(1, "task_overview", ["Validated PhysX rollout video"])
    frames = list(iio.imiter(video))
    labels = ["Task setup", "Approach and grasp", "Object transfer", "Place and release"]
    indices = [25, 92, 205, min(292, len(frames) - 1)]
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 2.05))
    grid = fig.add_gridspec(2, 4, height_ratios=[3.2, 0.8], hspace=0.06, wspace=0.05)
    image_axes = []
    for column, (label, index) in enumerate(zip(labels, indices)):
        axis = fig.add_subplot(grid[0, column])
        frame = frames[min(index, len(frames) - 1)]
        crop = frame[int(0.04 * len(frame)) :]
        axis.imshow(crop)
        axis.set_axis_off()
        axis.set_title(label, pad=3)
        image_axes.append(axis)
    constraint_axis = fig.add_subplot(grid[1, :])
    constraint_axis.set_xlim(0, 1)
    constraint_axis.set_ylim(0, 1)
    constraint_axis.axis("off")
    constraints = [
        ("Pose accuracy", "#DCEAF7"),
        ("Environment and self-collision", "#E4F3E8"),
        ("Low acceleration and jerk", "#FFF0D5"),
        ("Stable contact transfer", "#EEE7F5"),
    ]
    for index, (label, color) in enumerate(constraints):
        _box(constraint_axis, (0.02 + index * 0.245, 0.23), 0.225, 0.54, label, color)
    annotate_panel_labels(image_axes)
    return _save(fig, 1, "task_overview")


def fig02(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 3.25))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0.015, 0.69), 0.97, 0.26, fc="#F7F8FA", ec="#CCCCCC", lw=0.7))
    axis.add_patch(Rectangle((0.015, 0.39), 0.97, 0.23, fc="#EEF4F8", ec="#A9C5DE", lw=0.7))
    axis.add_patch(Rectangle((0.015, 0.05), 0.97, 0.27, fc="#F1F8F2", ec="#A9CDB3", lw=0.7))
    axis.text(0.03, 0.91, "Task inputs", weight="bold", color="#444444")
    axis.text(0.03, 0.58, "Generative proposal", weight="bold", color="#1769AA")
    axis.text(0.03, 0.28, "Deterministic verification and execution", weight="bold", color="#217A4B")

    input_boxes = [
        (0.08, "$q_{start}$\nrobot state", 0.18),
        (0.34, "Grasp / place\ntarget pose", 0.20),
        (0.64, "Five-view observations\n+ geometry tokens", 0.25),
    ]
    for x_value, label, width in input_boxes:
        _box(axis, (x_value, 0.735), width, 0.12, label, "#FFFFFF")
        axis.add_patch(FancyArrowPatch((x_value + width / 2, 0.735), (0.26, 0.56), arrowstyle="->", mutation_scale=8, lw=0.8))

    generation = [
        (0.15, "Condition\nencoder", 0.21),
        (0.43, "Periodic CFM\n32-step Heun", 0.21),
        (0.71, "$K$ trajectory\ncandidates", 0.21),
    ]
    for index, (x_value, label, width) in enumerate(generation):
        _box(axis, (x_value, 0.445), width, 0.12, label, "#DCEAF7")
        if index < len(generation) - 1:
            next_x = generation[index + 1][0]
            axis.add_patch(FancyArrowPatch((x_value + width, 0.505), (next_x, 0.505), arrowstyle="->", mutation_scale=8, lw=0.8))
    # Route the transition between bands through the whitespace so it does not
    # cross any model block.
    axis.plot([0.815, 0.815, 0.12, 0.12], [0.445, 0.355, 0.355, 0.225], color="#333333", lw=0.8)
    axis.add_patch(FancyArrowPatch((0.12, 0.225), (0.12, 0.21), arrowstyle="->", mutation_scale=8, lw=0.8))
    axis.text(0.47, 0.365, "verify each candidate", ha="center", va="bottom", fontsize=6.2, color="#555555")

    verification = [
        (0.05, "Terminal\nprojection", 0.14),
        (0.22, "Minimum-jerk\n+ CFM residual", 0.17),
        (0.42, "256-point\ntime scaling", 0.15),
        (0.60, "Collision +\ndynamics screen", 0.16),
        (0.79, "Selected path\n+ PhysX task", 0.16),
    ]
    for index, (x_value, label, width) in enumerate(verification):
        _box(axis, (x_value, 0.105), width, 0.105, label, "#E4F3E8" if index >= 3 else "#FFF0D5")
        if index < len(verification) - 1:
            next_x = verification[index + 1][0]
            axis.add_patch(FancyArrowPatch((x_value + width, 0.158), (next_x, 0.158), arrowstyle="->", mutation_scale=8, lw=0.8))
    return _save(fig, 2, "system_pipeline")


def fig03(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    cameras = ["wrist", "cube", "chest", "abdomen", "head"]
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 3.25))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], hspace=0.12, wspace=0.05)
    base = PROJECT_ROOT / "outputs/physical_camera_debug"
    image_axes = []
    for column, camera in enumerate(cameras):
        axis = fig.add_subplot(grid[0, column])
        image_or_placeholder(axis, base / f"layout_007_{camera}.png", camera.title())
        image_axes.append(axis)
    flow = fig.add_subplot(grid[1, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    _box(flow, (0.04, 0.58), 0.17, 0.18, "Five-view, two-frame\nRGB-D-semantic input", "#FFFFFF")
    _box(flow, (0.28, 0.58), 0.19, 0.18, "Shared ResNet\n+ view embedding", "#DCEAF7")
    _box(flow, (0.54, 0.58), 0.17, 0.18, "FiLM($p_{goal}$)\nvisual feature", "#DCEAF7")
    _box(flow, (0.28, 0.18), 0.19, 0.18, "Component tokens\n+ Transformer", "#FFF0D5")
    _box(flow, (0.54, 0.18), 0.17, 0.18, "Predicted\ntarget token", "#FFF0D5")
    _box(flow, (0.80, 0.38), 0.16, 0.18, "Fused condition\n$z$", "#E4F3E8")
    for start, end in [
        ((0.21, 0.67), (0.28, 0.67)),
        ((0.47, 0.67), (0.54, 0.67)),
        ((0.375, 0.58), (0.375, 0.36)),
        ((0.47, 0.27), (0.54, 0.27)),
        ((0.71, 0.67), (0.80, 0.49)),
        ((0.71, 0.27), (0.80, 0.45)),
    ]:
        flow.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=8, lw=0.8))
    annotate_panel_labels(image_axes)
    return _save(fig, 3, "multiview_perception")


def fig04(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    bank_dir = PROJECT_ROOT / "outputs/paper_v622/candidate_diversity/k64_banks"
    representatives = []
    for path in sorted(bank_dir.glob("task_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for stage in ("grasp", "place"):
                success = np.asarray(data[f"{stage}_success"], dtype=bool)
                selected = int(data[f"{stage}_selected_index"])
                if success.sum() < 2 or not success[selected]:
                    continue
                tcp = np.asarray(data[f"{stage}_tcp"], dtype=np.float32)
                selected_distance = np.linalg.norm(tcp - tcp[selected], axis=-1).mean(axis=-1)
                representatives.append((float(selected_distance.mean()), path, stage))
    representatives.sort(reverse=True, key=lambda item: item[0])
    chosen = representatives[:3]
    figure_columns = max(2, len(chosen))
    fig, axes = plt.subplots(1, figure_columns, figsize=(DOUBLE_COLUMN_IN, 2.45), squeeze=False)
    axes = axes[0]
    if not chosen:
        for axis in axes:
            data_required(axis, "Need complete K=64 candidate banks with at least two valid paths")
    for axis, (_, path, stage) in zip(axes, chosen):
        with np.load(path, allow_pickle=False) as data:
            tcp = np.asarray(data[f"{stage}_tcp"], dtype=np.float32)
            success = np.asarray(data[f"{stage}_success"], dtype=bool)
            selected = int(data[f"{stage}_selected_index"])
            cloud = np.asarray(data["scene_points"], dtype=np.float32)[
                np.asarray(data["scene_point_mask"], dtype=bool)
            ]
            axis.scatter(cloud[:, 0], cloud[:, 2], s=1, color="#C7C7C7", alpha=0.22, rasterized=True)
            for index in np.flatnonzero(~success):
                axis.plot(tcp[index, :, 0], tcp[index, :, 2], color="#B8B8B8", alpha=0.22, lw=0.45)
            valid_indices = np.flatnonzero(success)
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(valid_indices), 2)))
            for color, index in zip(colors, valid_indices):
                axis.plot(
                    tcp[index, :, 0],
                    tcp[index, :, 2],
                    color=color,
                    alpha=0.8,
                    lw=1.9 if index == selected else 0.75,
                )
            axis.scatter(tcp[selected, 0, 0], tcp[selected, 0, 2], marker="o", s=24, color="#111111", zorder=5)
            axis.scatter(tcp[selected, -1, 0], tcp[selected, -1, 2], marker="*", s=48, color="#C44E52", zorder=5)
            axis.set_title(f"Task {int(data['task_index'])}, {stage.title()}\nvalid {success.sum()}/64")
            axis.set_xlabel("TCP x (m)")
            axis.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("TCP z (m)")
    annotate_panel_labels(axes)
    return _save(fig, 4, "candidate_geometry")


def fig05(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    source = (
        PROJECT_ROOT
        / "outputs/supplementary_experiments/figure5/candidate_dump/task_0000.npz"
    )
    if not source.is_file():
        return _missing_figure(
            5,
            "residual_reconstruction",
            ["Aligned raw CFM, minimum-jerk, residual controls, and final path"],
        )
    from src.robot.torch_kinematics import DifferentiableZBotFK

    with np.load(source, allow_pickle=False) as data:
        selected = int(data["grasp_selected_index"])
        raw = np.asarray(data["grasp_raw_cfm_q"], dtype=np.float64)
        reference = np.asarray(data["grasp_minimum_jerk_q"], dtype=np.float64)
        controls = np.asarray(data["grasp_reconstructed_control_q"], dtype=np.float64)
        residual_controls = np.asarray(data["grasp_residual_control_q"], dtype=np.float64)
        final = np.asarray(data["grasp_q"][selected], dtype=np.float64)
        duration = float(data["grasp_duration"][selected])
        cloud = np.asarray(data["scene_points"], dtype=np.float64)[
            np.asarray(data["scene_point_mask"], dtype=bool)
        ]
    fk = DifferentiableZBotFK(PROJECT_ROOT / "configs/kinematic_chain.json").to("cpu")

    def tcp(path: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            position, _ = fk(torch.as_tensor(path, dtype=torch.float32))
        return position.numpy()

    trajectories = {
        "Raw CFM": (raw, "#D55E00"),
        "Minimum-jerk": (reference, "#777777"),
        "Final": (final, "#1769AA"),
    }
    tcp_paths = {name: tcp(path) for name, (path, _) in trajectories.items()}
    control_tcp = tcp(controls)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.3))
    axes[0].scatter(cloud[:, 0], cloud[:, 2], s=1, color="#C7C7C7", alpha=0.18, rasterized=True)
    for name, (_, color) in trajectories.items():
        path = tcp_paths[name]
        axes[0].plot(path[:, 0], path[:, 2], color=color, label=name, lw=1.0 if name != "Final" else 1.8)
    axes[0].scatter(control_tcp[:, 0], control_tcp[:, 2], s=14, facecolor="white", edgecolor="#1769AA", linewidth=0.8, zorder=5)
    axes[0].scatter(tcp_paths["Final"][0, 0], tcp_paths["Final"][0, 2], s=22, marker="o", color="#111111", zorder=6)
    axes[0].scatter(tcp_paths["Final"][-1, 0], tcp_paths["Final"][-1, 2], s=42, marker="*", color="#C44E52", zorder=6)
    axes[0].set_xlabel("TCP x (m)")
    axes[0].set_ylabel("TCP z (m)")
    axes[0].set_title("Geometric reconstruction")
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_aspect("equal", adjustable="datalim")

    target_u = np.linspace(0.0, 1.0, len(final))
    raw_u = np.linspace(0.0, 1.0, len(raw))
    raw_dense = np.stack(
        [np.interp(target_u, raw_u, raw[:, joint]) for joint in range(6)], axis=1
    )
    difference = np.max(np.abs(raw_dense - reference), axis=0)
    representative = np.argsort(difference)[-2:][::-1]
    for joint, style in zip(representative, ("-", "--")):
        for name, (path, color) in trajectories.items():
            u = np.linspace(0.0, 1.0, len(path))
            axes[1].plot(u, path[:, joint], linestyle=style, color=color, alpha=0.95, label=f"{name}, q{joint + 1}")
        axes[1].scatter(
            np.linspace(0.0, 1.0, len(controls)),
            controls[:, joint],
            s=11,
            facecolor="white",
            edgecolor="#1769AA",
            linewidth=0.7,
            zorder=5,
        )
    axes[1].set_xlabel("Normalized path coordinate")
    axes[1].set_ylabel("Joint position (rad)")
    axes[1].set_title("Representative joints and 12 controls")
    axes[1].legend(frameon=False, fontsize=5.7, ncol=2, loc="best")

    def dynamic_rms(path: np.ndarray) -> tuple[float, float]:
        time = np.linspace(0.0, duration, len(path))
        velocity = np.gradient(np.unwrap(path, axis=0, period=2 * np.pi), time, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
        jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
        return float(np.sqrt(np.mean(acceleration**2))), float(np.sqrt(np.mean(jerk**2)))

    dynamic = {name: dynamic_rms(path) for name, (path, _) in trajectories.items()}
    normalizer = np.asarray(dynamic["Minimum-jerk"])
    x = np.arange(2)
    width = 0.24
    for index, (name, (_, color)) in enumerate(trajectories.items()):
        values = np.asarray(dynamic[name]) / np.maximum(normalizer, 1.0e-12)
        axes[2].bar(x + (index - 1) * width, values, width=width, color=color, label=name)
    axes[2].axhline(1.0, color="#555555", linewidth=0.7)
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.5, 1500)
    axes[2].set_xticks(x, ["Acceleration RMS", "Jerk RMS"])
    axes[2].set_ylabel("Ratio to minimum-jerk")
    axes[2].set_title(f"Dynamic reconstruction, duration={duration:.2f} s")
    axes[2].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=6.0,
    )

    cache = []
    for name, (path, _) in trajectories.items():
        tcp_values = tcp_paths[name]
        for index, (q_value, tcp_value) in enumerate(zip(path, tcp_values)):
            cache.append(
                {
                    "source": name,
                    "index": index,
                    "u": index / max(1, len(path) - 1),
                    **{f"q{joint + 1}": float(q_value[joint]) for joint in range(6)},
                    "tcp_x": float(tcp_value[0]),
                    "tcp_y": float(tcp_value[1]),
                    "tcp_z": float(tcp_value[2]),
                }
            )
    for index, (control, residual) in enumerate(zip(controls, residual_controls)):
        cache.append(
            {
                "source": "Residual control",
                "index": index,
                "u": index / max(1, len(controls) - 1),
                **{f"q{joint + 1}": float(control[joint]) for joint in range(6)},
                **{f"residual_q{joint + 1}": float(residual[joint]) for joint in range(6)},
            }
        )
    write_csv(FIGURE_ROOT / "data_cache/fig05_residual_reconstruction.csv", cache)
    annotate_panel_labels(axes)
    return _save(fig, 5, "residual_reconstruction")


def fig06(input_path: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    seed_values = (20260814, 20260815, 20260816)
    seeds = [load_csv_or_json(PROJECT_ROOT / f"outputs/paper_v622/main/full_seed{seed}.json") for seed in seed_values]
    records = seeds[0]["records"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.6), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})
    stage_definitions = (("Grasp", "grasp"), ("Place", "place"), ("Joint", "joint"))
    seed_colors = ("#1769AA", "#E69F00", "#009E73")
    x = np.arange(3)
    width = 0.22
    cache = []
    success_sets = []
    for seed_index, (seed, item, color) in enumerate(zip(seed_values, seeds, seed_colors)):
        rates = []
        lower = []
        upper = []
        success_sets.append({str(record["sample_id"]) for record in item["records"] if record["two_stage_success"]})
        for label, key in stage_definitions:
            if key == "joint":
                count = sum(bool(record["two_stage_success"]) for record in item["records"])
            else:
                count = sum(bool(record[key]["success"]) for record in item["records"])
            total = len(item["records"])
            rate = count / total
            low, high = wilson_ci(count, total)
            rates.append(rate)
            lower.append(rate - low)
            upper.append(high - rate)
            cache.append({"panel": "stage_success", "seed": seed, "stage": label, "successes": count, "tasks": total, "rate": rate, "ci_low": low, "ci_high": high})
        axes[0].bar(x + (seed_index - 1) * width, rates, width=width, color=color, label=str(seed), yerr=np.asarray([lower, upper]), capsize=2, error_kw={"linewidth": 0.7})
    axes[0].set_xticks(x, [label for label, _ in stage_definitions])
    axes[0].set_ylim(0.40, 0.98)
    axes[0].set_ylabel("Trajectory success rate")
    axes[0].set_title("Unseen-layout feasibility (n=49)")
    axes[0].legend(frameon=False, fontsize=5.8, title="Sampling seed")
    axes[0].grid(axis="y", alpha=0.2)

    all_ids = {str(record["sample_id"]) for record in records}
    persistence = [sum(task_id in success_set for success_set in success_sets) for task_id in all_ids]
    persistence_counts = [persistence.count(index) for index in range(4)]
    axes[1].bar(range(4), persistence_counts, color=["#C44E52", "#BBBBBB", "#BBBBBB", "#1769AA"])
    axes[1].set_xticks(range(4), ["0/3", "1/3", "2/3", "3/3"])
    axes[1].set_xlabel("Seeds yielding joint success")
    axes[1].set_ylabel("Tasks")
    axes[1].set_title("30 successes are seed-stable")
    axes[1].grid(axis="y", alpha=0.2)
    for index, count in enumerate(persistence_counts):
        axes[1].text(index, count + 0.6, str(count), ha="center", fontsize=6.3)

    failures = [record for record in records if not record["two_stage_success"]]
    reasons = {"Env.": 0, "Self": 0, "Other": 0}
    for record in failures:
        stages = (record["grasp"], record["place"])
        if any(not stage["environment_ok"] for stage in stages):
            reasons["Env."] += 1
        elif any(not stage["self_clearance_ok"] for stage in stages):
            reasons["Self"] += 1
        else:
            reasons["Other"] += 1
    axes[2].bar(reasons.keys(), reasons.values(), color=["#C44E52", "#8172B2", "#999999"])
    axes[2].set_ylabel("Failed tasks")
    axes[2].set_title("All 19 failures are environmental")
    axes[2].grid(axis="y", alpha=0.2)
    annotate_panel_labels(axes)
    cache.extend({"panel": "seed_persistence", "success_seeds": index, "tasks": count} for index, count in enumerate(persistence_counts))
    cache.extend({"panel": "failure_reason", "reason": reason, "tasks": count} for reason, count in reasons.items())
    write_csv(FIGURE_ROOT / "data_cache/fig06_unseen_task_feasibility.csv", cache)
    return _save(fig, 6, "unseen_task_feasibility")


def fig07(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    ours = load_csv_or_json(PROJECT_ROOT / "outputs/paper_v622/main/full_seed20260814.json")
    baseline = load_csv_or_json(PROJECT_ROOT / "outputs/paper_v622/baselines/minimum_jerk.json")
    by_id = {record["sample_id"]: record for record in baseline["records"]}
    common = [(record, by_id[record["sample_id"]]) for record in ours["records"] if record["two_stage_success"] and by_id[record["sample_id"]]["two_stage_success"]]
    definitions = [
        ("grasp", "acceleration_rms_rad_s2", "Δ Grasp Acc. RMS"),
        ("place", "acceleration_rms_rad_s2", "Δ Place Acc. RMS"),
        ("place", "jerk_rms_rad_s3", "Δ Place Jerk RMS"),
        ("place", "duration_s", "Δ Duration"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.25))
    cache = []
    for axis, (stage, field, title) in zip(axes, definitions):
        values = np.asarray([first[stage][field] - second[stage][field] for first, second in common])
        axis.scatter(np.zeros(len(values)), values, s=10, color="#AAAAAA", alpha=0.75)
        mean, low, high = paired_bootstrap_diff(values, np.zeros_like(values))
        axis.errorbar([0.18], [mean], yerr=[[mean - low], [high - mean]], fmt="o", color=METHOD_STYLES["Ours"]["color"], capsize=3)
        axis.axhline(0, color="#333333", lw=0.8)
        axis.set_xlim(-0.2, 0.4)
        axis.set_xticks([0, 0.18], ["Tasks", "Mean"])
        axis.set_title(f"{title}\ncommon success n={len(values)}")
        cache.extend({"metric": title, "task": index, "paired_difference": float(value)} for index, value in enumerate(values))
    axes[0].set_ylabel("Ours − Minimum-Jerk")
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig07_low_impact_tradeoff.csv", cache)
    return _save(fig, 7, "low_impact_tradeoff")


def fig08(input_path: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    summary = load_csv_or_json(input_path or PROJECT_ROOT / "outputs/paper_v622/inference_sweeps_summary.json")
    candidates = summary["candidate_count_rows"]
    steps = summary["heun_step_rows"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN_IN, 2.35))
    k = [row["candidates"] for row in candidates]
    axes[0].plot(k, [row["grasp_successes"] / row["tasks"] for row in candidates], marker="o", label="Grasp")
    axes[0].plot(k, [row["place_successes"] / row["tasks"] for row in candidates], marker="s", label="Place")
    axes[0].plot(k, [row["two_stage_success_rate"] for row in candidates], marker="^", label="Joint")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(k, [str(value) for value in k])
    axes[0].set_ylim(0, 1)
    axes[0].axvline(16, color="#BBBBBB", lw=1)
    axes[0].text(16, 0.05, "Recommended", rotation=90, ha="right", va="bottom", fontsize=6)
    axes[0].set_xlabel("Candidates K")
    axes[0].set_ylabel("Success rate")
    axes[0].legend(frameon=False)
    runtime_path = FIGURE_ROOT / "data_cache/runtime_breakdown.csv"
    runtime_rows = load_csv_or_json(runtime_path) if runtime_path.is_file() else []
    if runtime_rows:
        runtime_rows.sort(key=lambda row: int(row["candidates"]))
        runtime_k = [int(row["candidates"]) for row in runtime_rows]
        total = np.asarray([float(row["total_planning_ms"]) for row in runtime_rows])
        components = {
            "Encoding": np.asarray(
                [float(row["component_encoding_ms"]) + 2 * float(row["vision_encoding_ms"]) for row in runtime_rows]
            ),
            "CFM": np.asarray([2 * float(row["cfm_sampling_ms"]) for row in runtime_rows]),
            "Postprocess": np.asarray(
                [
                    2
                    * (
                        float(row["terminal_projection_residual_reconstruction_ms"])
                        + float(row["densification_time_scaling_ms"])
                        + float(row["candidate_ranking_ms"])
                    )
                    for row in runtime_rows
                ]
            ),
            "Screening": np.asarray([2 * float(row["collision_screening_ms"]) for row in runtime_rows]),
        }
        accounted = sum(components.values())
        components["Other"] = np.maximum(0.0, total - accounted)
        bottom = np.zeros_like(total)
        colors = ["#8FB9D8", "#E69F00", "#7A5195", "#C44E52", "#B8B8B8"]
        for (label, values), color in zip(components.items(), colors):
            percentage = 100.0 * values / total
            axes[1].bar(range(len(runtime_k)), percentage, bottom=bottom, label=label, color=color)
            bottom += percentage
        for index, value in enumerate(total):
            axes[1].text(index, 101.5, f"{value / 1000:.2f}s", ha="center", va="bottom", fontsize=6)
        axes[1].set_xticks(range(len(runtime_k)), [str(value) for value in runtime_k])
        axes[1].set_ylim(0, 110)
        axes[1].set_xlabel("Candidates K")
        axes[1].set_ylabel("Runtime composition (%)")
        axes[1].legend(frameon=False, fontsize=5.5, ncol=2, loc="lower right")
    else:
        axes[1].plot(k, [1000 * row["elapsed_s"] / row["tasks"] for row in candidates], marker="o", color="#7A5195")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(k, [str(value) for value in k])
        axes[1].set_xlabel("Candidates K")
        axes[1].set_ylabel("Planning time / task (ms)")
    diversity_path = FIGURE_ROOT / "data_cache/candidate_diversity_per_task.csv"
    diversity_rows = load_csv_or_json(diversity_path) if diversity_path.is_file() else []
    if diversity_rows:
        diversity_k = sorted({int(row["candidates"]) for row in diversity_rows})
        diversity_values = []
        for count in diversity_k:
            values = [
                float(row["mean_pairwise_joint_distance_rad"])
                for row in diversity_rows
                if int(row["candidates"]) == count
                and row["mean_pairwise_joint_distance_rad"].lower() != "nan"
            ]
            diversity_values.append(float(np.mean(values)) if values else np.nan)
        axes[2].plot(diversity_k, diversity_values, marker="o", color="#009E73")
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks(diversity_k, [str(value) for value in diversity_k])
        axes[2].set_xlabel("Candidates K")
        axes[2].set_ylabel("Pairwise joint distance (rad)")
    else:
        data_required(axes[2], "Candidate-bank diversity sweep")
    axes[3].plot([row["ode_steps"] for row in steps], [1000 * row["grasp_mean_raw_terminal_error_m"] for row in steps], marker="o", label="Grasp error")
    axes[3].plot([row["ode_steps"] for row in steps], [1000 * row["place_mean_raw_terminal_error_m"] for row in steps], marker="s", label="Place error")
    axes[3].set_xlabel("Heun steps")
    axes[3].set_ylabel("Raw terminal error (mm)")
    axes[3].legend(frameon=False)
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig08_candidate_scaling.csv", candidates + steps)
    return _save(fig, 8, "candidate_scaling")


def fig09(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    sweep_path = FIGURE_ROOT / "data_cache/residual_weight_sweep.csv"
    files = {
        "0": "reference_residual_0.json",
        "0.15": "reference_timing_64.json",
        "1": "reference_residual_1.json",
        "geometry": "geometry_preserving.json",
        "hard 5 s": "reference_hard_5s.json",
    }
    rows = []
    for label, name in files.items():
        path = PROJECT_ROOT / "outputs/paper_v622/postprocess_ablations" / name
        if path.is_file():
            data = load_csv_or_json(path)
            durations = [record["place"]["duration_s"] for record in data.get("records", [])]
            rows.append(
                {
                    "setting": label,
                    "joint_rate": data["two_stage"]["success_rate"],
                    "duration": float(np.mean(durations)) if durations else np.nan,
                }
            )
    extended = load_csv_or_json(PROJECT_ROOT / "outputs/paper_v622/extended_experiments_summary.json")
    loss = extended["loss_ablations_fulltrain_k8"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.2))
    sweep_rows = load_csv_or_json(sweep_path) if sweep_path.is_file() else []
    if sweep_rows:
        weights = [float(row["residual_weight"]) for row in sweep_rows]
        tasks = [int(row["tasks"]) for row in sweep_rows]
        axes[0, 0].plot(
            weights,
            [float(row["joint_success_rate"]) for row in sweep_rows],
            marker="o",
            label="Joint success",
        )
        axes[0, 0].plot(
            weights,
            [float(row["environment_failure_tasks"]) / total for row, total in zip(sweep_rows, tasks)],
            marker="s",
            label="Env. collision",
        )
        axes[0, 0].plot(
            weights,
            [float(row["self_collision_failure_tasks"]) / total for row, total in zip(sweep_rows, tasks)],
            marker="^",
            label="Self collision",
        )
        axes[0, 0].axvline(0.15, color="#777777", linestyle="--", linewidth=0.8)
        axes[0, 0].set_xlabel("Residual weight")
        axes[0, 0].set_ylabel("Task fraction")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title("Continuous residual sweep")
        axes[0, 0].legend(frameon=False, ncol=1)
    elif rows:
        axes[0, 0].plot(range(len(rows)), [row["joint_rate"] for row in rows], marker="o")
        axes[0, 0].set_xticks(range(len(rows)), [row["setting"] for row in rows], rotation=30, ha="right")
        axes[0, 0].set_ylabel("Joint success")
        axes[0, 0].set_title("Existing discrete residual settings")
    else:
        data_required(axes[0, 0], "Residual postprocess evaluations")
    timing_64_path = PROJECT_ROOT / "outputs/paper_v622/postprocess_ablations/reference_timing_64.json"
    timing_256_path = PROJECT_ROOT / "outputs/paper_v622/main/full_seed20260814.json"
    if timing_64_path.is_file() and timing_256_path.is_file():
        timing_reports = [load_csv_or_json(timing_64_path), load_csv_or_json(timing_256_path)]
        rates = [report["two_stage"]["success_rate"] for report in timing_reports]
        errors = []
        for report, rate in zip(timing_reports, rates):
            low, high = wilson_ci(report["two_stage"]["successes"], report["two_stage"]["tasks"])
            errors.append((rate - low, high - rate))
        axes[0, 1].bar(["64", "256"], rates, color=["#8FB9D8", "#1769AA"])
        axes[0, 1].errorbar(
            [0, 1],
            rates,
            yerr=np.asarray(errors).T,
            fmt="none",
            color="#222222",
            capsize=3,
            linewidth=0.8,
        )
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xlabel("Dense checking points")
        axes[0, 1].set_ylabel("Joint success")
        axes[0, 1].set_title("Matched K=64, n=49")
    else:
        data_required(axes[0, 1], "Matched 64 vs 256-point evaluation")
    if rows:
        duration_rows = [row for row in rows if np.isfinite(row["duration"])]
        axes[1, 0].bar([row["setting"] for row in duration_rows], [row["duration"] for row in duration_rows], color="#7A5195")
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].set_ylabel("Place duration (s)")
        axes[1, 0].set_title("Hard clipping vs adaptive scaling")
    labels = list(loss)
    axes[1, 1].bar(range(len(labels)), [loss[key]["joint_rate"] for key in labels], color="#1769AA")
    axes[1, 1].set_xticks(range(len(labels)), [label.replace("no_", "−") for label in labels], rotation=30, ha="right")
    axes[1, 1].set_ylabel("Joint success")
    axes[1, 1].set_title("Loss ablations (full training)")
    annotate_panel_labels(axes)
    write_csv(
        FIGURE_ROOT / "data_cache/fig09_executability_ablation.csv",
        sweep_rows if sweep_rows else rows,
    )
    return _save(fig, 9, "executability_ablation")


def fig10(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    extended = load_csv_or_json(PROJECT_ROOT / "outputs/paper_v622/extended_experiments_summary.json")
    camera = extended["data_camera_ablations_k8"]
    robust = load_csv_or_json(PROJECT_ROOT / "outputs/paper_v622/robustness_summary.json")["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.1))
    names = ["data100", "single_wrist"]
    axes[0, 0].bar(["Five-camera", "Wrist-only"], [1000 * camera[key]["target_center_mean_error_m"] for key in names], color=["#1769AA", "#D55E00"])
    axes[0, 0].set_ylabel("Target center error (mm)")
    axes[0, 0].set_title("Camera configuration")
    fractions = ["data25", "data50", "data100"]
    axes[0, 1].plot([25, 50, 100], [camera[key]["joint_rate"] for key in fractions], marker="o")
    axes[0, 1].set_xlabel("Training data (%)")
    axes[0, 1].set_ylabel("Joint success")
    selected = [row for row in robust if row["name"] in {"clean", "rgb_std_15", "depth_std_0p02", "blackout_wrist", "visual_blackout_all"}]
    axes[1, 0].barh([row["name"].replace("_", " ") for row in selected], [row["two_stage_success_rate"] for row in selected], color="#4C9F70")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_xlabel("Trajectory feasibility")
    axes[1, 0].set_title("Visual perturbations")
    axes[1, 1].scatter([1000 * row["target_center_mean_error_m"] for row in robust], [row["two_stage_success_rate"] for row in robust], s=18, color="#7A5195")
    axes[1, 1].set_xlabel("Perception error (mm)")
    axes[1, 1].set_ylabel("Trajectory feasibility")
    axes[1, 1].set_title("Perception vs feasibility")
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig10_vision_ablation.csv", robust)
    return _save(fig, 10, "vision_ablation")


def fig11(input_path: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    import imageio.v3 as iio

    video = input_path or PROJECT_ROOT / "outputs/nasa_astronaut_no_intersection_demo/media_final/pose_grasp_place_final.mp4"
    if not Path(video).is_file():
        fig, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.0))
        data_required(axis, f"Missing PhysX rollout video: {video}")
        return _save(fig, 11, "physx_sequence")
    frames = list(iio.imiter(video))
    labels = ["Approach", "Grasp", "Contact", "Lift", "Transfer", "Place", "Release"]
    indices = np.linspace(35, max(36, len(frames) - 35), len(labels)).astype(int)
    fig, axes = plt.subplots(1, len(labels), figsize=(DOUBLE_COLUMN_IN, 1.55))
    for axis, label, index in zip(axes, labels, indices):
        image = frames[min(index, len(frames) - 1)]
        crop = image[int(0.04 * len(image)) :]
        axis.imshow(crop)
        axis.set_axis_off()
        axis.set_title(label, pad=2)
    return _save(fig, 11, "physx_sequence")


def _missing_figure(number: int, slug: str, messages: list[str]) -> list[str]:
    configure_paper_rcparams()
    fig, axes = plt.subplots(1, len(messages), figsize=(DOUBLE_COLUMN_IN, 2.25))
    axes = np.atleast_1d(axes)
    for axis, message in zip(axes, messages):
        data_required(axis, message)
    annotate_panel_labels(axes)
    return _save(fig, number, slug)


def fig12(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    summary_path = PROJECT_ROOT / "outputs/supplementary_experiments/experiment_a/experiment_a_summary.json"
    if not summary_path.is_file():
        return _missing_figure(12, "planner_pareto", ["Experiment A task-level planner logs"])
    report = load_csv_or_json(summary_path)
    methods = ["Visual CFM (Ours)", "RRT-Connect", "STOMP"]
    short_names = ["Visual CFM", "RRT-Connect", "STOMP"]
    colors = ["#1769AA", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.15))
    cache_rows = []
    stages = ("Grasp", "Place", "Joint")
    x = np.arange(len(methods))
    width = 0.23
    for stage_index, stage in enumerate(stages):
        rates, lower, upper = [], [], []
        for method in methods:
            item = report["methods"][method][f"success_{stage.lower()}"]
            rate = float(item["rate"])
            low, high = item["wilson_95"]
            rates.append(rate)
            lower.append(rate - low)
            upper.append(high - rate)
            cache_rows.append({"panel": "success", "method": method, "stage": stage, "rate": rate, "ci_low": low, "ci_high": high, "n": 220})
        axes[0, 0].bar(x + (stage_index - 1) * width, rates, width=width, color=STAGE_COLORS[stage], label=stage, yerr=np.asarray([lower, upper]), capsize=2, error_kw={"linewidth": 0.7})
    axes[0, 0].set_xticks(x, short_names)
    axes[0, 0].set_ylim(0.76, 1.01)
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].set_title("Classical planners are more feasible (n=220)")
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[0, 0].grid(axis="y", alpha=0.2)

    planning = [report["methods"][method]["successful_task_metrics"]["planning_time_ms"]["median"] for method in methods]
    axes[0, 1].bar(x, planning, color=colors)
    axes[0, 1].set_xticks(x, short_names)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Median time per stage (ms, log)")
    axes[0, 1].set_title("Classical planners are faster")
    axes[0, 1].grid(axis="y", alpha=0.2)
    for method, value in zip(methods, planning):
        cache_rows.append({"panel": "planning_time", "method": method, "median_ms": value, "n": 220})

    comparison_colors = {"RRT-Connect": "#E69F00", "STOMP": "#009E73"}
    metric_panels = (
        (axes[1, 0], "acceleration_rms_rad_s2", "Acceleration RMS difference", "Ours - baseline (rad/s$^2$)"),
        (axes[1, 1], "jerk_rms_rad_s3", "Jerk RMS difference", "Ours - baseline (rad/s$^3$)"),
    )
    for axis, metric, title, xlabel in metric_panels:
        for position, (baseline_name, comparison) in enumerate(report["comparisons"].items()):
            item = comparison["paired_bootstrap_ours_minus_baseline"][metric]
            mean = item["mean_difference"]
            low, high = item["ci95"]
            axis.errorbar(mean, position, xerr=[[mean - low], [high - mean]], fmt="o", color=comparison_colors[baseline_name], capsize=3)
            cache_rows.append({"panel": metric, "baseline": baseline_name, "mean_difference": mean, "ci_low": low, "ci_high": high, "n": comparison["common_success_tasks"]})
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(range(2), list(report["comparisons"]))
        axis.set_xlabel(xlabel)
        axis.set_title(f"{title}\nnegative favors lower-impact Ours")
        axis.grid(axis="x", alpha=0.2)
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig12_planner_comparison.csv", cache_rows)
    return _save(fig, 12, "planner_pareto")


def fig13(_: Path | None = None) -> list[str]:
    return _missing_figure(
        13,
        "carrying_stability",
        [
            "Experiment B invalid: nominal native-contact task fails before closure",
            "No valid transfer-phase relative orientation population",
            "No valid paired contact-force population",
            "0/100 required randomized rollouts passed the nominal-task gate",
        ],
    )


def fig14(_: Path | None = None) -> list[str]:
    configure_paper_rcparams()
    summary_path = PROJECT_ROOT / "outputs/supplementary_experiments/experiment_c/difficulty_failure_summary.json"
    if not summary_path.is_file():
        return _missing_figure(14, "failure_modes", ["Experiment C difficulty summary"])
    summary = load_csv_or_json(summary_path)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.15))

    def success_bars(axis: plt.Axes, rows: list[dict], title: str) -> None:
        rows = [row for row in rows if row["n"] > 0]
        rates = np.asarray([row["success_rate"] for row in rows], dtype=float)
        low = np.asarray([row["wilson_95"][0] for row in rows], dtype=float)
        high = np.asarray([row["wilson_95"][1] for row in rows], dtype=float)
        positions = np.arange(len(rows))
        axis.bar(positions, rates, color="#1769AA")
        axis.errorbar(
            positions,
            rates,
            yerr=np.maximum(0.0, np.vstack((rates - low, high - rates))),
            fmt="none",
            color="#222222",
            capsize=3,
            linewidth=0.8,
        )
        for position, rate, row in zip(positions, rates, rows):
            axis.text(position, min(1.03, rate + 0.055), f"n={row['n']}", ha="center", fontsize=6)
        axis.set_xticks(positions, [row["bin"] for row in rows])
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Joint success")
        axis.set_title(title)

    success_bars(axes[0, 0], summary["clearance_bins"], "Direct-reference clearance")
    success_bars(axes[0, 1], summary["obstacle_bins"], "Obstacle count")
    axes[0, 1].set_xlabel("Non-target objects")

    failure_counts = summary["failure_counts"]
    names = list(failure_counts) or ["NONE"]
    counts = [failure_counts.get(name, 0) for name in names]
    axes[1, 0].barh(range(len(names)), counts, color="#C44E52")
    axes[1, 0].set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    axes[1, 0].set_xlabel("Failed tasks")
    axes[1, 0].set_title("Primary failure reason")

    representative_ids = [
        task_id
        for task_ids in summary["representative_task_ids"].values()
        for task_id in task_ids
    ][:4]
    rgb_by_id = {}
    data_dir = PROJECT_ROOT / "data/supplementary_unseen_v622_pose/left"
    for path in sorted(data_dir.glob("test_shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for index, sample_id in enumerate(data["sample_id"]):
                if str(sample_id) in representative_ids:
                    image = np.asarray(data["rgb"][index, 0])
                    while image.ndim > 3:
                        image = image[0]
                    rgb_by_id[str(sample_id)] = image
    axes[1, 1].set_axis_off()
    axes[1, 1].set_title("Representative failed tasks (initial wrist RGB)", pad=8)
    locations = ((0.00, 0.52, 0.48, 0.42), (0.52, 0.52, 0.48, 0.42), (0.00, 0.02, 0.48, 0.42), (0.52, 0.02, 0.48, 0.42))
    for task_id, location in zip(representative_ids, locations):
        inset = axes[1, 1].inset_axes(location)
        image = rgb_by_id.get(task_id)
        if image is None:
            data_required(inset, task_id)
            continue
        if image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        inset.imshow(np.clip(image, 0, 255).astype(np.uint8))
        inset.set_axis_off()
        label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0}
        inset.text(
            0.02,
            0.98,
            "ENV COLLISION",
            transform=inset.transAxes,
            va="top",
            fontsize=4.8,
            color="#9B2C2C",
            bbox=label_box,
        )
        inset.text(
            0.02,
            0.02,
            task_id,
            transform=inset.transAxes,
            va="bottom",
            fontsize=4.8,
            bbox=label_box,
        )
    annotate_panel_labels(axes)
    return _save(fig, 14, "failure_modes")


def fig15(_: Path | None = None) -> list[str]:
    """Learning-baseline overview replacing the manuscript result table."""
    configure_paper_rcparams()
    summary_path = PROJECT_ROOT / "outputs/paper_v622/extended_experiments_summary.json"
    minimum_jerk_path = PROJECT_ROOT / "outputs/paper_v622/baselines/minimum_jerk.json"
    if not summary_path.is_file() or not minimum_jerk_path.is_file():
        return _missing_figure(15, "learning_baselines", ["learning baseline summaries"])
    summary = load_csv_or_json(summary_path)["model_baselines_k16"]
    minimum_jerk = load_csv_or_json(minimum_jerk_path)
    methods = [
        ("Minimum-jerk", None),
        ("Condition CFM", "condition_cfm"),
        ("Matched CFM", "condition_cfm_capacity_matched"),
        ("Conditional DDIM", "condition_ddim"),
        ("Visual CFM", "visual_cfm"),
    ]
    colors = ["#D55E00", "#5B8FF9", "#61A0A8", "#CC79A7", "#1769AA"]
    stages = ("Grasp", "Place", "Joint")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.55), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})
    cache_rows = []
    x = np.arange(len(methods))
    width = 0.23
    for stage_index, stage in enumerate(stages):
        rates = []
        lower = []
        upper = []
        for label, key in methods:
            if key is None:
                item = minimum_jerk["two_stage" if stage == "Joint" else stage.lower()]
                successes = int(item["successes"])
                tasks = int(item["tasks"])
            else:
                item = summary[key]
                successes = int(item[f"{stage.lower() if stage != 'Joint' else 'joint'}_successes"])
                tasks = int(item["tasks"])
            rate = successes / tasks
            low, high = wilson_ci(successes, tasks)
            rates.append(rate)
            lower.append(rate - low)
            upper.append(high - rate)
            cache_rows.append({"panel": "success", "method": label, "stage": stage, "successes": successes, "tasks": tasks, "rate": rate, "ci_low": low, "ci_high": high})
        axes[0].bar(
            x + (stage_index - 1) * width,
            rates,
            width=width,
            color=STAGE_COLORS[stage],
            label=stage,
            yerr=np.asarray([lower, upper]),
            capsize=2,
            error_kw={"linewidth": 0.7},
        )
    short_labels = ["Min.-jerk", "Cond.\nCFM", "Matched\nCFM", "Cond.\nDDIM", "Visual\nCFM"]
    axes[0].set_xticks(x, short_labels, rotation=0)
    axes[0].tick_params(axis="x", labelsize=5.6)
    axes[0].set_ylim(0.30, 0.92)
    axes[0].set_ylabel("Trajectory success rate")
    axes[0].set_title("Paired unseen layouts (n=49)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.2)

    learned = methods[1:]
    learned_x = np.arange(len(learned))
    raw_grasp = [1000 * float(summary[key]["grasp_raw_error_mean_m"]) for _, key in learned]
    raw_place = [1000 * float(summary[key]["place_raw_error_mean_m"]) for _, key in learned]
    axes[1].bar(learned_x - 0.18, raw_grasp, width=0.36, color=STAGE_COLORS["Grasp"], label="Grasp")
    axes[1].bar(learned_x + 0.18, raw_place, width=0.36, color=STAGE_COLORS["Place"], label="Place")
    axes[1].set_yscale("log")
    axes[1].set_xticks(learned_x, ["Cond.\nCFM", "Matched\nCFM", "Cond.\nDDIM", "Visual\nCFM"])
    axes[1].tick_params(axis="x", labelsize=5.5)
    axes[1].set_ylabel("Raw terminal error (mm, log)")
    axes[1].set_title("Before terminal projection")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    for (label, key), grasp, place in zip(learned, raw_grasp, raw_place):
        cache_rows.extend([
            {"panel": "raw_terminal_error", "method": label, "stage": "Grasp", "value_mm": grasp},
            {"panel": "raw_terminal_error", "method": label, "stage": "Place", "value_mm": place},
        ])

    elapsed = [float(minimum_jerk["elapsed_s"])] + [float(summary[key]["elapsed_s"]) for _, key in learned]
    axes[2].barh(x, elapsed, color=colors)
    axes[2].set_yticks(x, [label for label, _ in methods])
    axes[2].set_xscale("log")
    axes[2].set_xlabel("49-task evaluation time (s, log)")
    axes[2].set_title("End-to-end evaluation cost")
    axes[2].grid(axis="x", alpha=0.2)
    for (label, _), value in zip(methods, elapsed):
        cache_rows.append({"panel": "elapsed", "method": label, "value_s": value, "tasks": 49})
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig15_learning_baselines.csv", cache_rows)
    return _save(fig, 15, "learning_baselines")


def fig16(_: Path | None = None) -> list[str]:
    """Continuous residual-weight response across success, failures and geometry."""
    configure_paper_rcparams()
    source = FIGURE_ROOT / "data_cache/residual_weight_sweep.csv"
    if not source.is_file():
        return _missing_figure(16, "residual_landscape", ["continuous residual-weight sweep"])
    import csv

    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    weights = np.asarray([float(row["residual_weight"]) for row in rows])
    tasks = np.asarray([int(row["tasks"]) for row in rows])
    successes = np.asarray([int(row["joint_successes"]) for row in rows])
    rates = successes / tasks
    cis = np.asarray([wilson_ci(int(s), int(n)) for s, n in zip(successes, tasks)])
    env_fail = np.asarray([int(row["environment_failure_tasks"]) for row in rows])
    self_fail = np.asarray([int(row["self_collision_failure_tasks"]) for row in rows])
    path = np.asarray([float(row["mean_path_length_rad_common_success"]) for row in rows])
    acceleration = np.asarray([float(row["mean_acc_rms_common_success"]) for row in rows])
    jerk = np.asarray([float(row["mean_jerk_rms_common_success"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 4.1))
    axes[0, 0].errorbar(weights, rates, yerr=np.vstack((rates - cis[:, 0], cis[:, 1] - rates)), color="#1769AA", marker="o", capsize=3)
    axes[0, 0].axvspan(0.15, 0.30, color="#E4F3E8", alpha=0.8, label="stable interval")
    axes[0, 0].set_ylabel("Joint success rate")
    axes[0, 0].set_title("Feasibility response (n=49)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].bar(weights, env_fail, width=0.045, color="#C44E52", label="Environment")
    axes[0, 1].bar(weights, self_fail, width=0.045, bottom=env_fail, color="#7A5195", label="Self")
    axes[0, 1].set_ylabel("Failed tasks")
    axes[0, 1].set_title("Collision failures")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.2)
    axes[1, 0].plot(weights, path, color="#009E73", marker="s")
    axes[1, 0].set_xlabel("Residual weight")
    axes[1, 0].set_ylabel("Mean joint path length (rad)")
    axes[1, 0].set_title("Geometry retained by the residual")
    axes[1, 0].grid(alpha=0.2)
    scatter = axes[1, 1].scatter(acceleration, jerk, c=weights, cmap="viridis", s=42, edgecolor="white", linewidth=0.4)
    for w, x_value, y_value in zip(weights, acceleration, jerk):
        axes[1, 1].annotate(f"{w:g}", (x_value, y_value), xytext=(3, 2), textcoords="offset points", fontsize=5.3)
    axes[1, 1].set_xlabel("Acceleration RMS (rad/s$^2$)")
    axes[1, 1].set_ylabel("Jerk RMS (rad/s$^3$)")
    axes[1, 1].set_title("Common-success dynamics")
    axes[1, 1].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[1, 1], label="Residual weight", fraction=0.05, pad=0.03)
    display_ticks = np.asarray([0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    display_labels = ["0", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"]
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(display_ticks, display_labels)
        axis.tick_params(axis="x", labelsize=5.8)
    annotate_panel_labels(axes)
    return _save(fig, 16, "residual_landscape")


def fig17(_: Path | None = None) -> list[str]:
    """Absolute and normalized inference-time decomposition."""
    configure_paper_rcparams()
    source = FIGURE_ROOT / "data_cache/runtime_breakdown.csv"
    if not source.is_file():
        return _missing_figure(17, "runtime_breakdown", ["synchronized runtime breakdown"])
    import csv

    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    candidates = np.asarray([int(row["candidates"]) for row in rows])
    categories = [
        ("Encoding", ("component_encoding_ms", "vision_encoding_ms"), "#7A5195"),
        ("CFM sampling", ("cfm_sampling_ms",), "#1769AA"),
        ("Reconstruction", ("terminal_projection_residual_reconstruction_ms", "densification_time_scaling_ms"), "#E69F00"),
        ("Collision screening", ("collision_screening_ms",), "#C44E52"),
        ("Ranking", ("candidate_ranking_ms",), "#009E73"),
    ]
    values = np.asarray([
        [2.0 * sum(float(row[key]) for key in keys) for label, keys, color in categories]
        for row in rows
    ])
    total = np.asarray([float(row["total_planning_ms"]) for row in rows])
    unaccounted = np.maximum(total - values.sum(axis=1), 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 3.0))
    x = np.arange(len(candidates))
    bottom = np.zeros(len(candidates))
    cache_rows = []
    for index, (label, _, color) in enumerate(categories):
        axes[0].bar(x, values[:, index], bottom=bottom, color=color, label=label)
        bottom += values[:, index]
        for k, value in zip(candidates, values[:, index]):
            cache_rows.append({"candidates": int(k), "component": label, "two_stage_ms": float(value), "share": float(value / total[candidates == k][0])})
    axes[0].bar(x, unaccounted, bottom=bottom, color="#BBBBBB", label="Other")
    axes[0].plot(x, total, color="#222222", marker="o", linewidth=1.0, label="Measured total")
    axes[0].set_xticks(x, candidates)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Candidates K")
    axes[0].set_ylabel("Two-stage planning time (ms, log)")
    axes[0].set_title("Absolute inference cost")
    axes[0].grid(axis="y", alpha=0.2)
    normalized = values / total[:, None]
    bottom = np.zeros(len(candidates))
    for index, (label, _, color) in enumerate(categories):
        axes[1].bar(x, 100 * normalized[:, index], bottom=100 * bottom, color=color, label=label)
        bottom += normalized[:, index]
    axes[1].bar(x, 100 * unaccounted / total, bottom=100 * bottom, color="#BBBBBB", label="Other")
    axes[1].set_xticks(x, candidates)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Candidates K")
    axes[1].set_ylabel("Share of total planning time (%)")
    axes[1].set_title("Compute bottleneck")
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=5.6,
    )
    axes[1].grid(axis="y", alpha=0.2)
    annotate_panel_labels(axes)
    write_csv(FIGURE_ROOT / "data_cache/fig17_runtime_breakdown.csv", cache_rows)
    return _save(fig, 17, "runtime_breakdown")


BUILDERS: dict[int, Any] = {
    1: fig01,
    2: fig02,
    3: fig03,
    4: fig04,
    5: fig05,
    6: fig06,
    7: fig07,
    8: fig08,
    9: fig09,
    10: fig10,
    11: fig11,
    12: fig12,
    13: fig13,
    14: fig14,
    15: fig15,
    16: fig16,
    17: fig17,
}
