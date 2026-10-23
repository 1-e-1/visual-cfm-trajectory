"""Build model-internal Visual CFM and joint-training demonstration figures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper_figures" / "conceptual"
COLORS = {
    "paper": "#F7F9FB",
    "panel": "#FFFFFF",
    "ink": "#24313C",
    "muted": "#687985",
    "grid": "#D9E0E5",
    "blue": "#2478B5",
    "cyan": "#168F91",
    "green": "#2E8B57",
    "amber": "#D08316",
    "red": "#C04B48",
    "violet": "#7356A5",
}
JOINT_COLORS = ["#2478B5", "#D08316", "#2E8B57", "#C04B48", "#7356A5", "#168F91"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
        }
    )


def style_axis(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color("#BCC7CE")
        spine.set_linewidth(1.2)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65, alpha=0.75)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.set_title(title, loc="left", fontsize=12.5, color=COLORS["ink"], pad=12)
    if subtitle:
        ax.text(
            1.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=COLORS["muted"],
        )


def panel_background(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    x, y, width, height = bounds
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=1.1,
            edgecolor="#C7D0D6",
            facecolor=COLORS["panel"],
            zorder=-2,
        )
    )


def chip(fig: plt.Figure, x: float, y: float, text: str, color: str) -> None:
    fig.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.12",
            "facecolor": mpl.colors.to_rgba(color, 0.12),
            "edgecolor": color,
            "linewidth": 1.1,
        },
    )


def fig_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    connectionstyle: str = "arc3,rad=0",
    width: float = 1.5,
) -> None:
    fig.patches.append(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            zorder=8,
        )
    )


def expert_joint_trajectory() -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, 64)
    blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    starts = np.array([-0.55, 0.35, -0.25, 0.65, -0.4, 0.15])
    goals = np.array([0.85, -0.7, 0.55, -0.25, 0.7, -0.55])
    q = starts[None, :] + blend[:, None] * (goals - starts)[None, :]
    q += 0.08 * np.sin(np.pi * s)[:, None] * np.array([1.0, -0.6, 0.8, -1.0, 0.5, -0.7])[None]
    return s, q


def noisy_joint_trajectory() -> np.ndarray:
    s, q = expert_joint_trajectory()
    rng = np.random.default_rng(20260816)
    noise = rng.normal(0.0, 0.48, q.shape)
    noise = 0.5 * noise + 0.3 * np.roll(noise, 1, axis=0) + 0.2 * np.roll(noise, 2, axis=0)
    noisy = 0.25 * q + noise
    noisy[0] = q[0]
    return noisy


def flow_progress(phase: int | None, count: int) -> float:
    if phase is None:
        return 1.0
    return np.linspace(0.0, 1.0, count)[phase]


def flow_embedding(progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(27)
    initial = rng.uniform([-2.8, -1.75], [2.0, 1.75], size=(12, 2))
    target_x = np.linspace(-2.55, 2.55, len(initial))
    target = np.column_stack((target_x, 0.72 * np.sin(1.18 * target_x)))
    bend = np.column_stack((0.22 * np.sin(np.arange(len(initial))), 0.5 * np.cos(np.arange(len(initial)))))
    current = (1.0 - progress) * initial + progress * target + progress * (1.0 - progress) * bend
    history = []
    for beta in np.linspace(0.0, progress, 36):
        history.append((1.0 - beta) * initial + beta * target + beta * (1.0 - beta) * bend)
    return initial, target, current, np.asarray(history)


def draw_visual_cfm(phase: int | None = None) -> plt.Figure:
    progress = flow_progress(phase, 8)
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    fig.text(0.045, 0.945, "Visual CFM 条件速度场运算演示", fontsize=23, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.955,
        0.95,
        "仅展示模型内部：条件编码、速度场预测与 Heun 积分",
        fontsize=10,
        color=COLORS["muted"],
        ha="right",
    )
    chip(fig, 0.20, 0.875, r"状态条件  $c=[q_{start},q_{pose\_IK},p_{goal}]$", COLORS["blue"])
    chip(fig, 0.50, 0.875, r"视觉与几何条件  $z_{env}$", COLORS["cyan"])
    chip(fig, 0.78, 0.875, r"速度场  $v_\theta(x_t,t,c,z_{env})$", COLORS["violet"])
    fig_arrow(fig, (0.31, 0.875), (0.40, 0.875), COLORS["muted"])
    fig_arrow(fig, (0.61, 0.875), (0.68, 0.875), COLORS["muted"])

    field_bounds = (0.05, 0.13, 0.58, 0.68)
    panel_background(fig, field_bounds)
    field_ax = fig.add_axes((0.065, 0.16, 0.55, 0.61))
    style_axis(field_ax, "轨迹状态在条件速度场中的连续输运", "二维示意投影，不是机械臂工作空间坐标")
    field_ax.set_xlim(-3.15, 3.15)
    field_ax.set_ylim(-2.15, 2.15)
    field_ax.set_xlabel("轨迹状态投影维度 1", fontsize=9, color=COLORS["muted"])
    field_ax.set_ylabel("轨迹状态投影维度 2", fontsize=9, color=COLORS["muted"])
    gx, gy = np.meshgrid(np.linspace(-2.9, 2.9, 17), np.linspace(-1.9, 1.9, 13))
    target_y = 0.72 * np.sin(1.18 * gx)
    vx = 0.32 * (2.65 - gx)
    vy = 1.15 * (target_y - gy) + 0.18 * np.cos(1.18 * gx)
    magnitude = np.sqrt(vx**2 + vy**2)
    field_ax.quiver(
        gx,
        gy,
        vx / (magnitude + 1.0e-6),
        vy / (magnitude + 1.0e-6),
        magnitude,
        cmap="YlGnBu",
        alpha=0.62,
        width=0.0032,
        scale=24,
        zorder=1,
    )
    curve_x = np.linspace(-2.9, 2.9, 300)
    curve_y = 0.72 * np.sin(1.18 * curve_x)
    field_ax.plot(curve_x, curve_y, color=COLORS["green"], linewidth=3.0, label="专家轨迹分布", zorder=3)
    initial, target, current, history = flow_embedding(progress)
    for index in range(len(initial)):
        field_ax.plot(
            history[:, index, 0],
            history[:, index, 1],
            color=COLORS["violet"],
            linewidth=1.2,
            alpha=0.42,
            zorder=2,
        )
    field_ax.scatter(initial[:, 0], initial[:, 1], s=28, marker="x", color=COLORS["muted"], alpha=0.55, label=r"初始噪声 $x_0$")
    field_ax.scatter(current[:, 0], current[:, 1], s=58, color=COLORS["violet"], edgecolor="white", linewidth=0.8, label=r"当前状态 $x_t$", zorder=5)
    if progress < 0.999:
        velocity = target - current
        field_ax.quiver(
            current[:, 0],
            current[:, 1],
            velocity[:, 0],
            velocity[:, 1],
            color=COLORS["red"],
            angles="xy",
            scale_units="xy",
            scale=4.2,
            width=0.004,
            alpha=0.9,
            zorder=4,
            label=r"$v_\theta$",
        )
    field_ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95, ncol=2)
    field_ax.text(
        0.02,
        0.97,
        rf"流时间  $t={progress:.2f}$",
        transform=field_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["violet"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": COLORS["violet"]},
    )

    joint_bounds = (0.66, 0.47, 0.30, 0.34)
    panel_background(fig, joint_bounds)
    joint_ax = fig.add_axes((0.685, 0.515, 0.255, 0.245))
    style_axis(joint_ax, r"同一 $x_t\in\mathbb{R}^{64\times6}$ 展开为六条关节曲线", "虚线为专家轨迹")
    s, expert = expert_joint_trajectory()
    noisy = noisy_joint_trajectory()
    current_q = (1.0 - progress) * noisy + progress * expert
    current_q[0] = expert[0]
    for joint in range(6):
        joint_ax.plot(s, expert[:, joint], color=JOINT_COLORS[joint], linestyle="--", linewidth=1.0, alpha=0.35)
        joint_ax.plot(s, current_q[:, joint], color=JOINT_COLORS[joint], linewidth=1.7, label=rf"$q_{joint + 1}$")
    joint_ax.set_xlim(0, 1)
    joint_ax.set_xlabel("64 个轨迹点的归一化索引", fontsize=8.5)
    joint_ax.set_ylabel("关节角 / rad", fontsize=8.5)
    joint_ax.legend(ncol=3, fontsize=7.5, loc="upper right", framealpha=0.9)

    heun_bounds = (0.66, 0.13, 0.30, 0.29)
    panel_background(fig, heun_bounds)
    heun_ax = fig.add_axes((0.69, 0.17, 0.24, 0.20))
    heun_ax.set_xlim(0, 1)
    heun_ax.set_ylim(0, 1)
    heun_ax.axis("off")
    heun_ax.set_title("一个 Heun ODE 步中的两次速度场计算", loc="left", fontsize=12.5, color=COLORS["ink"], pad=8)
    points = {"current": (0.12, 0.25), "predict": (0.58, 0.72), "correct": (0.82, 0.43)}
    heun_ax.scatter(*points["current"], s=80, color=COLORS["blue"], zorder=4)
    heun_ax.scatter(*points["predict"], s=80, color=COLORS["amber"], zorder=4)
    heun_ax.scatter(*points["correct"], s=80, color=COLORS["green"], zorder=4)
    heun_ax.add_patch(FancyArrowPatch(points["current"], points["predict"], arrowstyle="-|>", mutation_scale=14, linewidth=2, color=COLORS["amber"]))
    heun_ax.add_patch(FancyArrowPatch(points["current"], points["correct"], arrowstyle="-|>", mutation_scale=14, linewidth=2.4, color=COLORS["green"]))
    heun_ax.plot([points["predict"][0], points["correct"][0]], [points["predict"][1], points["correct"][1]], linestyle=":", color=COLORS["muted"], linewidth=1.2)
    heun_ax.text(0.06, 0.10, r"$x_t$", fontsize=10, color=COLORS["blue"], fontweight="bold")
    heun_ax.text(0.50, 0.80, r"Euler 预测  $\tilde{x}$", fontsize=9, color=COLORS["amber"])
    heun_ax.text(0.73, 0.31, r"校正后  $x_{t+h}$", fontsize=9, color=COLORS["green"])
    heun_ax.text(0.03, 0.96, r"$\tilde{x}=x_t+h\,v_\theta(x_t,t)$", fontsize=9.5, va="top", color=COLORS["ink"])
    heun_ax.text(0.03, 0.84, r"$x_{t+h}=wrap[x_t+\frac{h}{2}(v_\theta+\tilde v_\theta)]$", fontsize=9.5, va="top", color=COLORS["ink"])

    fig.text(
        0.5,
        0.065,
        r"32 个 Heun ODE 步将噪声轨迹状态连续输运到数据分布；64 表示每条轨迹包含的关节状态点数。模型同时由独立 Duration Head 输出 $\hat T$。",
        ha="center",
        fontsize=10.5,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF0F5", "edgecolor": "#9DAAB3"},
    )
    if phase is not None:
        fig.text(0.955, 0.885, f"动画：ODE 进度 {phase + 1}/8", ha="right", fontsize=9.5, color=COLORS["violet"])
    return fig


def rotate(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return matrix @ vector


def draw_pose_frame(
    ax: plt.Axes,
    origin: np.ndarray,
    angle: float,
    color: str,
    label: str,
    label_offset_y: float,
) -> None:
    length = 0.44
    x_axis = length * np.array([np.cos(angle), np.sin(angle)])
    y_axis = length * np.array([-np.sin(angle), np.cos(angle)])
    ax.arrow(*origin, *x_axis, width=0.012, head_width=0.10, color=color, length_includes_head=True, zorder=4)
    ax.arrow(*origin, *y_axis, width=0.012, head_width=0.10, color=mpl.colors.to_rgba(color, 0.65), length_includes_head=True, zorder=4)
    ax.scatter(*origin, s=55, color=color, edgecolor="white", linewidth=0.8, zorder=5)
    ax.text(
        origin[0],
        origin[1] + label_offset_y,
        label,
        color=color,
        fontsize=9,
        fontweight="bold",
        ha="center",
    )


def draw_joint_training(phase: int | None = None) -> plt.Figure:
    progress = 0.92 if phase is None else flow_progress(phase, 7)
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    fig.text(0.045, 0.945, "面向抓取位姿与低冲击运动的联合训练演示", fontsize=23, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.955,
        0.95,
        "仅展示训练内部：流匹配样本、模型预测、联合损失与反向传播",
        fontsize=10,
        color=COLORS["muted"],
        ha="right",
    )
    chip(fig, 0.18, 0.875, r"采样  $x_0,\ t$", COLORS["blue"])
    chip(fig, 0.39, 0.875, r"构造  $x_t=(1-t)x_0+t x_1$", COLORS["cyan"])
    chip(fig, 0.62, 0.875, r"预测  $v_\theta(x_t,t,c,z_{env})$", COLORS["violet"])
    chip(fig, 0.84, 0.875, r"重构  $\hat x_1=x_t+(1-t)v_\theta$", COLORS["green"])
    fig_arrow(fig, (0.25, 0.875), (0.30, 0.875), COLORS["muted"])
    fig_arrow(fig, (0.48, 0.875), (0.53, 0.875), COLORS["muted"])
    fig_arrow(fig, (0.72, 0.875), (0.75, 0.875), COLORS["muted"])

    vector_bounds = (0.05, 0.49, 0.40, 0.32)
    panel_background(fig, vector_bounds)
    vector_ax = fig.add_axes((0.075, 0.535, 0.35, 0.225))
    style_axis(vector_ax, "Flow Matching 监督：预测速度对齐目标速度", "箭头越重合，流场损失越小")
    vector_ax.set_xlim(-2.3, 2.65)
    vector_ax.set_ylim(-1.65, 1.8)
    x0 = np.array([-1.75, -1.05])
    x1 = np.array([1.90, 1.15])
    t = 0.56
    xt = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted_velocity = rotate(target_velocity, 42.0 * (1.0 - progress)) * (0.62 + 0.38 * progress)
    vector_ax.plot([x0[0], x1[0]], [x0[1], x1[1]], linestyle="--", color=COLORS["muted"], linewidth=1.2, alpha=0.65)
    vector_ax.scatter(*x0, s=75, color=COLORS["blue"], label=r"噪声 $x_0$", zorder=5)
    vector_ax.scatter(*xt, s=85, color=COLORS["cyan"], label=r"插值状态 $x_t$", zorder=5)
    vector_ax.scatter(*x1, s=75, color=COLORS["green"], label=r"专家数据 $x_1$", zorder=5)
    scale = 0.34
    vector_ax.arrow(*xt, *(scale * target_velocity), width=0.022, head_width=0.16, color=COLORS["green"], length_includes_head=True, label="target")
    vector_ax.arrow(*xt, *(scale * predicted_velocity), width=0.018, head_width=0.15, color=COLORS["violet"], length_includes_head=True, label="predicted")
    vector_ax.text(*(xt + scale * target_velocity + np.array([0.0, 0.12])), r"目标速度 $u_t=x_1-x_0$", fontsize=8.7, color=COLORS["green"], ha="center")
    vector_ax.text(*(xt + scale * predicted_velocity + np.array([0.0, -0.20])), r"预测速度 $v_\theta$", fontsize=8.7, color=COLORS["violet"], ha="center")
    vector_ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.95)
    flow_error = np.mean((predicted_velocity - target_velocity) ** 2)
    vector_ax.text(0.02, 0.96, rf"$L_{{flow}}={flow_error:.3f}$", transform=vector_ax.transAxes, va="top", fontsize=10.5, color=COLORS["violet"], fontweight="bold")

    joint_bounds = (0.48, 0.49, 0.48, 0.32)
    panel_background(fig, joint_bounds)
    joint_ax = fig.add_axes((0.505, 0.535, 0.43, 0.225))
    style_axis(joint_ax, "模型重构的 64×6 轨迹与专家轨迹", "实线：模型重构；虚线：专家标签")
    s, expert = expert_joint_trajectory()
    noisy = noisy_joint_trajectory()
    terminal_bias = (1.0 - progress) * np.array([0.20, -0.16, 0.18, -0.12, 0.14, -0.18])
    reconstructed = expert + (1.0 - progress) * 0.72 * (noisy - expert) + s[:, None] * terminal_bias[None]
    reconstructed[0] = expert[0]
    for joint in range(6):
        joint_ax.plot(s, expert[:, joint], color=JOINT_COLORS[joint], linestyle="--", linewidth=1.0, alpha=0.35)
        joint_ax.plot(s, reconstructed[:, joint], color=JOINT_COLORS[joint], linewidth=1.6, label=rf"$q_{joint + 1}$")
    joint_ax.set_xlim(0, 1)
    joint_ax.set_xlabel("轨迹点归一化索引", fontsize=8.5)
    joint_ax.set_ylabel("关节角 / rad", fontsize=8.5)
    joint_ax.legend(ncol=6, fontsize=7.3, loc="upper center", framealpha=0.92)

    pose_bounds = (0.05, 0.12, 0.25, 0.29)
    panel_background(fig, pose_bounds)
    pose_ax = fig.add_axes((0.075, 0.165, 0.20, 0.19))
    style_axis(pose_ax, "抓取末端位姿损失", r"$L_{pose}=L_{pos}+L_{ori}$")
    pose_ax.set_xlim(-0.2, 2.2)
    pose_ax.set_ylim(-0.15, 1.8)
    pose_ax.set_aspect("equal", adjustable="box")
    target_origin = np.array([1.55, 1.02])
    predicted_origin = target_origin + (1.0 - progress) * np.array([-0.85, -0.55])
    target_angle = np.deg2rad(35.0)
    predicted_angle = target_angle + np.deg2rad(58.0 * (1.0 - progress))
    draw_pose_frame(pose_ax, target_origin, target_angle, COLORS["green"], "目标位姿", 0.20)
    draw_pose_frame(pose_ax, predicted_origin, predicted_angle, COLORS["violet"], "预测位姿", -0.26)
    pose_ax.plot([predicted_origin[0], target_origin[0]], [predicted_origin[1], target_origin[1]], linestyle=":", color=COLORS["red"], linewidth=1.8)
    pose_error = np.linalg.norm(target_origin - predicted_origin)
    orientation_error = abs(target_angle - predicted_angle)
    pose_ax.text(0.02, 0.03, f"位置误差 {pose_error:.3f}\n姿态误差 {np.rad2deg(orientation_error):.1f}°", transform=pose_ax.transAxes, fontsize=8.5, color=COLORS["red"], va="bottom")

    impact_bounds = (0.33, 0.12, 0.36, 0.29)
    panel_background(fig, impact_bounds)
    impact_ax = fig.add_axes((0.355, 0.165, 0.31, 0.19))
    style_axis(impact_ax, "低冲击正则项", "由预测 duration 确定 Δt，并对关节限制归一化")
    dt = 1.0 / 63.0
    velocity = np.diff(reconstructed, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    baseline_velocity = np.diff(noisy, axis=0) / dt
    baseline_acceleration = np.diff(baseline_velocity, axis=0) / dt
    baseline_jerk = np.diff(baseline_acceleration, axis=0) / dt
    v_mag = np.linalg.norm(velocity, axis=1) / max(
        np.linalg.norm(baseline_velocity, axis=1).max(), 1.0e-6
    )
    a_mag = np.linalg.norm(acceleration, axis=1) / max(
        np.linalg.norm(baseline_acceleration, axis=1).max(), 1.0e-6
    )
    j_mag = np.linalg.norm(jerk, axis=1) / max(
        np.linalg.norm(baseline_jerk, axis=1).max(), 1.0e-6
    )
    impact_ax.plot(np.linspace(0, 1, len(v_mag)), v_mag, color=COLORS["blue"], linewidth=1.6, label=r"速度 $L_v$")
    impact_ax.plot(np.linspace(0, 1, len(a_mag)), a_mag, color=COLORS["amber"], linewidth=1.6, label=r"加速度 $L_a$")
    impact_ax.plot(np.linspace(0, 1, len(j_mag)), j_mag, color=COLORS["red"], linewidth=1.6, label=r"jerk $L_j$")
    impact_ax.set_xlim(0, 1)
    impact_ax.set_ylim(0, 1.08)
    impact_ax.set_xlabel("轨迹归一化时间", fontsize=8.5)
    impact_ax.set_ylabel("归一化幅值", fontsize=8.5)
    impact_ax.legend(loc="upper right", fontsize=7.8, ncol=3, framealpha=0.92)

    loss_bounds = (0.72, 0.12, 0.24, 0.29)
    panel_background(fig, loss_bounds)
    loss_ax = fig.add_axes((0.75, 0.165, 0.19, 0.19))
    style_axis(loss_ax, "联合损失与反向传播", "柱高为示意归一化值")
    losses = np.array(
        [
            min(flow_error / 3.0, 1.0),
            min(pose_error / 1.1 + orientation_error / 2.0, 1.0),
            min(float(v_mag.mean() + a_mag.mean() + j_mag.mean()), 1.0),
        ]
    )
    loss_ax.bar([0, 1, 2], losses, color=[COLORS["violet"], COLORS["green"], COLORS["amber"]], width=0.62)
    loss_ax.set_xticks([0, 1, 2], [r"$L_{flow}$", r"$L_{pose}$", r"$L_{impact}$"])
    loss_ax.set_ylim(0, 1.05)
    loss_ax.set_ylabel("损失", fontsize=8.5)
    loss_ax.text(0.5, 0.96, r"$L=10L_{flow}+5L_{pose}+0.02(L_v+L_a)+0.01L_j+\cdots$", transform=loss_ax.transAxes, ha="center", va="top", fontsize=8.2, color=COLORS["ink"])

    fig.lines.append(
        mpl.lines.Line2D(
            [0.84, 0.972, 0.972],
            [0.41, 0.45, 0.835],
            transform=fig.transFigure,
            color=COLORS["red"],
            linewidth=1.7,
            zorder=8,
        )
    )
    fig_arrow(fig, (0.972, 0.835), (0.70, 0.835), COLORS["red"], width=1.7)
    fig.text(
        0.978,
        0.62,
        "总损失反向传播",
        fontsize=9.2,
        color=COLORS["red"],
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        0.5,
        0.055,
        "训练过程中，速度场逐渐对齐目标速度；重构轨迹的抓取位姿误差和速度、加速度、jerk 同时下降。图中曲线用于解释机制，不作为实验统计结果。",
        ha="center",
        fontsize=10.5,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF0F5", "edgecolor": "#9DAAB3"},
    )
    if phase is not None:
        fig.text(0.955, 0.885, f"动画：训练收敛进度 {phase + 1}/7", ha="right", fontsize=9.5, color=COLORS["violet"])
    return fig


def figure_to_image(fig: plt.Figure, dpi: int = 120) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_static(draw_function, stem: str) -> None:
    fig = draw_function(None)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_animation(draw_function, stem: str, phases: int) -> None:
    frames = [figure_to_image(draw_function(phase)) for phase in range(phases)]
    durations = [820] * phases
    durations[-1] = 1800
    frames[0].save(
        OUTPUT_DIR / f"{stem}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_static(draw_visual_cfm, "visual_cfm_computation_process")
    save_animation(draw_visual_cfm, "visual_cfm_computation_process", phases=8)
    save_static(draw_joint_training, "grasp_pose_low_impact_joint_training")
    save_animation(draw_joint_training, "grasp_pose_low_impact_joint_training", phases=7)
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(f"{path.relative_to(ROOT)}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
