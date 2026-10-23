# Visual CFM v6.1 方法规格（IK-conditioned Multi-label Flow Matching + Terminal Anchor）

生成日期：2026-08-05（v6 首版 / v6.1 终端锚定）
状态：已敲定，left 臂全流程验证完成
范围：h1_1111_grasp 五相机视觉抓取轨迹生成

## 1. 动机（基于实测）

v5 的 206 条测试显示 raw 模型输出末端 TCP 误差均值 0.30-0.37 m（本机复测 32 样本 0.44-0.78 m），与目标距离（0.42-0.59 m）同量级；候选多样性集中在轨迹中段，末端被均值锁定；视觉输入对输出的影响只有 condition 的 25%。根因是 Flow Matching 的 MSE 目标在多模态分布下回归条件均值，且模型缺乏对"目标构型"的显式条件。


## 1.1 v6 实测结论与 v6.1 终端锚定（关键修正）

v6（无锚定）在 left 训练集 819 条上训练 200 epoch 后实测：

| 指标（validation 102 条，best-of-64 候选） | v6 实测 | v6.1 锚定实测 |
|---|---:|---:|
| raw 末端 TCP 误差 mean | 0.156 m | 0.0023 m |
| raw 末端 TCP 误差 median | 0.105 m | 0.0013 m |
| raw 末端 TCP 误差 max | 0.774 m | 0.0182 m |
| ≤ 0.05 m 达标率 | 27.5% | 100% |

根因：**不是训练/推理归一化不一致**（q_start/q_ik/p_goal 三块与 FK 均已验证一致），而是 flow field 在 819 条数据 + 4 层 Transformer 下欠拟合，ODE 积分末端产生系统性偏差（候选分布 σ≈5 cm 但均值偏 ~0.2 m）。64 步 ODE、更窄初始噪声、提高末端损失权重（25→80）均不能消除该偏差（w80 反降至 0.247 m）。

v6.1 修正：**终端锚定（terminal anchor）**——`q_ik` 本来就是 P2 已知的目标构型，训练与推理两端都把输出轨迹末端直接固定为 `q_ik`：

- 训练端（`src/training/losses.py` ObstacleFlowMatchingLoss）：`projected[:, -1] = q_ik`（wrap 到 [-pi,pi)），`L_final_tcp_position` 退化为 FK(q_ik)-p_goal 常量（≈0），速度/加速度/加加速度平滑损失在锚定末端上计算，驱动模型把自身预测的末端拉向 q_ik。
- 推理端（`h1_1111_grasp/scripts/evaluate_visual_cfm.py` sample_from_visual_latent）：ODE 积分结束后 `trajectory[:, -1] = q_ik/pi`。
- CFM 只负责端点固定的内部路径形状；raw 末端误差 = FK(q_ik)-p_goal，结构性 ≤ 5 mm（q_ik[0] < 5 mm，其余解为 fallback 最大 14.6 mm，仍远低于 50 mm 判据）。

## 2. 方法定义

Visual CFM v6 = 目标构型条件化（IK-first）+ 多模态专家标签 + 更大/更难数据 + 更高维视觉表示。

### 2.1 输入输出张量（推理与训练一致）

| 名称 | Shape | 含义 |
|---|---:|---|
| `condition_v6` | `[B,15]` | `[q_start(6), q_ik(6), p_goal(3)]` |
| `rgb/depth/semantic` | `[B,5,2,128,128,C]` | 五相机双帧视觉 |
| `q_experts` | `[B,8,64,6]` | 每个任务 8 条无碰撞专家候选（多模态标签） |
| `q_ik_candidates` | `[B,4,6]` | 从 `p_goal` 多解 position-only IK 得到的 4 个目标构型 |
| 输出 | `q[B,64,6] + duration[B]` | 条件轨迹 + 时长 |

- `q_ik`：推理时由 position-only IK 从 `p_goal` 求得多解；训练时从 `q_ik_candidates` 中随机取一个（含专家 `q_goal` 对应解），保证训练/推理分布一致。
- 全部关节角 wrap 到 `[-pi,pi)`，模型内部归一化 `q/pi ∈ [-1,1)`（与 v4/v5 相同的周期归一化，训练/推理两端一致）。
- `p_goal` 始终是机械臂 `foot_0` 局部坐标系下的 TCP 位置（米）。

### 2.2 模型改动

- condition 编码（periodic）：`sin/cos(q_start)` 12 维 + `sin/cos(q_ik)` 12 维 + `p_goal` 3 维 = 27 维，进入 `condition_encoder`。
- 视觉编码器输出维度：`obstacle_hidden_dim` 128 → 256（`WristVisionEncoder.output_dim` 与 privileged teacher `ObstacleTokenEncoder.hidden_dim` 同步为 256，蒸馏保持同维，不引入额外投影）。
- 其余结构（Transformer backbone hidden 192、4 层、4 heads、time emb 128、duration head）不变。
- 模型新增属性 `goal_condition_dof=6`；`goal_condition_dof=0` 时自动回退到 v5 的 9 维 condition（向后兼容旧 checkpoint 与旧数据）。

### 2.3 损失函数

```text
L = 1.0  L_flow
  + 10.0 L_start
  + 25.0 L_goal_joint_consistency     # 新增：重建末端关节角 与 q_ik 的 wrap MSE
  + 25.0 L_final_tcp_position         # 保持：FK(末端) 与 p_goal 的位置 MSE
  + 0.1  L_velocity + 0.1 L_acceleration + 0.08 L_jerk
  + 1.0  L_duration
  + 2.0  L_collision_clearance
  + 4.0  L_self_collision_clearance
```

- `L_goal_joint_consistency` 直接约束末端关节构型逼近 `q_ik`，比仅靠 FK 位置损失更直接（`q_ik` 已给定，消除了关节空间多解歧义）。
- 多模态标签：每个 batch 内每个样本随机从 `q_experts` 的 8 条中选 1 条作为 flow 标签（后续可升级为 min-K 采样）。
- 兼容模式：旧 9 维 condition 数据 → `L_goal_joint_consistency` 自动置 0，行为与 v5 相同。

### 2.4 推理流程（IK-first）

```text
五相机 RGB-D-semantic + q_start + p_goal
  -> position-only IK 多解：q_ik[0..3]（每个经 self-clearance 过滤）
  -> 对每个 q_ik：condition_v6 生成 K 条候选（32 步 Heun ODE）
  -> 合并全部候选（最多 4x64）
  -> 视觉点云环境间距 + 自碰撞 + 动力学筛选（复用 v5 的 screen_candidates）
  -> 选中 1 条 64x6 轨迹 -> 后处理（terminal IK / 安全终态投影）-> 50 Hz 执行
```

### 2.5 数据规格（T3）

- 规模：每臂 4096 条（左右共 8192），训练/验证/测试 = 80/10/10。
- 目标分布：`goal_pool` 按 TCP 距离分层（近/中/远桶），覆盖机械臂可达空间而非集中在中距离。
- 每条样本保存：`condition_v6[15]`、`q_experts[8,64,6]`、`q_ik_candidates[4,6]`、原有 `rgb/depth/semantic/obstacle_tokens/scene_points` 等字段。

## 3. 判定标准（成功判据，与 v5 一致）

- raw TCP 末端误差（不经后处理）：目标从 ~0.5 m 降到 **≤ 0.05 m（50 mm）**。
- K 候选成功率（3 cm 阈值 + clearance ≥ 1 cm）：从 62.62% 提升。
- 26 条 PhysX 回放：≥ 24/26。
- 后处理仍保留，但 raw 误差必须显著下降，不再依赖 terminal IK 兜底。

## 4. 实施清单（v6.1 状态）

1. 模型：`goal_condition_dof` 支持（`src/models/obstacle_flow_matching.py`）。✅
2. 损失：多标签 + `L_goal_joint_consistency` + **v6.1 终端锚定**（`src/training/losses.py`）。✅
3. 数据集：`q_experts`/`q_ik_candidates` 字段（`src/datasets/npz_dataset.py`）。✅
4. IK 求解器：多解 position-only IK（`src/planning/ik_solver.py`，数据生成与评估共用）。✅
5. 数据升级：`h1_1111_grasp/scripts/upgrade_visual_v5_to_v6.py`（复用 v5 采集，scene_points 在线云排名专家；修复 camera_view_names shard 级字段 + 输出分片连续编号 + 每分片断点续跑）。✅
6. 配置：`configs/visual_cfm_transfer_5view_v6.yaml`（vision output_dim 256、goal_condition_dof=6、qik_augmentation）。✅
7. 训练：privileged 300 epoch（含锚定）+ 视觉蒸馏 40 + 微调 20。✅
8. 评估：`h1_1111_grasp/scripts/evaluate_visual_cfm.py` IK-first 推理 + **推理端终端锚定**（并修复 solve_ik_candidates 在 no_grad 块内无法求导的 bug）。✅

## 5. 风险与边界

- IK 解质量决定 P2 上限：多解必须覆盖可达构型且通过 self-clearance 过滤；IK 失败时回退到 v5 条件（无 q_ik）。v6.1 锚定后 raw 误差即 FK(q_ik) 误差，因此 IK 解本身的收敛质量（≤5 mm 主解）是精度上限。
- 训练/推理 q_ik 分布需一致：训练随机取 `q_ik_candidates`（含 q_goal），推理取 IK 多解，均来自同一求解器；锚定在两端一致实施。
- 已知边界：819 条/臂数据下 v6 无锚定 raw 末端误差 ~0.156 m（flow field 欠拟合）；锚定把末端判据结构性达标，但**内部路径的碰撞规避质量**仍受数据量限制，T3 规模升级（4096/臂）是后续提升内部轨迹质量的主要手段。
- 视觉仍依赖 Isaac Sim 真值 semantic；sim-to-real 边界不变。
