> Historical source documentation. See the repository root README and PUBLICATION_NOTES.md for current distribution limits.

# v6.2.1 视觉引导轨迹生成与抓取放置工程

本工程研究航天员辅助机械臂的视觉引导多候选轨迹生成、低冲击后处理和 PhysX 抓取放置验证。全部项目文件均位于本目录，不写入 `D:\Isaac_lab\custom_tasks`，也不修改 `D:\Isaac_lab` 和 `D:\conda_envs\env_isaaclab`。

## 工程入口

- 导师汇报：[研究工作汇报_视觉几何融合多候选轨迹生成与抓取放置.md](研究工作汇报_视觉几何融合多候选轨迹生成与抓取放置.md)
- 论文初稿：`docs/v621论文_条件流匹配轨迹生成方法_初稿.md`
- 数据契约：`docs/DATA_CONTRACT.md`
- 最终位姿方法配置：`configs/visual_cfm_v622_pose.yaml`
- 训练脚本：`scripts/train_visual_cfm_v621.py`
- 评估脚本：`scripts/evaluate_visual_cfm_v621.py`
- 模型：`outputs/checkpoints/`
- 主实验：`outputs/paper_v622/`
- 补充实验：`outputs/supplementary_experiments/`
- 论文图片：`paper_figures/manuscript/png/`

## 核心场景

- `1111_copy.usd`：基础机器人与辅助机械臂场景。
- `1111_copy_native_contact.usd`：原生接触抓取放置场景。
- `1111_nasa_astronaut_aux_arm_20tasks.usd`：20 组航天员外观抓取任务源场景。
- `1111_nasa_astronaut_no_intersection_demo.usd`：无穿模论文展示场景。
- `1111_nasa_astronaut_waist_mounted_aux_arm.usd`：腰部安装式辅助机械臂最终展示场景。
- `supplementary_experiment_scene.usd`：补充实验复现场景。

## 常用命令

```powershell
# 单元测试
D:\conda_envs\env_isaaclab\python.exe -m pytest tests -q

# 最终位姿 Visual CFM 训练
D:\conda_envs\env_isaaclab\python.exe scripts\train_visual_cfm_v621.py --data-dir data\visual_v622_pose --flow-config configs\visual_cfm_v622_pose.yaml --init-checkpoint outputs\checkpoints\visual_cfm_v621_left.pt --output outputs\checkpoints\visual_cfm_v622_pose.pt --report outputs\paper_v622\training\visual_cfm_v622_stagewise_training.json --save-stage-checkpoints

# 未见任务评估（K=16，32 步 Heun，256 点时间参数化）
D:\conda_envs\env_isaaclab\python.exe scripts\evaluate_visual_cfm_v621.py --data_dir data\visual_v622_pose --checkpoint outputs\checkpoints\visual_cfm_v622_pose.pt --config configs\visual_cfm_v622_pose.yaml --output outputs\evaluation_current.json --candidates 16 --ode_steps 32 --postprocess --timing_points 256 --unclipped_duration --independent_stage_generators

# 重新生成腰部安装式展示场景
D:\conda_envs\env_isaaclab\python.exe scripts\configure_nasa_astronaut_waist_mounted_scene.py
```

## 使用边界

当前模型与场景仅用于 Isaac Sim/Isaac Lab 仿真研究。航天员网格是无骨骼、无碰撞的外观组件；实机部署前必须重新完成相机标定、动作限幅、急停、碰撞力阈值、通信延迟和 sim-to-real 风险验证。
