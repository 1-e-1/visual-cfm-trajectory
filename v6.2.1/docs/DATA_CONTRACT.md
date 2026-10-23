# 1111 场景视觉轨迹数据契约

## 输入

- `condition: float32[9]`：`q_start[6] + p_goal_in_arm_base[3]`。
- `rgb: uint8[4,2,128,128,3]`：视角顺序为腕部、Cube 侧面、同侧胸部、头顶。
- `depth: float16[4,2,128,128,1]`：`distance_to_image_plane`，单位为 m。
- `semantic: int16[4,2,128,128,1]`：仅用于目标物体 mask；缺失时置零，不读取 H1 状态。

左右机械臂均使用各自 `foot_0` 坐标系，因此共享同一 Visual CFM。训练和部署均不读取 H1 的关节位置、速度或基座状态。

## 监督标签

- `q: float32[64,6]`：低冲击关节位置序列。
- `duration: float32[]`：轨迹总时长，范围 1.0–5.0 s。
- `q_goal: float32[6]`：仅作为监督和审计字段，不进入模型输入。
- `collision_free: bool[]`：由视觉点云几何先验筛选并由 PhysX 离线复核。

## 在线输出与安全链

Visual CFM 采用 32 步 Heun ODE 生成 64 个离散关节位置，并独立回归 duration。推理生成 64 条候选，先检查机械臂自碰撞，再用四视角 RGB-D 重建点云检查 H1、另一机械臂、桌面与非目标物体间隙。在线部署不使用 PhysX；PhysX 只用于仿真训练标签和离线评估。
