# 无PPO目标条件视觉抓取轨迹生成研究流程

## 1. 研究题目与目标

建议的新研究题目为：

> 面向航天员辅助机械臂的目标条件多视角视觉低冲击避障抓取轨迹生成方法

研究目标是在目标物体三维坐标由任务系统直接给出的条件下，利用五相机视觉理解目标周围障碍物、机器人本体和另一机械臂的空间分布，通过 Visual Conditional Flow Matching（Visual CFM）生成多条关节轨迹，最终完成安全、平滑、低冲击的抓取与抬升。

新方法完全移除PPO，不再包含强化学习轨迹修正器。旧PPO实验只作为方法选择依据和失败分析保留，不进入新方法训练、推理或性能统计。

## 2. 方法边界

### 2.1 当前研究负责的内容

1. 五相机 RGB-D-semantic/instance环境观测；
2. 根据给定目标坐标关联图像中的目标物体；
3. 区分目标物体与其余障碍物；
4. Visual CFM多候选关节轨迹生成；
5. 机械臂自碰撞、机械臂与机器人互碰撞及环境碰撞复筛；
6. 64到1024点轨迹重采样和未截断时间缩放；
7. 夹爪接触白名单与抓取状态机；
8. 抓取、抬升、保持及PhysX验证。

### 2.2 当前不研究的内容

1. 不使用PPO或其他强化学习轨迹细化器；
2. 不通过视觉估计目标物体世界坐标，目标坐标由外部任务系统直接提供；
3. 暂不预测任意六自由度抓取姿态，第一阶段使用固定夹爪方向；
4. 暂不研究手指级灵巧操作；
5. 不将视觉点云检查表述为真实硬件接触安全保证。

## 3. 总体技术路线

```text
给定目标物体坐标 p_object_in_foot0
五相机 RGB-D-semantic/instance + 当前关节角 q_start
        ↓
目标坐标投影到五个相机
        ↓
多视角目标实例关联
        ↓
target mask + obstacle mask
        ↓
目标条件视觉编码器
        ↓
z_target + z_obstacle
        ↓
计算预抓取目标 p_pregrasp
        ↓
Visual CFM以32步Heun生成K=64条64×6候选轨迹
        ↓
position-only terminal IK
        ↓
64→1024 periodic-aware cubic spline
        ↓
未截断时间缩放
        ↓
视觉环境复筛 + 自碰撞复筛 + 机器人互碰撞复筛
        ↓
选择安全、平滑、低冲击候选
        ↓
50 Hz执行至预抓取点
        ↓
低速接近目标物体
        ↓
闭合夹爪并验证夹持
        ↓
给定抬升目标 p_lift
        ↓
再次运行同一Visual CFM规划流程
        ↓
抬升、保持、平衡与PhysX验证
```

## 4. 坐标系与输入输出

### 4.1 坐标系

所有目标位置统一表示在对应辅助机械臂的 `foot_0` 坐标系中：

```text
p_object_in_foot0: [B,3]
```

相机外参由机械臂FK和固定安装变换计算：

```text
p_camera_i = T_camera_i_from_foot0(q_start) × p_object_in_foot0
```

其中腕部相机外参随 `q_start` 变化，四个固定相机使用固定安装变换。

### 4.2 Observation

```text
q_start:       float32[B,6]
p_object:      float32[B,3]
RGB:           uint8[B,5,2,128,128,3]
Depth:         float16/float32[B,5,2,128,128,1]
Semantic:      int32[B,5,2,128,128,1]
Instance ID:   int32[B,5,2,128,128,1]
camera pose:   float32[B,5,7]
intrinsics:    float32[B,5,3,3]
```

视觉的两帧维度在静态实验中可由相同画面复制；研究动态障碍物时必须来自不同仿真时刻。

### 4.3 模型输出

Visual CFM原始输出：

```text
q_candidates:    float32[B,64,64,6]
duration_raw:    float32[B,64]
```

最终输出：

```text
q_final:                    float32[B,1024,6]
duration:                   float32[B]
planning_success:           bool[B]
terminal_error_m:           float32[B]
environment_clearance_m:    float32[B]
self_clearance_m:           float32[B]
robot_mutual_clearance_m:   float32[B]
```

最后一维 `6` 始终表示六个机械臂关节角。

### 4.4 Action与控制频率

轨迹生成器的输出是关节位置序列，不是关节速度或力矩：

```text
action_control = q_target[6]
```

- Isaac Sim physics timestep：`0.005 s`，200 Hz；
- 机械臂控制周期：`0.020 s`，50 Hz；
- 1024是路径表示点数，不是控制频率；
- 执行器按最终duration对1024点路径进行时间插值。

## 5. Step N1：目标坐标投影

### 输入

```text
p_object_in_foot0 [B,3]
camera extrinsics [B,5,4,4]
camera intrinsics [B,5,3,3]
```

### 处理

对每个相机计算：

```text
p_camera = T_camera_from_foot0 × p_object
[u,v,1]^T ∝ K × p_camera
```

输出归一化投影：

```text
goal_projection [B,5,3]
```

最后一维为 `[u/W,v/H,z_camera]`。目标在相机后方、图像外或被遮挡时，设置对应 `valid_view=false`。

### 目的

显式建立三维目标坐标与二维视觉特征之间的几何对应，避免要求网络自行学习坐标系变换。

## 6. Step N2：多视角目标实例关联

### 输入

```text
goal_projection
RGB-D-semantic/instance
```

### 处理

1. 在目标投影像素附近建立局部窗口；
2. 由Depth反投影得到局部点云；
3. 选择与给定三维目标坐标最近的点云簇；
4. 使用instance ID扩展到完整物体mask；
5. 融合五个相机结果并计算置信度；
6. 所有视角均不可见或关联冲突时返回目标关联失败。

### 输出

```text
target_mask:       bool[B,5,128,128]
obstacle_mask:     bool[B,5,128,128]
target_confidence: float32[B]
```

### 目的

目标物体不需要视觉定位，但必须在视觉环境中被识别为“允许最终接触的目标”，其余物体作为障碍物。

## 7. Step N3：目标条件视觉编码器

### 网络结构

建议在原五相机视觉编码器上加入目标条件adapter：

```text
多视角图像特征 F_visual
目标投影编码 E_goal
        ↓ cross-attention / gated fusion
target features + obstacle features
        ↓
z_target[128] + z_obstacle[128]
```

第一阶段冻结：

```text
Visual CFM trajectory backbone
Duration Head
原始Visual CFM数据集
```

训练：

```text
目标条件adapter
视觉融合层
必要时解冻视觉backbone最后一个stage
```

### Loss

```text
L_vision = lambda_mask L_target_mask
         + lambda_instance L_target_instance
         + lambda_contrast L_target_obstacle_contrast
         + lambda_latent L_latent_distillation
```

- `L_target_mask`：BCE + Dice Loss；
- `L_target_instance`：目标实例关联损失；
- `L_target_obstacle_contrast`：目标与障碍物表示分离；
- `L_latent_distillation`：保持原视觉避障表示，减少灾难性遗忘。

本方法没有PPO，因此没有Reward设计。

## 8. Step N4：Visual CFM多候选生成

### 条件输入

```text
condition = q_start[6]
          + p_goal[3]
          + z_target[128]
          + z_obstacle[128]
```

### 轨迹状态与推理

```text
x_t: [B,64,6]
ODE solver: Heun
ODE steps: 32
candidates: 64
```

32表示流匹配ODE数值积分次数，64表示每条轨迹的路径点数。

### 输出

```text
q_raw_candidates [B,64,64,6]
duration_raw [B,64]
```

### 目的

利用条件流匹配的多模态生成能力覆盖目标周围不同绕行方向，避免只生成单一路径。

## 9. Step N5：终态修正与1024点时间缩放

### 处理

1. 对候选末端执行position-only terminal IK；
2. 将终端修正平滑分布到64点轨迹；
3. 强制先将64点轨迹插值为1024点；
4. 在1024点样条上计算一至三阶导数；
5. 根据速度、加速度和Jerk soft limits反算duration；
6. 不执行 `clip(...,5.0)`；
7. `duration>5 s`只记录名义超限；
8. `duration>30 s`或非有限值时标记无效。

### 输出

```text
q_dense_candidates [B,64,1024,6]
duration_candidates [B,64]
duration_valid [B,64]
```

## 10. Step N6：目标感知多约束候选复筛

### 分阶段碰撞规则

预抓取阶段：

```text
目标物体作为普通障碍物
所有机械臂连杆保持clearance
```

最终接近阶段：

```text
只允许夹爪指端与目标接触区域接触
手腕和机械臂其他连杆仍禁止接触目标
```

抓取后：

```text
目标物体作为附着payload
检查payload与机器人、机械臂和环境碰撞
```

### 候选成功判据

```text
terminal_error <= 0.030 m
environment_clearance >= 0.010 m
self_clearance >= 0.010 m
robot_mutual_clearance >= 0.010 m
velocity/acceleration/Jerk <= soft limits
duration_valid = true
```

### 排序

安全候选优先，再按以下指标排序：

```text
terminal error
minimum clearance
acceleration RMS
Jerk RMS
duration
```

若当前候选验证失败，依次检查下一条候选；64条均失败时返回规划失败，禁止执行。

## 11. Step N7：安全终态投影与最终验证

### 安全终态投影

终点构型自碰撞时，使用 Neural C-SDF提供搜索方向，并由exact capsule oracle验证最终构型。投影后必须重新执行：

```text
64→1024插值
未截断时间缩放
完整碰撞验证
```

### 最终验证器

取消PPO后，原adaptive safety gate改为单路径最终安全验证器：

```text
adaptive exact self-collision validation
视觉点云环境clearance
机械臂与机器人互碰撞检查
TCP终点误差
动力学soft limits
duration有效性
```

最终验证失败时不能执行该轨迹，只能尝试下一候选或返回规划失败。

## 12. Step N8：抓取状态机

```text
IDLE
  ↓ 收到目标坐标
TARGET_ASSOCIATION
  ↓ 目标关联成功
PLAN_PREGRASP
  ↓ 轨迹验证成功
EXECUTE_PREGRASP
  ↓ 到达
FINAL_APPROACH
  ↓ 指端进入接触区域
CLOSE_GRIPPER
  ↓ 夹持验证成功
PLAN_LIFT
  ↓ 抬升轨迹验证成功
EXECUTE_LIFT
  ↓
HOLD
```

任何阶段发生以下情况均进入 `EMERGENCY_STOP`：

```text
非白名单碰撞
接触冲量超限
关节跟踪误差超限
H1高度、倾角或漂移超限
目标关联置信度过低
轨迹规划失败
```

## 13. 抓取成功判据

单次任务只有同时满足以下条件才算成功：

```text
到达预抓取点
夹爪指端与目标建立允许接触
夹爪闭合后目标未滑落
目标被抬升至少0.10 m
保持时间不少于2.0 s
机械臂无自碰撞
机械臂与机器人无互碰撞
无非白名单环境接触
H1未跌倒且漂移未超限
速度、加速度和Jerk满足soft limits
```

## 14. 数据集设计

### 14.1 场景随机化

- 目标物体位置、尺寸、形状、质量、摩擦系数和纹理；
- 静态障碍物位置、大小和形状；
- 动态小型障碍物速度、方向和出现时刻；
- 左右辅助机械臂任务；
- `q_start`在 `[-pi,pi]^6` 均匀采样后做端点安全筛选；
- 目标必须位于可达空间且存在安全抓取路径。

### 14.2 视觉随机化

- 光照强度和方向；
- RGB噪声、Depth噪声和缺失像素；
- 相机外参小扰动；
- 目标部分遮挡；
- 目标与障碍物外观相似；
- 腕部相机因初始关节角产生的视角变化。

### 14.3 数据标签

```text
target instance mask
obstacle mask
p_object_in_foot0
q_start
expert q_trajectory[64,6]
duration
self/environment/mutual clearance
grasp contact region
grasp success
```

训练集、验证集和测试集必须按场景ID与目标物体实例划分，禁止同一场景或同一轨迹同时进入不同split。

## 15. 训练流程

### Stage T1：目标关联预训练

冻结CFM，只训练目标投影adapter和目标mask分支。

### Stage T2：目标条件视觉表示微调

冻结CFM和Duration Head，训练视觉融合层，使 `z_target` 与 `z_obstacle`稳定分离。

### Stage T3：轨迹条件对齐

保持CFM主体冻结，使用较小学习率解冻视觉编码器最后一个stage，优化轨迹Flow Matching Loss与终点、碰撞和平滑损失。

### Stage T4：仿真抓取验证

不再更新模型，只进行Isaac Sim/PhysX闭环执行和固定测试集评估。

## 16. 实验设计

### 16.1 最终实验组

```text
五相机目标条件视觉编码器
+ Visual CFM K=64
+ 目标感知多约束复筛
+ 安全终态投影
+ 1024点未截断时间缩放
+ 抓取状态机
+ PhysX验证
```

### 16.2 对照实验

| 实验 | 验证目的 |
|---|---|
| 不使用目标条件视觉，只输入目标坐标 | 验证目标视觉关联的必要性 |
| 目标坐标投影但无学习adapter | 验证学习式多视角关联的作用 |
| 单腕相机与五相机 | 验证固定相机补盲效果 |
| K=1与K=64 | 验证多候选覆盖率 |
| Visual CFM与确定性minimum-jerk规划 | 验证生成式多路径优势 |
| 无目标接触白名单 | 验证分阶段接触规则的必要性 |

### 16.3 消融实验

| 消融 | 验证性质 |
|---|---|
| 去除Depth | 三维几何感知作用 |
| 去除instance关联 | 同类目标与障碍物区分能力 |
| 去除latent distillation | 是否发生原避障能力遗忘 |
| 去除self-collision loss | CFM隐式自碰撞先验作用 |
| 去除Jerk loss | 低冲击性能 |
| 取消终态投影 | 终态安全修正作用 |
| 使用5 s硬截断 | 未截断时间缩放作用 |

### 16.4 鲁棒性实验

| 干扰 | 建议范围 |
|---|---|
| 目标坐标噪声 | 标准差5、10、20 mm |
| 相机外参误差 | 平移2、5、10 mm；旋转0.5、1、2 deg |
| Depth噪声 | 标准差2、5、10 mm |
| 目标遮挡率 | 0%、25%、50%、75% |
| 视觉丢失 | 随机关闭1至3个相机 |
| 控制延迟 | 0、20、40、80 ms |
| 物体质量变化 | 标称值的0.5至2.0倍 |

## 17. 主要评价指标

```text
target association accuracy
target mask IoU
planning success rate
pregrasp reach success rate
grasp success rate
lift success rate
hold success rate
environment collision rate
self-collision rate
robot mutual-collision rate
minimum clearance
TCP terminal error
acceleration RMS
Jerk RMS
motion duration
PhysX fall rate
emergency-stop rate
```

所有成功率报告任务数、成功数、比例和95% Wilson置信区间。

## 18. 当前已有证据与尚缺实验

### 已有证据

1. 无PPO完整视觉轨迹流程的206条结果为 `176/206 = 85.44%`；
2. 当前PPO在206条集成任务和1000条自然随机任务中的接受数均为0；
3. 取消PPO不会降低当前206条最终结果；
4. 1024点未截断时间缩放使动力学和duration有效性通过率达到100%；
5. PhysX预检查后执行成功率为 `18/19 = 94.74%`。

### 尚缺实验

1. 目标条件视觉编码器训练；
2. 多视角目标实例关联准确率；
3. 抓取物体随机化数据集；
4. 预抓取、闭合、抬升完整状态机测试；
5. 抓取成功率与失效原因统计；
6. payload碰撞与H1平衡耦合验证；
7. 目标坐标、相机标定和遮挡鲁棒性实验。

因此，当前文档给出的是新的完整技术路线，不能在上述实验完成前宣称已经获得完整抓取性能。

## 19. 风险与Sim-to-Real边界

1. 目标坐标误差可能导致夹爪接触位置偏移；
2. 相机外参误差会导致目标实例关联错误；
3. 视觉点云不能覆盖完全遮挡区域；
4. capsule几何可能低估真实机械结构尺寸；
5. 夹爪接触白名单设置过大会掩盖真实碰撞；
6. payload会改变机械臂惯量和H1平衡力矩；
7. 仿真摩擦和接触参数与真实物体不一致；
8. 真实执行必须设置关节速度、加速度、Jerk和位置误差硬保护；
9. 必须具备独立于神经网络的急停、接触检测和通信超时机制。

## 20. 验证流水线

```text
冻结数据与checkpoint哈希校验
        ↓
目标投影单元测试
        ↓
多视角目标实例关联测试
        ↓
视觉编码器validation split评估
        ↓
CFM离线候选生成测试
        ↓
exact几何与时间缩放验证
        ↓
Isaac Sim单任务抓取smoke test
        ↓
并行固定测试集评估
        ↓
PhysX接触与H1平衡验证
        ↓
Omniverse Recorder + TensorBoard + JSON/NPZ报告
```

所有正式实验必须锁定NumPy、PyTorch、CUDA和Isaac Lab seed，并保存配置、checkpoint哈希、数据split、失败原因和原始逐任务记录。
