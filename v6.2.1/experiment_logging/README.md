# 补充实验统一日志

`summary.csv` 每行对应一个 `task × method × seed`。列定义与单位见
`schema.py`，未知或暂时不可读取的值必须写成 `NaN`，不得用 `0` 代替。

连续时序以独立压缩 NPZ 保存，包含关节状态、接触力、物体/夹爪位姿、
相对漂移、物体速度、状态机阶段和 emergency-stop 标志。可读取基座反力时
记录 `base_reaction_force/torque`；不可读取时数组填 `NaN` 并在 metadata 说明。

使用方法：

```powershell
D:\conda_envs\env_isaaclab\python.exe experiment_logging\validate_logs.py `
  outputs\supplementary_experiments\summary.csv
```

验证器检查：必需列、NaN 警告、成功任务轨迹文件、失败枚举、正 duration，
以及所有方法是否在同一 `task_id` 上配对。结果写入 `validation_report.json`。
