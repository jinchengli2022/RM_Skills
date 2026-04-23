# RM_Skills

面向睿尔曼机械臂（RM）的技能录制与执行项目，当前代码主流程是：

1. 录制相对路点（可同时记录单夹爪状态）
2. 将录制结果写入技能注册表
3. 执行指定技能（机械臂运动 + 单夹爪开合）

以下说明以当前仓库真实结构和代码为准。

## 目录结构（当前）

```text
RM_Skills/
├── README.md
├── requirements.txt
├── setup.py
├── tmp.py
└── src/
    ├── execute_skill.py
    ├── get_skill.py
    ├── diagnose_grippers.py
    ├── hand_eye_calibration.py
    ├── core/
    │   ├── demo_project.py
    │   ├── get_skill.py
    │   ├── skills.py
    │   ├── web_gripper_control.py
    │   └── zhixing.py
    └── Robotic_Arm/
        ├── rm_ctypes_wrap.py
        └── rm_robot_interface.py
```

## 快速开始

建议在仓库根目录执行命令。

```bash
pip install -r requirements.txt
python src/execute_skill.py
```

默认会连接：

- 机械臂：`169.254.128.19:8080`
- 技能名：`test`
- 单夹爪 Modbus 端口：`1`
- 单夹爪设备 ID：`1`

## 使用流程（重点）

## 1. 先改配置

### 1.1 执行技能配置

编辑 `src/execute_skill.py` 中 `__main__` 下的常量：

```python
ROBOT_IP = "169.254.128.19"
ROBOT_PORT = 8080
SKILL_NAME = "test"
GRIPPER_MODBUS_PORT = 1
GRIPPER_DEVICE_ID = 1
```

### 1.2 录制技能配置

编辑 `src/core/get_skill.py` 顶部常量：

```python
ROBOT_IP = "169.254.128.19"
ROBOT_PORT = 8080

AFFORDANCE_POSE = [0.090005, 0.376255, -0.182519, 3.08, 0.112, -1.897]
RESET_BASE_TO_AFFORDANCE_ON_R = False

GRIPPER_HOST = "169.254.128.18"  # 不采集夹爪可设为 None
GRIPPER_TCP_PORT = 8080
GRIPPER_MODBUS_PORT = 1
GRIPPER_DEVICE_ID = 1
```

## 2. 录制技能点位

执行：

```bash
python src/get_skill.py
```

按键说明：

- 直接回车：保存当前路点（并尝试记录当前夹爪状态）
- `r` + 回车：重置基准点
- `d` + 回车：删除最后一个路点
- `q` + 回车：退出并打印技能片段

退出时会输出可复制内容，核心字段有：

- `affordance_pose`
- `waypoints`
- `gripper_positions`

注意：当前实现里 `gripper_positions` 的值是单个整数（单夹爪），不是双元素数组。

## 3. 注册新技能

把录制结果粘贴到 `src/core/skills.py` 的 `SKILL_REGISTRY`。

最小可用模板：

```python
"my_skill": {
    "name": "我的技能",
    "description": "可选描述",
    "speed": 25,
    "affordance_pose": [x, y, z, rx, ry, rz],
    "waypoints": [
        [dx1, dy1, dz1, drx1, dry1, drz1],
        [dx2, dy2, dz2, drx2, dry2, drz2],
    ],
    "gripper_positions": {
        0: 1,
        1: 0,
    },
},
```

字段说明：

- `waypoints` 必填，单位是米/弧度
- `gripper_positions` 可选，键是路点索引（从 0 开始）
- 夹爪执行逻辑：`<=0` 视为打开，`>0` 视为关闭

## 4. 执行技能

执行：

```bash
python src/execute_skill.py
```

执行器基准位姿优先级（高到低）：

1. 外部传入 `ref_pose`
2. 外部传入 `affordance_pose`
3. 技能内 `ref_pose`
4. 技能内 `affordance_pose`
5. 当前机械臂位姿

如果你要在代码里显式覆盖基准位姿，可参考 `src/execute_skill.py` 中：

```python
executor.execute(SKILL_NAME, affordance_pose=AFFORDANCE_POSE, ref_pose=REF_POSE)
```

## 5. 手眼标定（RealSense）

脚本：`src/hand_eye_calibration.py`

示例：

```bash
python src/hand_eye_calibration.py \
  --robot-ip 169.254.128.19 \
  --board-cols 8 \
  --board-rows 6 \
  --square-size-mm 25 \
  --min-samples 10 \
  --output-dir handeye_output
```

运行后：

- `s` 采样
- `c` 计算标定结果
- `q` 退出

结果保存到：`handeye_output/handeye_result.json`

## 常见问题

## 1. `git push` 提示 Password authentication is not supported

GitHub 已禁用账号密码推送，请改用 PAT（Personal Access Token）或 SSH。

## 2. 录制时提示夹爪初始化失败

先把 `GRIPPER_HOST` 设为 `None`，只录制机械臂路点，后续再排查夹爪网络连通性与参数。

## 3. `diagnose_grippers.py` 无法运行

当前 `src/diagnose_grippers.py` 依赖 `src/core/dual_gripper.py`，但该文件不在当前目录结构中。若需要该诊断流程，请先补齐对应模块或调整脚本依赖。

## 开发说明

- 入口脚本：`src/get_skill.py`、`src/execute_skill.py`
- 核心执行器：`src/core/skills.py` 中 `SkillExecutor`
- 机械臂通信封装：`src/core/demo_project.py`
- 单夹爪控制：`src/core/zhixing.py`
