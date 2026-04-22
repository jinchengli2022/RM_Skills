# RM_Skills — 机械臂技能框架

## 1. 项目介绍

本项目是一个面向睿尔曼机械臂的**技能录制与执行框架**，支持：

- 通过拖动示教录制相对路点（waypoints）
- 同步录制两只外置伺服夹爪（CTAG2F90D）的状态
- 将录制好的技能一键回放，含路点运动与双夹爪开闭控制
- 夹爪控制基于 Modbus RTU / RS-485，通过 `grasp_resource` SDK 独立驱动，与机械臂通信完全解耦

## 2. 代码结构

```
RM_Skills/
│
├── README.md               <- 项目说明文档
├── requirements.txt        <- 依赖列表
├── setup.py                <- 安装脚本
│
├── src/                    <- 源代码
│   ├── get_skill.py        <- 技能录制入口
│   ├── execute_skill.py    <- 技能执行入口
│   └── core/
│       ├── get_skill.py    <- 录制核心（路点 + 双夹爪状态采集）
│       ├── skills.py       <- 技能注册表 + SkillExecutor 执行引擎
│       ├── dual_gripper.py <- 双夹爪适配层（Modbus RTU SDK 封装）
│       ├── demo_project.py <- 机械臂连接封装（RobotArmController）
│       └── demo_simple_process.py
│
├── grasp_resource/         <- 夹爪相关资料与 SDK
│   ├── sdk/
│   │   └── changingtek_p_rtu_Servo.py  <- 夹爪 Modbus RTU Python SDK
│   ├── ROS/                <- 夹爪 ROS 可视化包
│   └── *.pdf               <- 硬件与协议手册
│
└── src/Robotic_Arm/        <- 睿尔曼机械臂二次开发包
```

## 3. 项目下载

```bash
git clone https://github.com/RealManRobot/RM_API2.git
```

## 4. 环境配置

| 项目       | Linux（推荐）| Windows |
| :--        | :--          | :--     |
| 系统架构   | x86 / ARM    | x86     |
| Python     | 3.10 以上    | 3.10 以上 |
| 夹爪依赖   | minimalmodbus, pyserial | 同左 |

```bash
pip install -r requirements.txt
# 夹爪额外依赖（若 requirements.txt 未包含）
pip install minimalmodbus pyserial
```

Linux 下若出现串口权限问题：

```bash
sudo chmod 666 /dev/ttyUSB1
sudo chmod 666 /dev/ttyUSB2
```

## 5. 注意事项

- 机械臂 IP 地址默认为 `169.254.128.19`，请按实际情况修改 `src/core/get_skill.py` 和 `src/execute_skill.py` 中的 `ROBOT_IP`。
- 夹爪串口默认为 `/dev/ttyUSB1`（夹爪 1）和 `/dev/ttyUSB2`（夹爪 2），在同一串口配置文件的顶部常量处修改。
- **夹爪驱动严禁使用 `src/Robotic_Arm/` 目录下的任何夹爪接口**，统一通过 `src/core/dual_gripper.py` 调用 `grasp_resource/sdk/changingtek_p_rtu_Servo.py`。
- 技能中路点位置单位为**米**，姿态单位为**弧度**，均为相对于基准位姿的偏移量。
- 夹爪位置值（`gripper_positions`）单位与 `MotorController.read_real_position()` 返回值一致，需根据实际机构标定（示例中全开=0，全闭≈9000）。

## 6. 使用指南

### 6.1 录制技能（get_skill）

1. **修改配置**：打开 `src/core/get_skill.py`，确认以下常量：

   ```python
   ROBOT_IP      = "169.254.128.19"  # 机械臂 IP
   GRIPPER_PORT1 = "/dev/ttyUSB1"    # 夹爪 1 串口（无夹爪设为 None）
   GRIPPER_PORT2 = "/dev/ttyUSB2"    # 夹爪 2 串口（无夹爪设为 None）
   ```

2. **运行录制**：

   ```bash
   python src/get_skill.py
   ```

3. **操作说明**：

   | 按键 | 功能 |
   | :-- | :-- |
   | `Enter` | 保存当前路点（同时记录双夹爪位置） |
   | `r + Enter` | 重置基准点为当前位姿 |
   | `d + Enter` | 删除最后一个路点 |
   | `q + Enter` | 退出并打印完整 `waypoints` 与 `gripper_positions` |

4. **录制输出示例**：

   ```
   "waypoints": [
       [0.0000, 0.0000, 0.0500, 0.0000, 0.0000, 0.0000],  # 路点 1
       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # 路点 2
   ],
   "gripper_positions": {
       0: [9000, 9000],  # 路点 1 夹爪状态
       1: [0, 0],        # 路点 2 夹爪状态
   },
   ```

   将输出内容粘贴到 `src/core/skills.py` 的 `SKILL_REGISTRY` 中即可注册新技能。

### 6.2 执行技能（execute_skill）

1. **修改配置**：打开 `src/execute_skill.py`，设置目标技能和机械臂参数：

   ```python
   GRIPPER_PORT1   = "/dev/ttyUSB1"
   GRIPPER_PORT2   = "/dev/ttyUSB2"
   SKILL_NAME      = "set_kettle"        # 技能名称（需已在注册表中）
   AFFORDANCE_POSE = [0.0900, 0.3763, -0.1825, 3.0800, 0.1120, -1.8970]
   ```

2. **运行执行**：

   ```bash
   python src/execute_skill.py
   ```

3. **执行逻辑**：
   - 机械臂依次运动到每个路点（相对于 `AFFORDANCE_POSE` 的绝对位姿）
   - 到达路点后，若该路点索引存在于 `gripper_positions`，则向双夹爪下发目标位置并等待到位
   - 任一夹爪通信失败或超时，立即中止整个技能

### 6.3 注册新技能

在 `src/core/skills.py` 的 `SKILL_REGISTRY` 中添加条目：

```python
SKILL_REGISTRY = {
    "my_skill": {
        "name": "我的技能",
        "description": "简要描述",
        "speed": 20,
        "waypoints": [
            [dx1, dy1, dz1, drx1, dry1, drz1],
            [dx2, dy2, dz2, drx2, dry2, drz2],
        ],
        # 可选：仅在指定路点索引触发双夹爪动作
        "gripper_positions": {
            0: [9000, 9000],  # 路点 1 到达后闭合双夹爪
            1: [0, 0],        # 路点 2 到达后张开双夹爪
        },
    },
}
```

`gripper_positions` 是可选字段：省略时跳过夹爪控制（纯运动技能完全兼容）。

### 6.4 双夹爪 API 速查

```python
from src.core.dual_gripper import DualGripper, DualGripperConfig

cfg = DualGripperConfig(
    port1="/dev/ttyUSB1",
    port2="/dev/ttyUSB2",
    speed_pct=50,         # 速度百分比
    force_pct=25,         # 夹紧力百分比
    reach_timeout=5.0,    # 到位超时（秒）
)
dg = DualGripper(cfg)
dg.connect()

g1, g2 = dg.get_positions()          # 读取双夹爪当前位置
ok = dg.set_positions(9000, 9000)     # 闭合双夹爪（阻塞等待到位）
ok = dg.set_positions(0, 0)           # 张开双夹爪
state = dg.get_full_state()           # 读取位置/速度/电流

dg.disconnect()
```

## 7. 许可证信息

- 本项目遵循 MIT 许可证。
