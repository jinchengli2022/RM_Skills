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
├── README.md                 <- 项目说明文档
├── requirements.txt          <- 依赖列表
├── setup.py                  <- 安装脚本
├── diagnose_grippers.py      <- 夹爪连接诊断工具
│
├── src/                      <- 源代码
│   ├── get_skill.py          <- 技能录制入口
│   ├── execute_skill.py      <- 技能执行入口
│   └── core/
│       ├── get_skill.py      <- 录制核心（路点 + 双夹爪状态采集 + Affordance/ref_pose）
│       ├── skills.py         <- 技能注册表 + SkillExecutor 执行引擎
│       ├── dual_gripper.py   <- 双夹爪适配层（Modbus RTU SDK 封装）
│       ├── demo_project.py   <- 机械臂连接封装（RobotArmController）
│       └── demo_simple_process.py
│
├── grasp_resource/           <- 夹爪相关资料与 SDK
│   ├── sdk/
│   │   └── changingtek_p_rtu_Servo.py  <- 夹爪 Modbus RTU Python SDK
│   ├── ROS/                  <- 夹爪 ROS 可视化包
│   └── *.pdf                 <- 硬件与协议手册
│
└── src/Robotic_Arm/          <- 睿尔曼机械臂二次开发包
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
- 夹爪串口连接方式支持两种：**双独立串口**（模式A）或**单串口主从级联**（模式B）。不确定时，运行 `diagnose_grippers.py` 自动检测。
- **夹爪驱动严禁使用 `src/Robotic_Arm/` 目录下的任何夹爪接口**，统一通过 `src/core/dual_gripper.py` 调用 `grasp_resource/sdk/changingtek_p_rtu_Servo.py`。
- 技能中路点位置单位为**米**，姿态单位为**弧度**，均为相对于基准位姿的偏移量。
- 夹爪位置值（`gripper_positions`）单位与 `MotorController.read_real_position()` 返回值一致，需根据实际机构标定（示例中全开=0，全闭≈9000）。
- **基准位姿优先级**（从高到低）：外部 `ref_pose` 参数 > 外部 `affordance_pose` 参数 > 技能内 `ref_pose` > 技能内 `affordance_pose` > 当前机械臂位姿。

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
   "affordance_pose": [0.0900, 0.3763, -0.1825, 3.0800, 0.1120, -1.8970],  # 基准点（可选）
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
   ```

2. **运行执行**（基准点由优先级确定）：

   ```bash
   # 使用技能内定义的 ref_pose 或 affordance_pose
   python src/execute_skill.py
   
   # 或通过代码传入外部参数，覆盖技能内配置
   # executor.execute(SKILL_NAME, ref_pose=[x, y, z, rx, ry, rz])
   ```

3. **基准点确定优先级**（从高到低）：
   1. 外部传入的 `ref_pose` 参数
   2. 外部传入的 `affordance_pose` 参数
   3. 技能内定义的 `ref_pose`
   4. 技能内定义的 `affordance_pose`
   5. 当前机械臂末端位姿

4. **执行逻辑**：
   - 机械臂依次运动到每个路点（相对于确定的基准位姿的绝对位姿）
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
        # 可选：基准点定义
        "ref_pose": [x, y, z, rx, ry, rz],           # 推荐使用
        # 可选：备选基准点（优先级低于 ref_pose）
        "affordance_pose": [x, y, z, rx, ry, rz],
        # 必须：路点定义
        "waypoints": [
            [dx1, dy1, dz1, drx1, dry1, drz1],
            [dx2, dy2, dz2, drx2, dry2, drz2],
        ],
        # 可选：指定路点索引处的双夹爪动作
        "gripper_positions": {
            0: [9000, 9000],  # 路点 1 到达后闭合双夹爪
            1: [0, 0],        # 路点 2 到达后张开双夹爪
        },
    },
}
```

**字段说明**：
- `ref_pose` 和 `affordance_pose` 都是可选的基准点定义，`ref_pose` 优先级更高
- `gripper_positions` 是可选字段：省略时跳过夹爪控制（纯运动技能完全兼容）
- 所有字段省略时，执行器使用当前机械臂位姿作为基准点

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

### 6.5 诊断夹爪连接（diagnose_grippers）

若不确定夹爪的连接方式（双独立串口 vs 单串口主从级联），可使用诊断工具自动检测：

```bash
python diagnose_grippers.py
```

**诊断流程**：

| 步骤 | 操作 | 说明 |
| :-- | :-- | :-- |
| 1 | 扫描可用串口 | 列举系统中所有 USB 串口 |
| 2 | 模式 A | 尝试双独立串口（最常见） |
| - | 模式 B | 尝试单串口主从级联 |
| - | 模式 C | 检测单夹爪（故障排查） |
| 3 | 输出配置 | 打印推荐的常量配置 |

**输出示例**：

```
✓ 诊断成功！

  推荐配置（dual_independent）:
    GRIPPER_PORT1 = '/dev/ttyUSB1'
    GRIPPER_PORT2 = '/dev/ttyUSB2'
    SLAVE_ID_1 = 1
    SLAVE_ID_2 = 1
```

将诊断结果配置到 `src/execute_skill.py` 和 `src/core/get_skill.py` 的常量部分即可。

## 7. API 参考

### 7.1 SkillExecutor.execute() 方法

```python
def execute(
    self,
    skill_name: str,
    ref_pose: list[float] = None,          # 优先级最高
    affordance_pose: list[float] = None,   # 优先级次高
    speed: int = None,
    block: int = 1
) -> int:
    """
    执行指定技能。
    
    Args:
        skill_name: 技能名称
        ref_pose: 外部基准点（可覆盖技能内配置）
        affordance_pose: 备选外部基准点
        speed: 运动速度百分比（1-100）
        block: 1=阻塞等待完成，0=非阻塞
    
    Returns:
        0: 执行成功
        负数: 执行失败（负值表示失败步骤号）
    """
```

### 7.2 DualGripper 类

见 **6.4 双夹爪 API 速查** 部分。

## 8. 许可证信息

- 本项目遵循 MIT 许可证。
