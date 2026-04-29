"""
机械臂技能模块 (Robot Arm Skills)

通过任务名称驱动机械臂执行预定义的相对轨迹。
每个技能由一组相对于 affordance 局部坐标系的相对位姿组成，机械臂依次经过这些点完成任务。

使用示例:
    from core.skills import SkillExecutor
    executor = SkillExecutor(robot_controller)
    executor.execute("open_door")

扩展方式:
    在 SKILL_REGISTRY 字典中添加新的技能条目即可。
"""

import sys
import os
import time
import threading

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Robotic_Arm.rm_robot_interface import *
from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t


# =============================================================================
# 技能注册表 —— 所有技能在此定义
# =============================================================================
# 每个技能包含:
#   "name":           技能显示名称
#   "description":    技能描述
#   "speed":          默认执行速度 (1-100)
#   "ref_pose":       (可选) 参考位姿 [x,y,z,rx,ry,rz]，所有 waypoints 相对于此位姿
#   "affordance_pose": (可选) 可承受点位姿 [x,y,z,rx,ry,rz]，备选基准点（优先级低于 ref_pose）
#   "waypoints":      affordance 局部坐标系下的相对位姿列表，每个点为 [dx, dy, dz, drx, dry, drz]
#                     位置单位: 米，姿态单位: 弧度
#                     所有 waypoints 都会作为局部变换右乘到 affordance_pose 上
#
# 占位符说明:
#   A, B, C... 等大写字母标注的数值为占位值，需根据实际场景标定后替换
#
# 基准点优先级: 外部 ref_pose > 外部 affordance_pose > 技能 ref_pose > 技能 affordance_pose > 当前位姿
# =============================================================================

# 0为开，1为闭合

SKILL_REGISTRY = {
    "goto_affordance": {
        "name": "到达Affordance",
        "description": "先沿 affordance 局部 -Z 方向退后到预备位，再回到 affordance_pose",
        "speed": 20,
        "waypoints": [
            [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.5236],
        ],
        "gripper_positions": {
            0: 0,  # 路点 1 夹爪状态（单夹爪位置）
            1: 1
        },
    },

    "test": {
        "name": "测试",
        "speed": 25,
        "affordance_pose":   [-0.2532, 0.3638, 0.0810, -2.2960, -0.0170, 1.6140],
        "ref_pose":   [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "waypoints": [
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
        ],
        "gripper_positions": {
            0: 1,  # 路点 1 夹爪状态（单夹爪位置）
        },
    },


    "close_light": {
        "name": "关灯",
        "speed": 20,
        "ref_pose": [0.090005, 0.376255, -0.182519, 3.08, 0.112, -1.897],
        "affordance_pose": [0.090005, 0.376255, -0.182519, 3.08, 0.112, -1.897],
        "waypoints": [
            [0.0019, -0.0931, 0.0253, 1.0350, -0.0620, 1.0350],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    },

    "send_object": {
        "name": "递送物品",
        "speed": 25,
        "waypoints": [
            [-0.0024, 0.0132, 0.1195, 6.2370, -0.0470, 0.0170],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0024, 0.0132, 0.1195, 6.2370, -0.0470, 0.0170],
            [0.0597, 0.2232, 0.3527, 5.8150, 1.4090, -0.1440]
        ],
        # 特殊动作标记: waypoint索引 -> 动作类型
        # "actions": {
        #     0: "gripper_close",   # 到达第0个waypoint后闭合夹爪
        #     3: "gripper_open",    # 到达第3个waypoint后松开夹爪
        # },
    },

    "set_kettle": {
        "name": "放置碗",
        "speed": 25,
        "waypoints": [
            [-0.0026, 0.0112, 0.1074, 0.0500, -0.0780, -0.0230],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0026, 0.0112, 0.1074, 0.0500, -0.0780, -0.0230],
            [0.1151, -0.0183, 0.0850, 0.0500, -0.1170, -0.3850],
            [0.1176, -0.0348, -0.0221, 0.0210, -0.0200, -0.2450]
        ],
        # 特殊动作标记: waypoint索引 -> 动作类型
        # "actions": {
        #     0: "gripper_close",   # 到达第0个waypoint后闭合夹爪
        #     3: "gripper_open",    # 到达第3个waypoint后松开夹爪
        # },
    },

    "open_door": {
        "name": "开门",
        "description": "机械臂从当前位置执行开门动作：前伸→下压门把手→向外拉开→回到起始偏移",
        "speed": 20,
        "waypoints": [
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, -0.05, 0.0, 0.0, 0.0],
            [-0.05, 0.15, -0.05, 0.0, 0.0, 0.0],
            [-0.05, 0.15, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        ],
    },

    "close_door": {
        "name": "关门",
        "description": "机械臂从当前位置执行关门动作：移动到门边→推门→收回",
        "speed": 20,
        "waypoints": [
            # A: 侧向移动到门边
            [0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
            # B: 前伸接触门面
            [0.10, 0.15, 0.0, 0.0, 0.0, 0.0],
            # C: 推门关闭
            [0.10, -0.05, 0.0, 0.0, 0.0, 0.0],
            # D: 收回手臂
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    },



    "receive_object": {
        "name": "接收物品",
        "description": "前伸到递送位置，夹取物品后收回放下",
        "speed": 25,
        "waypoints": [
            # A: 前伸到接收位置
            [0.20, 0.0, 0.0, 0.0, 0.0, 0.0],
            # B: 收回
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # C: 下降放置
            [0.0, 0.0, -0.10, 0.0, 0.0, 0.0],
            # D: 抬起复位
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        # gripper_positions: 仅在记录了夹爪值的路点索引下发命令
        # 示例（需根据实际标定值替换）：
        # "gripper_positions": {
        #     0: 9000,   # 路点 1 到位后夹爪闭合
        #     2: 0,      # 路点 3 到位后夹爪张开
        # },
    },

    # ── 示例：带夹爪动作的技能（新格式） ────────────────────────────
    "example_dual_gripper": {
        "name": "夹爪示例",
        "description": "演示 gripper_positions 格式：在路点 0 闭合、路点 2 张开",
        "speed": 20,
        "waypoints": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        # 键为路点索引（0 起始），值为单夹爪位置
        "gripper_positions": {
            0: 9000,   # 路点 1 到位后夹爪闭合
            2: 0,      # 路点 3 到位后夹爪张开
        },
    },

    "operate_device": {
        "name": "操作设备",
        "description": "按下设备按钮：前伸→对准按钮→按压→释放→收回",
        "speed": 15,
        "waypoints": [
            # A: 前伸靠近设备
            [0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
            # B: 微调对准按钮
            [0.12, 0.02, -0.01, 0.0, 0.0, 0.0],
            # C: 按压按钮
            [0.15, 0.02, -0.01, 0.0, 0.0, 0.0],
            # D: 释放按钮
            [0.12, 0.02, -0.01, 0.0, 0.0, 0.0],
            # E: 收回手臂
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    },

    "wave_hello": {
        "name": "挥手打招呼",
        "description": "机械臂执行挥手动作",
        "speed": 30,
        "waypoints": [
            # A: 抬起手臂
            [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            # B: 向左挥
            [0.0, 0.08, 0.15, 0.0, 0.0, 0.3],
            # C: 向右挥
            [0.0, -0.08, 0.15, 0.0, 0.0, -0.3],
            # D: 向左挥
            [0.0, 0.08, 0.15, 0.0, 0.0, 0.3],
            # E: 向右挥
            [0.0, -0.08, 0.15, 0.0, 0.0, -0.3],
            # F: 回到起始上方
            [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            # G: 回到起始
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    },

    "wipe_surface": {
        "name": "擦拭表面",
        "description": "在当前位置前方执行左右擦拭动作",
        "speed": 20,
        "waypoints": [
            # A: 前伸贴近表面
            [0.10, 0.0, -0.02, 0.0, 0.0, 0.0],
            # B: 向左擦
            [0.10, 0.08, -0.02, 0.0, 0.0, 0.0],
            # C: 向右擦
            [0.10, -0.08, -0.02, 0.0, 0.0, 0.0],
            # D: 向左擦
            [0.10, 0.08, -0.02, 0.0, 0.0, 0.0],
            # E: 向右擦
            [0.10, -0.08, -0.02, 0.0, 0.0, 0.0],
            # F: 收回
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    },
}


# =============================================================================
# 技能执行器
# =============================================================================

class SkillExecutor:
    """
    技能执行器，负责解析技能定义并驱动机械臂执行。

    Args:
        robot_controller: RobotArmController 实例（来自 demo_project.py）
        dual_gripper: DualGripper 实例（可选）。提供时按技能的 gripper_positions
                      在对应路点下发双夹爪命令；未提供时跳过夹爪控制。
    """

    def __init__(self, robot_controller, dual_gripper=None):
        self.rc = robot_controller
        self.robot = robot_controller.robot
        self.dual_gripper = dual_gripper

    # ----- 公开接口 -----

    def list_skills(self) -> list[str]:
        """列出所有已注册的技能名称"""
        return list(SKILL_REGISTRY.keys())

    def get_skill_info(self, skill_name: str):
        """获取指定技能的描述信息"""
        skill = SKILL_REGISTRY.get(skill_name)
        if skill is None:
            return None
        return {
            "name": skill["name"],
            "description": skill.get("description", ""),
            "speed": skill["speed"],
            "num_waypoints": len(skill["waypoints"]),
        }

    def execute(self, skill_name: str, ref_pose: list[float] = None, affordance_pose: list[float] = None, speed: int = None, block: int = 1) -> int:
        """
        执行指定技能。

        机械臂先运动到第一个路点（绝对位姿），再依次执行所有路点。
        所有路点均为相对于 affordance 局部坐标系的相对位姿。

        Args:
            skill_name (str): 技能名称，必须在 SKILL_REGISTRY 中已注册。
            ref_pose (list[float], optional): 外部参考位姿 [x,y,z,rx,ry,rz]（最高优先级）。
            affordance_pose (list[float], optional): 技能基准位姿 [x,y,z,rx,ry,rz]，
                同时定义 affordance 局部坐标系的原点与朝向。
                若两者均不提供，则优先读取技能内 ref_pose，其次 affordance_pose，最后当前位姿。
            speed (int, optional): 覆盖技能默认速度，范围 1-100。
            block (int): 阻塞模式，1=等待运动完成，0=非阻塞。

        Returns:
            int: 0 表示全部执行成功，非0 表示某步骤失败（返回失败步骤的索引+1的负数）。
        """
        # 1. 查找技能
        skill = SKILL_REGISTRY.get(skill_name)
        if skill is None:
            print(f"\n[SkillExecutor] ❌ 未知技能: '{skill_name}'")
            print(f"  可用技能: {', '.join(self.list_skills())}")
            return -1

        v = speed if speed is not None else skill["speed"]
        waypoints = skill["waypoints"]
        actions = skill.get("actions", {})
        gripper_positions = skill.get("gripper_positions", {})

        print(f"\n{'='*60}")
        print(f"  执行技能: {skill['name']} ({skill_name})")
        print(f"  描述: {skill.get('description', '')}")
        print(f"  速度: {v}%  |  路点数: {len(waypoints)}")
        if gripper_positions:
            print(f"  夹爪动作路点: {sorted(gripper_positions.keys())}")
        print(f"{'='*60}")

        # 2. 确定基准位姿：优先级为 外部ref_pose > 外部affordance_pose > 技能ref_pose > 技能affordance_pose > 当前位姿
        if ref_pose is not None:
            base_pose = ref_pose
            print(f"  基准位姿（外部 ref_pose）: [{', '.join(f'{x:.4f}' for x in base_pose)}]")
        elif affordance_pose is not None:
            base_pose = affordance_pose
            print(f"  基准位姿（外部 affordance_pose）: [{', '.join(f'{x:.4f}' for x in base_pose)}]")
        elif skill.get("ref_pose") is not None:
            base_pose = list(skill["ref_pose"])
            print(f"  基准位姿（技能 ref_pose）: [{', '.join(f'{x:.4f}' for x in base_pose)}]")
        elif skill.get("affordance_pose") is not None:
            base_pose = list(skill["affordance_pose"])
            print(f"  基准位姿（技能 affordance_pose）: [{', '.join(f'{x:.4f}' for x in base_pose)}]")
        else:
            base_pose = self._get_current_pose()
            if base_pose is None:
                print("[SkillExecutor] ❌ 无法获取当前机械臂位姿，中止执行")
                return -2
            print(f"  基准位姿（当前位姿）: [{', '.join(f'{x:.4f}' for x in base_pose)}]")

        if not isinstance(base_pose, list) or len(base_pose) != 6:
            print(f"[SkillExecutor] ❌ 基准位姿格式错误: {base_pose}")
            return -2

        if not waypoints:
            print("[SkillExecutor] ⚠️  该技能没有任何路点，跳过执行")
            return 0



        # 3. 先运动到第一个路点（movel 绝对位姿）
        first_target = self._apply_local_waypoint(base_pose, waypoints[0])
        print(f"\n  ➤ 前往起始路点 1/{len(waypoints)}: affordance局部相对位姿 {waypoints[0]}")
        print(f"    目标位姿: [{', '.join(f'{x:.4f}' for x in first_target)}]")

        ret = self._movel_with_monitor(first_target, v, block)
        if ret != 0:
            print(f"  ❌ 前往起始路点失败, 错误码: {ret}")
            return -1

        print(f"    ✓ 到达起始路点")
        if 0 in actions:
            self._execute_action(actions[0])
        if gripper_positions and 0 in gripper_positions:
            ret = self._execute_gripper(gripper_positions[0])
            if not ret:
                print("  ❌ 夹爪执行失败，中止技能")
                return -10
            
        

        # 4. 依次执行后续路点
        for i, offset in enumerate(waypoints[1:], start=1):
            target_pose = self._apply_local_waypoint(base_pose, offset)

            print(f"\n  ▶ 路点 {i+1}/{len(waypoints)}: affordance局部相对位姿 {offset}")
            print(f"    目标位姿: [{', '.join(f'{x:.4f}' for x in target_pose)}]")

            ret = self._movel_with_monitor(target_pose, v, block)

            if ret != 0:
                print(f"  ❌ 路点 {i+1} 运动失败, 错误码: {ret}")
                return -(i + 1)

            print(f"    ✓ 到达")

            # 执行该路点关联的特殊动作（兼容旧技能）
            if i in actions:
                self._execute_action(actions[i])

            # 执行双夹爪命令
            if gripper_positions and i in gripper_positions:
                ret = self._execute_gripper(gripper_positions[i])
                if not ret:
                    print("  ❌ 夹爪执行失败，中止技能")
                    return -(i + 10)

        print(f"\n  ✅ 技能 '{skill['name']}' 执行完成")
        print(f"{'='*60}\n")
        return 0

    # ----- 内部方法 -----

    def _get_current_pose(self):
        """获取当前末端位姿 [x, y, z, rx, ry, rz]"""
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"  [警告] rm_get_current_arm_state 失败, 错误码: {ret}")
            return None
        pose = state.get("pose")
        if pose is None or len(pose) != 6:
            print(f"  [警告] 获取到的位姿数据异常: {pose}")
            return None
        return list(pose)

    def _start_pose_monitor(self, interval: float = 0.15) -> threading.Event:
        """启动后台线程实时打印末端位姿，返回用于停止的 Event。"""
        stop_event = threading.Event()

        def _loop():
            while not stop_event.is_set():
                pose = self._get_current_pose()
                if pose is not None:
                    ps = ", ".join(f"{v:8.4f}" for v in pose)
                    print(f"    [末端位姿] [{ps}]")
                stop_event.wait(interval)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return stop_event

    def _movel_with_monitor(self, target: list[float], speed: int, block: int) -> int:
        """执行 rm_movel 并在阻塞期间实时输出末端位姿。"""
        stop_event = self._start_pose_monitor()
        ret = self.robot.rm_movel(target, speed, 0, 0, block)
        stop_event.set()
        return ret

    def _pose_to_matrix(self, pose: list[float]) -> np.ndarray:
        """将 RM pose 转换为 4x4 齐次变换矩阵。"""
        rm_matrix = self.robot.rm_algo_pos2matrix(pose)
        values = [float(v) for v in rm_matrix.data]
        return np.array(values, dtype=np.float64).reshape(4, 4)

    def _matrix_to_pose(self, matrix: np.ndarray) -> list[float]:
        """将 4x4 齐次变换矩阵转换为 RM pose。"""
        rm_mat = rm_matrix_t(data=matrix.tolist())
        pose = self.robot.rm_algo_matrix2pos(rm_mat, 1)
        return [float(v) for v in pose]

    def _apply_local_waypoint(self, base_pose: list[float], waypoint: list[float]) -> list[float]:
        """将 affordance 局部相对位姿应用到基准位姿上，返回新的绝对位姿。"""
        base_transform = self._pose_to_matrix(base_pose)
        waypoint_transform = self._pose_to_matrix(waypoint)
        target_transform = base_transform @ waypoint_transform
        return self._matrix_to_pose(target_transform)

    def _execute_gripper(self, positions) -> bool:
        """直接调用 zhixing.py 的 open_gripper/close_gripper。"""
        if self.dual_gripper is None:
            print("    [夹爪] ⚠️  未配置 DualGripper，跳过夹爪命令")
            return True  # 无夹爪时不视为失败，避免阻断纯位姿技能

        try:
            target = int(positions)
        except Exception:
            print(f"    [夹爪] ⚠️  positions 格式错误(需为单个整数): {positions}")
            return False
        
        # 注意每次执行都要重新连接和断开，否则可能会因为连接状态异常导致命令无法执行！！！
        if target <= 0:
            print(f"    建立连接")
            self.dual_gripper.connect()  # 确保连接已建立
            print(f"    ⚙ 执行单夹爪打开: g={target}")
            self.dual_gripper.open_gripper(delay=1.0)
            self.dual_gripper.disconnect()
        else:
            print(f"    建立连接")
            self.dual_gripper.connect()  # 确保连接已建立
            print(f"    ⚙ 执行单夹爪关闭: g={target}")
            self.dual_gripper.close_gripper(delay=1.0)
            self.dual_gripper.disconnect()
        return True

    def _execute_action(self, action: str):
        """执行路点关联的旧式特殊动作标记（仅保留日志，不再调用 src 夹爪驱动）。

        Note:
            旧式 actions 字典中的夹爪动作已由 gripper_positions 替代。
            此方法保留是为了兼容已有技能中的非夹爪动作扩展点。
        """
        if action in ("gripper_close", "gripper_open"):
            print(f"    [actions] ⚠️  旧式夹爪动作 '{action}' 已弃用，"
                  "请改用 gripper_positions 字段。")
        else:
            print(f"    [警告] 未知动作类型: {action}")


# =============================================================================
# 工具函数 —— 用于注册自定义技能
# =============================================================================

def register_skill(name: str, display_name: str, description: str,
                   waypoints: list[list[float]], speed: int = 20,
                   actions: dict[int, str] = None):
    """
    动态注册一个新技能到全局注册表。

    Args:
        name: 技能唯一标识名（英文，如 "pick_up"）
        display_name: 技能显示名称（如 "拾取物品"）
        description: 技能描述
        waypoints: affordance 局部坐标系下的相对位姿列表 [[dx,dy,dz,drx,dry,drz], ...]
        speed: 默认速度百分比
        actions: 路点索引到动作的映射，如 {0: "gripper_close", 2: "gripper_open"}
    """
    if name in SKILL_REGISTRY:
        print(f"[register_skill] ⚠️ 覆盖已有技能: {name}")

    skill_def = {
        "name": display_name,
        "description": description,
        "speed": speed,
        "waypoints": waypoints,
    }
    if actions:
        skill_def["actions"] = actions

    SKILL_REGISTRY[name] = skill_def
    print(f"[register_skill] ✓ 已注册技能: {name} ({display_name})")


# =============================================================================
# 可视化函数 —— 将技能路点和夹爪朝向绘制为 3D 图
# =============================================================================

def visualize_skill(skill_name: str, base_pose: list[float] = None):
    """
    将指定技能的所有路点及夹爪朝向可视化为 3D 图。

    图中内容:
      - 灰色星号 (*)      : affordance 原点（执行技能时的局部坐标系原点）
      - 彩色圆点 (●)      : 各路点位置，颜色随序号渐变（蓝→红）
      - 彩色箭头           : 夹爪 Z 轴朝向（由欧拉角 rx/ry/rz 计算的旋转矩阵第三列）
      - 数字标签            : 路点序号
      - 黑色虚线箭头       : 路点间的运动顺序
      - 橙色圆点 + "G"标签: 带夹爪动作的路点（open/close）

    Args:
        skill_name (str): 技能名称，必须在 SKILL_REGISTRY 中已注册。
        base_pose (list[float], optional): affordance 基准位姿 [x,y,z,rx,ry,rz]，
            默认全零（局部坐标原点）。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[visualize_skill] ❌ Missing dependencies. Please install matplotlib and numpy:")
        print("    pip install matplotlib numpy")
        return

    skill = SKILL_REGISTRY.get(skill_name)
    if skill is None:
        print(f"[visualize_skill] ❌ Unknown skill: '{skill_name}'")
        return

    if base_pose is None:
        base_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    waypoints = skill["waypoints"]
    actions   = skill.get("actions", {})

    # --- Local waypoint transform composition ---
    origin = base_pose[:]

    def euler_to_rot(rx, ry, rz):
        """ZYX order: R = Rz @ Ry @ Rx"""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,  0,   0 ],
                       [0,  cx, -sx],
                       [0,  sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                       [ 0,  1, 0 ],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [0,   0,  1]])
        return Rz @ Ry @ Rx

    def rot_to_euler(rotation):
        """ZYX intrinsic inverse: rotation = Rz @ Ry @ Rx."""
        sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
        singular = sy < 1e-8
        if not singular:
            rx = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
            ry = float(np.arctan2(-rotation[2, 0], sy))
            rz = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        else:
            rx = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
            ry = float(np.arctan2(-rotation[2, 0], sy))
            rz = 0.0
        return rx, ry, rz

    def pose_to_transform(pose):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = euler_to_rot(pose[3], pose[4], pose[5])
        transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
        return transform

    def transform_to_pose(transform):
        rx, ry, rz = rot_to_euler(transform[:3, :3])
        translation = transform[:3, 3]
        return [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            rx,
            ry,
            rz,
        ]

    def apply_local_waypoint(base, waypoint):
        return transform_to_pose(pose_to_transform(base) @ pose_to_transform(waypoint))

    abs_poses = [apply_local_waypoint(base_pose, wp) for wp in waypoints]

    # --- Color Mapping: Sequence 0 (Blue) -> n-1 (Red) ---
    cmap = plt.cm.coolwarm
    n = len(abs_poses)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # --- Calculate Arrow Length (scaled to trajectory size) ---
    all_xyz = np.array([[p[0], p[1], p[2]] for p in abs_poses] + [[origin[0], origin[1], origin[2]]])
    span = np.ptp(all_xyz, axis=0)
    arrow_len = max(np.max(span) * 0.12, 0.02)  # Minimum 2cm

    # --- Plotting ---
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Base Origin
    ax.scatter(*origin[:3], marker="*", s=220, c="grey", zorder=5, label="Affordance Origin")
    ax.text(origin[0], origin[1], origin[2], "  Affordance", fontsize=8, color="grey")

    # Trajectory Lines (Movement order)
    traj_xyz = [origin[:3]] + [p[:3] for p in abs_poses]
    xs = [p[0] for p in traj_xyz]
    ys = [p[1] for p in traj_xyz]
    zs = [p[2] for p in traj_xyz]
    for i in range(len(traj_xyz) - 1):
        ax.quiver(xs[i], ys[i], zs[i],
                  xs[i+1] - xs[i], ys[i+1] - ys[i], zs[i+1] - zs[i],
                  color="black", alpha=0.35, linewidth=1.2,
                  arrow_length_ratio=0.12, linestyle="dashed")

    # Individual Waypoints
    for i, pose in enumerate(abs_poses):
        x, y, z, rx, ry, rz = pose
        c = colors[i]

        has_action = i in actions
        marker_size = 100 if not has_action else 160
        ax.scatter(x, y, z, c=[c], s=marker_size, zorder=6,
                   edgecolors="orange" if has_action else "none", linewidths=2)

        # Labels
        label = f"  {i+1}"
        if has_action:
            act = actions[i]
            # Using text labels instead of emojis for maximum compatibility
            icon = "[C]" if act == "gripper_close" else "[O]"
            label += f" {icon}"
        ax.text(x, y, z, label, fontsize=9, color="black")

        # Gripper Orientation (Rotation Matrix Column 3 = Z-axis)
        R = euler_to_rot(rx, ry, rz)
        zvec = R[:, 2] * arrow_len
        ax.quiver(x, y, z, zvec[0], zvec[1], zvec[2],
                  color=c, linewidth=2.0, arrow_length_ratio=0.3)

    # --- Legend and Labels ---
    action_patch = mpatches.Patch(edgecolor="orange", facecolor="white",
                                  linewidth=2, label="Gripper Action Waypoint")
    arrow_patch  = mpatches.Patch(color="steelblue", label="Gripper Z-Axis (Orientation)")
    traj_patch   = mpatches.Patch(color="black", alpha=0.4,
                                  label="Movement Path (Dashed)")
    ax.legend(handles=[action_patch, arrow_patch, traj_patch],
              loc="upper left", fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    title_str = f"Skill Trajectory: {skill.get('name', 'Unnamed')} ({skill_name})\n"
    title_str += f"{skill.get('description', 'No description available')}"
    ax.set_title(title_str, fontsize=10)

    # Equalize Axis Scaling
    max_range = np.max(np.ptp(all_xyz, axis=0)) / 2.0
    if max_range < 0.01:
        max_range = 0.05
    mid = all_xyz.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


def execute_skill(skill_name: str, affordance_pose: list[float] = None):
    from core.demo_project import RobotArmController, rm_api_version

    # 连接机械臂
    robot_controller = RobotArmController("169.254.128.19", 8080, 3)
    print(f"\nAPI Version: {rm_api_version()}\n")

    executor = SkillExecutor(robot_controller)

    # 列出所有可用技能
    # print("可用技能列表:")
    # for skill_name in executor.list_skills():
    #     info = executor.get_skill_info(skill_name)
    #     print(f"  - {skill_name}: {info['name']} ({info['num_waypoints']}个路点)")

    # 动态注册一个自定义技能示例
    # register_skill(
    #     name="push_button",
    #     display_name="按按钮",
    #     description="前伸按下按钮后收回",
    #     waypoints=[
    #         [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],   # A: 靠近
    #         [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],     # B: 按下
    #         [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],     # C: 释放
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # D: 收回
    #     ],
    #     speed=15,
    # )

    # 执行技能（取消注释以实际运行）
    executor.execute(skill_name, affordance_pose=affordance_pose)

    # ── 可视化（无需连接机械臂，可单独调用）─────────────────────────────────
    # visualize_skill("open_door")
    # visualize_skill("send_object")
    # visualize_skill("operate_device")

    robot_controller.disconnect()


if __name__ == "__main__":
    execute_skill("open_door")
