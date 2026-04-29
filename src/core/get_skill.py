"""
技能点位采集工具 (Skill Waypoint Recorder)

启动后记录 affordance 基准位姿，然后实时输出当前位置相对于该基准的局部相对位姿，
格式与 skills.py 的 waypoints 完全一致，可直接复制粘贴使用。

用法:
    python src/core/get_skill.py

操作说明:
    - 启动后以 Affordance 点作为基准点（可配置）
    - 手动拖动机械臂到目标位置（需先开启拖动示教模式）
    - 实时终端输出 affordance 局部坐标系下的相对位姿，格式为 [dx, dy, dz, drx, dry, drz]
    - 按 Enter 键保存当前位置为一个路点
    - 按 q 键退出并打印完整的 waypoints 列表
    - 按 r 键重置基准点为 Affordance 点（可配置）
    - 按 d 键删除最后一个已保存的路点
"""

from __future__ import annotations

import sys
import os
import time
import threading
import socket
import json
import select
import termios
import tty

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Robotic_Arm.rm_robot_interface import *
from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
from src.core.demo_project import RobotArmController
from src.core.zhixing import GripperController

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROBOT_IP   = "169.254.128.18"
ROBOT_PORT = 8080
REFRESH_HZ = 10          # 实时刷新频率（次/秒）
PRINT_DIGITS = 4         # 输出小数位数

# Affordance 点绝对位姿（技能基准点）。
# 设为 None 时，启动后使用当前末端位姿作为 affordance 原点与局部坐标轴。
# 录制输出时会以 "affordance_pose" 字段保留在技能定义里。
AFFORDANCE_POSE = None

# r 键的重置策略
# True: 重置到 AFFORDANCE_POSE
# False: 重置到当前机械臂位姿
RESET_BASE_TO_AFFORDANCE_ON_R = False

# 单夹爪 TCP 配置（基于 zhixing.py 的 GripperController）
# 不使用夹爪时将 GRIPPER_HOST 设为 None
GRIPPER_HOST = "169.254.128.19"
GRIPPER_TCP_PORT = 8080
GRIPPER_MODBUS_PORT = 1
GRIPPER_DEVICE_ID = 1
# ─────────────────────────────────────────────────────────────────────────────


def get_current_pose(robot):
    """获取当前末端位姿 [x, y, z, rx, ry, rz]"""
    ret, state = robot.rm_get_current_arm_state()
    if ret != 0:
        return None
    pose = state.get("pose")
    if pose is None or len(pose) != 6:
        return None
    return list(pose)


def pose_to_matrix(robot, pose: list[float]) -> np.ndarray:
    """将 RM pose 转换为 4x4 齐次变换矩阵。"""
    rm_matrix = robot.rm_algo_pos2matrix(pose)
    values = [float(v) for v in rm_matrix.data]
    return np.array(values, dtype=np.float64).reshape(4, 4)


def matrix_to_pose(robot, matrix: np.ndarray) -> list[float]:
    """将 4x4 齐次变换矩阵转换为 RM pose。"""
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(v) for v in pose]


def calc_local_offset(robot, base_pose: list[float], current_pose: list[float]) -> list[float]:
    """计算当前位姿相对于 affordance 基准位姿的局部相对位姿。"""
    base_transform = pose_to_matrix(robot, base_pose)
    current_transform = pose_to_matrix(robot, current_pose)
    local_transform = np.linalg.inv(base_transform) @ current_transform
    local_pose = matrix_to_pose(robot, local_transform)
    return [round(value, PRINT_DIGITS) for value in local_pose]


def format_waypoint(offset: list[float]) -> str:
    """格式化为 skills.py waypoint 格式"""
    parts = ", ".join(f"{v:.{PRINT_DIGITS}f}" for v in offset)
    return f"[{parts}]"


def clear_line():
    """清除当前行"""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


class SingleKeyReader:
    """在终端中读取单键输入；非 TTY 时回退到逐行输入。"""

    def __init__(self):
        self._fd: int | None = None
        self._old_settings = None
        self._is_tty = False

    def __enter__(self) -> "SingleKeyReader":
        self._is_tty = sys.stdin.isatty()
        if self._is_tty:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._is_tty and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self, stop_event: threading.Event) -> str | None:
        """阻塞等待单键；若收到 stop_event 则返回 None。"""
        if not self._is_tty:
            try:
                line = input()
            except EOFError:
                return None
            if line == "":
                return "\n"
            line = line.strip().lower()
            return line[:1] if line else "\n"

        assert self._fd is not None
        while not stop_event.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue

            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                return "\n"
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1b":
                while True:
                    ready, _, _ = select.select([self._fd], [], [], 0)
                    if not ready:
                        break
                    sys.stdin.read(1)
                continue
            return ch.lower()
        return None


def print_saved_waypoints(affordance_pose: list[float], waypoints: list[list[float]], gripper_positions: dict):
    """打印技能片段：affordance_pose、局部坐标系下的 waypoints 及单夹爪位置字典。"""
    print("\n" + "=" * 60)
    print("  已保存的技能片段（可直接粘贴到 skills.py）:")
    print("=" * 60)
    print(f"  \"affordance_pose\": {format_waypoint(affordance_pose)},")
    print("  \"waypoints\": [")
    for i, wp in enumerate(waypoints):
        comma = "," if i < len(waypoints) - 1 else ""
        print(f"      {format_waypoint(wp)}{comma}  # 路点 {i+1}（affordance 局部相对位姿）")
    print("  ],")
    if gripper_positions:
        print("  \"gripper_positions（仅为示例，需根据实际标定值替换）\": {")
        indices = sorted(gripper_positions.keys())
        for j, idx in enumerate(indices):
            g = gripper_positions[idx]
            comma = "," if j < len(indices) - 1 else ""
            print(f"      {idx}: {g}{comma}  # 路点 {idx+1} 夹爪状态")
        print("  },")
    print("=" * 60 + "\n")


class SingleGripperViaZhixing:
    """使用 zhixing.py 的 GripperController 实现单夹爪位置读取。"""

    def __init__(self, host: str, tcp_port: int, modbus_port: int, device_id: int):
        self.host = host
        self.tcp_port = tcp_port
        self.modbus_port = modbus_port
        self.device_id = device_id
        self._sock: socket.socket | None = None
        self._gripper: GripperController | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.tcp_port))

        setup_cmd = {
            "command": "set_modbus_mode",
            "port": self.modbus_port,
            "baudrate": 115200,
            "timeout": 3,
        }
        self._send(setup_cmd)

        self._gripper = GripperController(self._sock, device_id=self.device_id, port=self.modbus_port)

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._gripper = None

    def get_position(self) -> int:
        if self._gripper is None:
            raise RuntimeError("夹爪未连接，请先调用 connect()")
        return self._read_position(self._gripper)

    def _read_position(self, controller: GripperController) -> int:
        # 读取 0x0102/0x0103 (十进制 258/259) 作为 32 位位置值
        cmd = {
            "command": "read_holding_registers",
            "port": controller.port,
            "address": 258,
            "num": 2,
            "device": controller.device_id,
        }
        resp = self._send(cmd)
        regs = self._extract_regs(resp)
        if len(regs) >= 2:
            return ((int(regs[0]) & 0xFFFF) << 16) | (int(regs[1]) & 0xFFFF)
        if len(regs) == 1:
            return int(regs[0])
        raise RuntimeError(f"夹爪位置响应无寄存器数据: {resp}")

    def _send(self, command: dict) -> str:
        if self._sock is None:
            raise RuntimeError("夹爪 socket 未连接")
        payload = json.dumps(command) + "\n"
        with self._lock:
            self._sock.sendall(payload.encode("utf-8"))
            return self._sock.recv(1024).decode()

    @staticmethod
    def _extract_regs(resp_text: str) -> list[int]:
        try:
            data = json.loads(resp_text.strip())
        except Exception:
            return []

        for key in ("data", "registers", "values"):
            val = data.get(key)
            if isinstance(val, list):
                return [int(v) for v in val]

        val = data.get("data")
        if isinstance(val, int):
            return [int(val)]
        return []


def main():
    print("\n" + "=" * 60)
    print("  技能点位采集工具 (Skill Waypoint Recorder)")
    print("=" * 60)

    # ── 连接机械臂 ──
    print(f"\n正在连接机械臂 {ROBOT_IP}:{ROBOT_PORT} ...")
    rc = RobotArmController(ROBOT_IP, ROBOT_PORT, level=3)
    robot = rc.robot

    # ── 连接单夹爪（可选，失败时继续录制但跳过夹爪采集）──
    dual_gripper: SingleGripperViaZhixing | None = None
    if GRIPPER_HOST:
        try:
            dual_gripper = SingleGripperViaZhixing(
                host=GRIPPER_HOST,
                tcp_port=GRIPPER_TCP_PORT,
                modbus_port=GRIPPER_MODBUS_PORT,
                device_id=GRIPPER_DEVICE_ID,
            )
            dual_gripper.connect()
            print("  ✓ 单夹爪已连接（zhixing TCP 模式），将同步采集夹爪状态")
        except Exception as e:
            dual_gripper = None
            print(f"  ⚠️  单夹爪初始化失败，跳过夹爪采集: {e}")


    # while True:
    #     print("绝对位置：", get_current_pose(robot))

    # ── 记录基准位姿（Affordance）──
    base_pose = list(AFFORDANCE_POSE) if AFFORDANCE_POSE is not None else get_current_pose(robot)
    if base_pose is None:
        print("❌ 无法获取当前位姿，请检查连接后重试。")
        rc.disconnect()
        return
    affordance_pose = base_pose[:]

    print(f"\n✓ Affordance 基准位姿已记录:")
    print(f"  {format_waypoint(base_pose)}")
    print(f"\n操作说明:")
    print(f"  Enter  → 保存当前局部相对位姿为路点")
    print(f"  r      → 重置 affordance 原点与局部坐标轴（按配置：Affordance/当前位姿）")
    print(f"  d      → 删除最后一个路点")
    print(f"  q      → 退出并输出 affordance_pose + waypoints")
    print(f"\n{'─' * 60}")

    saved_waypoints: list[list[float]] = []
    gripper_positions: dict[int, int] = {}   # {waypoint_index: gripper_position}
    stop_flag = threading.Event()
    last_offset = [0.0] * 6
    last_gripper = 0   # 刷新线程写入，主线程读取（GIL 保护 int 赋值足够）

    # ── 实时刷新线程 ──
    def refresh_loop():
        nonlocal last_offset, base_pose, last_gripper
        while not stop_flag.is_set():
            pose = get_current_pose(robot)
            if pose is not None:
                offset = calc_local_offset(robot, base_pose, pose)
                last_offset = offset
                wp_str = format_waypoint(offset)
                count = len(saved_waypoints)

                # 采集夹爪状态
                g_str = ""
                if dual_gripper is not None:
                    try:
                        g = dual_gripper.get_position()
                        last_gripper = g
                        g_str = f"  夹爪=[{g}]"
                    except Exception:
                        g_str = "  夹爪=[?]"

                clear_line()
                sys.stdout.write(
                    f"  实时相对位姿(affordance局部): {wp_str}{g_str}  "
                    f"[已保存 {count} 个路点]  "
                    f"| Enter保存 r重置 d删除 q退出"
                )
                sys.stdout.flush()
            time.sleep(1.0 / REFRESH_HZ)

    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    # ── 主循环：处理键盘输入 ──
    try:
        with SingleKeyReader() as key_reader:
            while True:
                cmd = key_reader.read_key(stop_flag)
                if cmd is None:
                    stop_flag.set()
                    break

                if cmd == "q":
                    stop_flag.set()
                    break

                elif cmd == "r":
                    new_base = None
                    if RESET_BASE_TO_AFFORDANCE_ON_R and AFFORDANCE_POSE is not None:
                        new_base = list(AFFORDANCE_POSE)
                    else:
                        new_base = get_current_pose(robot)
                    if new_base is not None:
                        base_pose = new_base
                        affordance_pose = new_base[:]
                        clear_line()
                        print(f"\n  ✓ Affordance 原点与局部坐标轴已重置为: {format_waypoint(base_pose)}")
                    else:
                        clear_line()
                        print("\n  ⚠️  重置失败，无法获取当前位姿")

                elif cmd == "d":
                    if saved_waypoints:
                        last_idx = len(saved_waypoints) - 1
                        removed = saved_waypoints.pop()
                        gripper_positions.pop(last_idx, None)
                        clear_line()
                        print(f"\n  🗑  已删除路点 {last_idx+1}: {format_waypoint(removed)}")
                    else:
                        clear_line()
                        print("\n  ⚠️  没有可删除的路点")

                elif cmd == "\n":
                    # Enter → 保存当前局部相对位姿及夹爪状态
                    wp = last_offset[:]
                    idx = len(saved_waypoints)
                    saved_waypoints.append(wp)
                    g_info = ""
                    if dual_gripper is not None:
                        gv = int(last_gripper)
                        gripper_positions[idx] = gv
                        g_info = f"  夹爪=[{gv}]"
                    clear_line()
                    print(f"\n  ✓ 路点 {idx+1} 已保存: {format_waypoint(wp)}{g_info}")

    except KeyboardInterrupt:
        stop_flag.set()
        print("\n\n  (Ctrl+C 中断)")

    stop_flag.set()

    # ── 输出结果 ──
    if saved_waypoints:
        print_saved_waypoints(affordance_pose, saved_waypoints, gripper_positions)
    else:
        print("\n  未保存任何路点。")

    if dual_gripper is not None:
        dual_gripper.disconnect()
    rc.disconnect()


if __name__ == "__main__":
    main()
