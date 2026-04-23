from __future__ import annotations

import sys
import os
import json
import socket
import time

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.skills import execute_skill, SkillExecutor
from core.demo_project import RobotArmController
from core.zhixing import GripperController

# ── 单夹爪 TCP 配置（基于 zhixing.py 的 GripperController）────────────────────
# 不使用夹爪时将 GRIPPER_HOST 设为 None

# ─────────────────────────────────────────────────────────────────────────────


class SingleGripperViaZhixing:
    """适配 SkillExecutor 所需接口：connect/open_gripper/close_gripper/disconnect。"""

    def __init__(self, host: str, tcp_port: int, modbus_port: int, device_id: int):
        self.host = host
        self.tcp_port = tcp_port
        self.modbus_port = modbus_port
        self.device_id = device_id
        self._sock: socket.socket | None = None
        self._gripper: GripperController | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.tcp_port))

        setup_cmd = {
            "command": "set_modbus_mode",
            "port": self.modbus_port,
            "baudrate": 115200,
            "timeout": 3,
        }
        payload = json.dumps(setup_cmd) + "\n"
        self._sock.sendall(payload.encode("utf-8"))
        self._sock.recv(1024)

        self._gripper = GripperController(self._sock, device_id=self.device_id, port=self.modbus_port)
        #设置夹爪力度
        # self._gripper.set_gipper_force(100, 1)
        # time.sleep(1)
        # setup_cmd = {"command": "read_holding_registers", "port": 1, "address": 261, "device": 1}
        # self._sock.sendall(json.dumps(setup_cmd).encode('utf-8'))
        # print("Modbus mode setup response:", self._sock.recv(1024).decode())
        # self._gripper.open_gripper(delay=1.0)  # 打开夹爪
        # self._gripper.close_gripper(delay=1.0)  # 打开夹爪
        # self._gripper.open_gripper(delay=1.0)  # 打开夹爪
        # self._gripper.close_gripper(delay=1.0)  # 打开夹



    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._gripper = None

    def open_gripper(self, delay: float = 0.2) -> None:
        if self._gripper is None:
            raise RuntimeError("夹爪未连接，请先调用 connect()")
        self._gripper.open_gripper(delay=delay)

    def close_gripper(self, delay: float = 0.2) -> None:
        if self._gripper is None:
            raise RuntimeError("夹爪未连接，请先调用 connect()")
        self._gripper.close_gripper(delay=delay)

if __name__ == "__main__":
    ROBOT_IP   = "169.254.128.19"
    ROBOT_PORT = 8080
    SKILL_NAME = "test"
    GRIPPER_MODBUS_PORT = 1
    GRIPPER_DEVICE_ID = 1
    
    # 连接夹爪
    gripper_executor = SingleGripperViaZhixing(
                host=ROBOT_IP,
                tcp_port=ROBOT_PORT,
                modbus_port=GRIPPER_MODBUS_PORT,
                device_id=GRIPPER_DEVICE_ID,
            )
    gripper_executor.connect()

    # ── 执行技能 ──
    # execute_skill() 是便捷封装；如需传入夹爪，直接使用 SkillExecutor

    # 外部传入的 > 技能内置的，确保技能内不覆盖这个基准位姿
    AFFORDANCE_POSE = None
    REF_POSE = None

    rc = RobotArmController(ROBOT_IP, ROBOT_PORT, level=3)
    try:
        executor = SkillExecutor(rc, dual_gripper=gripper_executor)
        executor.execute(SKILL_NAME,  affordance_pose=AFFORDANCE_POSE, ref_pose=REF_POSE)
    finally:
        if gripper_executor is not None:
            gripper_executor.disconnect()
        rc.disconnect()








