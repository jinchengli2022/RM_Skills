"""
双夹爪控制适配层 (Dual Gripper Adapter)

封装两个基于 Modbus RTU/RS-485 的伺服夹爪（CTAG2F90D），
使用 grasp_resource/sdk/changingtek_p_rtu_Servo.py 中的 MotorController，
严禁调用 src/Robotic_Arm/ 目录下任何夹爪驱动接口。

用法示例:
    from src.core.dual_gripper import DualGripper, DualGripperConfig

    cfg = DualGripperConfig(
        port1="/dev/ttyUSB1",
        port2="/dev/ttyUSB2",
    )
    dg = DualGripper(cfg)
    dg.connect()

    # 读取当前位置
    g1, g2 = dg.get_positions()

    # 设置双夹爪位置（阻塞直到到位或超时）
    ok = dg.set_positions(g1=0, g2=9000)

    dg.disconnect()
"""

from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# 只从 grasp_resource SDK 引入，不使用 src 目录下任何夹爪驱动
_SDK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'grasp_resource', 'sdk')
)
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from changingtek_p_rtu_Servo import MotorController  # noqa: E402


# ===========================================================================
# 配置数据类
# ===========================================================================

@dataclass
class DualGripperConfig:
    """双夹爪硬件与运动参数配置。

    Attributes:
        port1: 夹爪 1 的串口路径（Linux: /dev/ttyUSBx，Windows: COMx）
        port2: 夹爪 2 的串口路径
        slave_id1: 夹爪 1 的 Modbus 从站地址（默认 1）
        slave_id2: 夹爪 2 的 Modbus 从站地址（默认 1）
        baudrate: 串口波特率（固定 115200）
        timeout: 串口读写超时（秒）
        speed_pct: 默认运动速度百分比
        force_pct: 默认夹紧力百分比
        accel: 默认加速度
        decel: 默认减速度
        position_tolerance: 判断"到位"的位置容差（与 read_real_position 单位一致）
        reach_timeout: 等待夹爪到位的最长时间（秒）
        poll_interval: 到位轮询间隔（秒）
        open_pos: 夹爪全开对应的目标位置值
        close_pos: 夹爪全闭对应的目标位置值
    """
    port1: str = "/dev/ttyUSB1"
    port2: str = "/dev/ttyUSB2"
    slave_id1: int = 1
    slave_id2: int = 1
    baudrate: int = 115200
    timeout: float = 1.0
    speed_pct: int = 50
    force_pct: int = 25
    accel: int = 60
    decel: int = 60
    position_tolerance: int = 80
    reach_timeout: float = 5.0
    poll_interval: float = 0.05
    open_pos: int = 0
    close_pos: int = 9000


# ===========================================================================
# 双夹爪控制器
# ===========================================================================

class DualGripper:
    """
    双夹爪控制器，封装两只独立串口的 MotorController 实例。

    线程安全：串口操作由 MotorController 内部 Lock 保护；
    双夹爪命令先后快速下发，再统一轮询到位，减少等待时间。
    """

    def __init__(self, config: DualGripperConfig = None):
        self.cfg = config or DualGripperConfig()
        self._g1: MotorController | None = None
        self._g2: MotorController | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """初始化并连接两只夹爪串口。失败时抛出异常。"""
        try:
            self._g1 = MotorController(
                self.cfg.port1, self.cfg.slave_id1,
                self.cfg.baudrate, self.cfg.timeout
            )
            print(f"[DualGripper] 夹爪 1 已连接: {self.cfg.port1} (slave={self.cfg.slave_id1})")
        except Exception as e:
            raise RuntimeError(f"[DualGripper] 夹爪 1 连接失败 ({self.cfg.port1}): {e}") from e

        try:
            self._g2 = MotorController(
                self.cfg.port2, self.cfg.slave_id2,
                self.cfg.baudrate, self.cfg.timeout
            )
            print(f"[DualGripper] 夹爪 2 已连接: {self.cfg.port2} (slave={self.cfg.slave_id2})")
        except Exception as e:
            self._g1 = None
            raise RuntimeError(f"[DualGripper] 夹爪 2 连接失败 ({self.cfg.port2}): {e}") from e

        self._connected = True

    def disconnect(self) -> None:
        """释放串口资源（MotorController 本身无显式 close，置 None 触发 GC）。"""
        self._g1 = None
        self._g2 = None
        self._connected = False
        print("[DualGripper] 已断开双夹爪连接")

    def is_connected(self) -> bool:
        return self._connected and self._g1 is not None and self._g2 is not None

    # ------------------------------------------------------------------
    # 读取状态
    # ------------------------------------------------------------------

    def get_positions(self) -> tuple[int, int]:
        """读取两只夹爪的当前实时位置。

        Returns:
            (g1_pos, g2_pos)

        Raises:
            RuntimeError: 未连接或读取失败。
        """
        self._check_connected()
        try:
            p1 = self._g1.read_real_position()
        except Exception as e:
            raise RuntimeError(f"[DualGripper] 读取夹爪 1 位置失败: {e}") from e
        try:
            p2 = self._g2.read_real_position()
        except Exception as e:
            raise RuntimeError(f"[DualGripper] 读取夹爪 2 位置失败: {e}") from e
        return p1, p2

    def get_full_state(self) -> dict:
        """读取两只夹爪的位置、速度、电流。

        Returns:
            dict with keys: g1_pos, g1_speed, g1_current, g2_pos, g2_speed, g2_current
        """
        self._check_connected()
        state = {}
        try:
            state["g1_pos"] = self._g1.read_real_position()
            state["g1_speed"] = self._g1.read_real_speed()
            state["g1_current"] = self._g1.read_real_current()
        except Exception as e:
            raise RuntimeError(f"[DualGripper] 读取夹爪 1 状态失败: {e}") from e
        try:
            state["g2_pos"] = self._g2.read_real_position()
            state["g2_speed"] = self._g2.read_real_speed()
            state["g2_current"] = self._g2.read_real_current()
        except Exception as e:
            raise RuntimeError(f"[DualGripper] 读取夹爪 2 状态失败: {e}") from e
        return state

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------

    def set_positions(
        self,
        g1: int,
        g2: int,
        wait: bool = True,
        speed_pct: int | None = None,
        force_pct: int | None = None,
        accel: int | None = None,
        decel: int | None = None,
    ) -> bool:
        """向两只夹爪同时下发目标位置命令，可选等待到位。

        先后快速下发两路命令（无需等待），再统一轮询到位，减少总耗时。

        Args:
            g1: 夹爪 1 目标位置
            g2: 夹爪 2 目标位置
            wait: 是否等待到位（True=阻塞轮询，False=仅下发指令）
            speed_pct/force_pct/accel/decel: 覆盖配置默认值

        Returns:
            True 表示成功（或非阻塞时直接返回 True），
            False 表示超时未到位或通信异常（不抛出，由调用者判断中止策略）。
        """
        self._check_connected()
        sp = speed_pct if speed_pct is not None else self.cfg.speed_pct
        fp = force_pct if force_pct is not None else self.cfg.force_pct
        ac = accel if accel is not None else self.cfg.accel
        dc = decel if decel is not None else self.cfg.decel

        # 快速下发两路命令
        try:
            self._g1.temp_move(g1, sp, fp, ac, dc, trigger=True)
        except Exception as e:
            print(f"[DualGripper] ❌ 夹爪 1 下发失败 (目标={g1}): {e}")
            return False
        try:
            self._g2.temp_move(g2, sp, fp, ac, dc, trigger=True)
        except Exception as e:
            print(f"[DualGripper] ❌ 夹爪 2 下发失败 (目标={g2}): {e}")
            return False

        if not wait:
            return True

        return self._wait_until_reached(g1, g2)

    def _wait_until_reached(self, target_g1: int, target_g2: int) -> bool:
        """轮询双夹爪位置直到到位或超时。

        Returns:
            True = 双夹爪均到位，False = 超时或读取失败。
        """
        deadline = time.time() + self.cfg.reach_timeout
        tol = self.cfg.position_tolerance
        while time.time() < deadline:
            try:
                p1 = self._g1.read_real_position()
                p2 = self._g2.read_real_position()
            except Exception as e:
                print(f"[DualGripper] ⚠️  到位轮询读取失败: {e}")
                return False
            ok1 = abs(p1 - target_g1) <= tol
            ok2 = abs(p2 - target_g2) <= tol
            if ok1 and ok2:
                print(f"[DualGripper] ✓ 双夹爪到位: g1={p1}, g2={p2}")
                return True
            time.sleep(self.cfg.poll_interval)

        # 超时时读最后一次位置打印诊断
        try:
            p1 = self._g1.read_real_position()
            p2 = self._g2.read_real_position()
        except Exception:
            p1, p2 = "?", "?"
        print(f"[DualGripper] ⏱ 到位超时 (t={self.cfg.reach_timeout}s) "
              f"| 目标 g1={target_g1} g2={target_g2} "
              f"| 当前 g1={p1} g2={p2}")
        return False

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _check_connected(self) -> None:
        if not self.is_connected():
            raise RuntimeError("[DualGripper] 夹爪未连接，请先调用 connect()")
