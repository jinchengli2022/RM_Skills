"""
CTAG2F90D（Servo）控制器SDK (基于Modbus RTU的RS-485通信)
---------------------------------------------
一个轻量级的Python SDK，用于通过minimalmodbus库控制电机，
具有线程安全、清晰的寄存器访问方式和实时数据监控功能。

依赖安装:
    pip install minimalmodbus pyserial

作者: 知行机器人
"""

import time
import threading
import glob
import minimalmodbus
import serial

# -----------------------------
# 寄存器映射 (保持寄存器，功能码0x03/0x06)
# -----------------------------
# 写入寄存器 (目标值/控制)
REG_TARGET_POS_HIGH       = 0x0102  # 目标位置（高16位）
REG_TARGET_POS_LOW        = 0x0103  # 目标位置（低16位）
REG_TARGET_SPEED          = 0x0104  # 目标速度
REG_TARGET_FORCE          = 0x0105  # 目标力/力矩
REG_TARGET_ACCELERATION   = 0x0106  # 目标加速度
REG_TARGET_DECELERATION   = 0x0107  # 目标减速度
REG_MOTION_TRIGGER        = 0x0108  # 运动触发（0: 空闲, 1: 触发）

# 读取寄存器 (反馈/状态)
REG_REAL_POS_HIGH         = 0x0418  # 实时位置（高16位）
REG_REAL_POS_LOW          = 0x0419  # 实时位置（低16位）
REG_REAL_SPEED            = 0x041A  # 实时速度反馈
REG_REAL_CURRENT          = 0x041B  # 实时电流反馈


class MotorController:
    def __init__(self, port: str, slave_id: int = 1, baudrate: int = 115200, timeout: float = 1.0):
        """初始化与电机控制器的Modbus RTU通信

        参数:
            port: 串口端口（例如，Windows上为'COM3'，Linux上为'/dev/ttyUSB0'）
            slave_id: Modbus从机地址（默认: 1）
            baudrate: 串口波特率（默认: 115200）
            timeout: 读写超时时间（秒，默认: 1.0）
        """
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = timeout
        self.instrument.mode = minimalmodbus.MODE_RTU
        self._lock = threading.Lock()  # 线程锁，确保串口操作线程安全
        self._monitor_running = False  # 监控线程运行标志
        self._monitor_thread = None    # 监控线程对象

    # -----------------------------
    # 低级寄存器操作工具方法
    # -----------------------------
    def _write_register(self, addr: int, value: int) -> None:
        """写入单个16位保持寄存器（Modbus功能码0x06）

        参数:
            addr: 寄存器地址
            value: 要写入的值
        """
        with self._lock:
            self.instrument.write_register(addr, value, functioncode=6)

    def _read_register(self, addr: int) -> int:
        """读取单个16位保持寄存器（Modbus功能码0x03）

        参数:
            addr: 寄存器地址

        返回:
            寄存器的值
        """
        with self._lock:
            return self.instrument.read_register(addr, functioncode=3)

    def _write_registers(self, addr: int, values: list) -> None:
        """写入多个16位保持寄存器（Modbus功能码0x10）

        参数:
            addr: 起始寄存器地址
            values: 要写入的值列表
        """
        with self._lock:
            self.instrument.write_registers(addr, values)

    def _read_registers(self, addr: int, count: int) -> list:
        """读取多个16位保持寄存器（Modbus功能码0x03）

        参数:
            addr: 起始寄存器地址
            count: 要读取的寄存器数量

        返回:
            寄存器值的列表
        """
        with self._lock:
            return self.instrument.read_registers(addr, count, functioncode=3)

    # -----------------------------
    # 目标值/控制写入操作
    # -----------------------------
    def set_target_position(self, position: int) -> None:
        """设置目标位置（组合高低16位寄存器）

        参数:
            position: 目标位置值
        """
        hi = (position >> 16) & 0xFFFF  # 提取高16位
        lo = position & 0xFFFF          # 提取低16位
        self._write_registers(REG_TARGET_POS_HIGH, [hi, lo])

    def set_target_speed(self, speed: int) -> None:
        """设置目标速度

        参数:
            speed: 目标速度值
        """
        self._write_register(REG_TARGET_SPEED, speed)

    def set_target_force(self, force: int) -> None:
        """设置目标力/力矩

        参数:
            force: 目标力/力矩值
        """
        self._write_register(REG_TARGET_FORCE, force)

    def set_target_acceleration(self, acceleration: int) -> None:
        """设置目标加速度

        参数:
            acceleration: 目标加速度值
        """
        self._write_register(REG_TARGET_ACCELERATION, acceleration)

    def set_target_deceleration(self, deceleration: int) -> None:
        """设置目标减速度

        参数:
            deceleration: 目标减速度值
        """
        self._write_register(REG_TARGET_DECELERATION, deceleration)

    def trigger_motion(self) -> None:
        """使用已配置的目标参数触发运动"""
        self._write_register(REG_MOTION_TRIGGER, 1)

    def temp_move(self, position_mm: int, speed_pct: int, force_pct: int,
                  accel: int, decel: int, trigger: bool = True) -> None:
        """临时运动控制方法，一次性设置所有运动参数并可选触发运动

        参数:
            position_mm: 目标位置(毫米)
            speed_pct: 速度百分比
            force_pct: 力/力矩百分比
            accel: 加速度值
            decel: 减速度值
            trigger: 是否立即触发运动，默认为True
        """
        self.set_target_position(position_mm)
        self.set_target_speed(speed_pct)
        self.set_target_force(force_pct)
        self.set_target_acceleration(accel)
        self.set_target_deceleration(decel)

        if trigger:
            self.trigger_motion()
            print(f"已触发运动到位置 {position_mm}，速度 {speed_pct}%，力 {force_pct}%")

    # -----------------------------
    # 反馈/状态读取操作
    # -----------------------------
    def read_real_position(self) -> int:
        """读取实时位置（组合高低16位寄存器）

        返回:
            组合后的32位实时位置（处理有符号值）
        """
        regs = self._read_registers(REG_REAL_POS_HIGH, 2)
        hi, lo = regs[0], regs[1]
        combined = (hi << 16) | lo
        # 处理32位有符号整数
        if combined & 0x80000000:  # 最高位为1表示负数
            combined = combined - 0x100000000
        return combined

    def read_real_speed(self) -> int:
        """读取实时速度反馈

        返回:
            实时速度值
        """
        return self._read_register(REG_REAL_SPEED)

    def read_real_current(self) -> int:
        """读取实时电流反馈

        返回:
            实时电流值
        """
        return self._read_register(REG_REAL_CURRENT)

    # -----------------------------
    # 实时监控线程
    # -----------------------------
    def _monitor_loop(self, interval: float = 0.5) -> None:
        """实时数据监控的后台循环"""
        while self._monitor_running:
            try:
                pos = self.read_real_position()
                speed = self.read_real_speed()
                current = self.read_real_current()
                print(f"[实时数据] 位置: {pos:6d}, 速度: {speed:3d}, 电流: {current:3d}")
            except Exception as e:
                print(f"监控错误: {e}")
            time.sleep(interval)

    def start_monitoring(self, interval: float = 0.01) -> None:
        """启动实时数据监控线程

        参数:
            interval: 监控间隔（秒，默认: 0.01）
        """
        if not self._monitor_running:
            self._monitor_running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,)
            )
            self._monitor_thread.daemon = True  # 守护线程，主程序退出时自动结束
            self._monitor_thread.start()
            print("实时监控已启动。")

    def stop_monitoring(self) -> None:
        """停止实时数据监控线程"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)  # 等待线程结束
        print("实时监控已停止。")


if __name__ == "__main__":
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    slave_ids = range(1, 11)

    if not ports:
        print("未发现 /dev/ttyUSB* 串口，请检查 USB 转 RS485 连接。")
        raise SystemExit(1)

    print("开始扫描可通信组合...")
    print(f"端口列表: {ports}")
    print(f"从机 ID 范围: {slave_ids.start}~{slave_ids.stop - 1}")

    hits = []
    for port in ports:
        for slave_id in slave_ids:
            sdk = None
            try:
                sdk = MotorController(port, slave_id, baudrate=115200, timeout=0.5)
                pos = sdk.read_real_position()
                print(f"[HIT] port={port}, slave_id={slave_id}, real_pos={pos}")
                hits.append((port, slave_id, pos))
            except Exception as e:
                print(f"[MISS] port={port}, slave_id={slave_id}, err={e}")
            finally:
                if sdk is not None:
                    try:
                        sdk.instrument.serial.close()
                    except Exception:
                        pass

    print("\n扫描完成。")
    if hits:
        print("可通信组合如下:")
        for port, slave_id, pos in hits:
            print(f"  - {port}, slave_id={slave_id}, real_pos={pos}")
    else:
        print("未找到任何可通信组合。")
