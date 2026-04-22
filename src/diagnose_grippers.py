#!/usr/bin/env python
"""
夹爪串口连接诊断脚本

用途：自动探测夹爪的连接配置
- 扫描系统所有可用串口
- 尝试不同的模式：双串口独立 vs 单串口主从
- 输出诊断结果和推荐配置

使用方法：
    python diagnose_grippers.py
"""

import sys
import os
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.core.dual_gripper import DualGripper, DualGripperConfig

# 尝试import pyserial，用于列举串口
try:
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[警告] serial 库未安装，无法自动列举串口")
    print("        运行: pip install pyserial")


# ===========================================================================
# 串口扫描
# ===========================================================================

def list_serial_ports() -> List[str]:
    """列出系统中所有可用的串口设备。"""
    if not HAS_SERIAL:
        # 手动列举常见串口
        ports = []
        for i in range(10):
            ports.append(f"/dev/ttyUSB{i}")
            ports.append(f"/dev/ttyACM{i}")
            ports.append(f"COM{i+1}")
        return ports
    
    ports = []
    for port_info in serial.tools.list_ports.comports():
        ports.append(port_info.device)
    return sorted(ports)


def filter_available_ports(ports: List[str]) -> List[str]:
    """过滤出实际存在的串口设备。"""
    available = []
    for port in ports:
        try:
            ser = __import__('serial').Serial(port, timeout=0.1)
            ser.close()
            available.append(port)
        except Exception:
            pass
    return available


# ===========================================================================
# 诊断模式
# ===========================================================================

def diagnose_dual_independent(port1: str, port2: str) -> Tuple[bool, str]:
    """
    诊断模式 A: 两个独立串口，各自连接一只夹爪（Slave ID 都是 1）
    
    Returns:
        (success, message)
    """
    print(f"\n[诊断] 模式 A: 双独立串口")
    print(f"       port1={port1} (Slave 1)")
    print(f"       port2={port2} (Slave 1)")
    
    try:
        cfg = DualGripperConfig(
            port1=port1,
            port2=port2,
            slave_id1=1,
            slave_id2=1,
        )
        dg = DualGripper(cfg)
        dg.connect()
        
        # 尝试读取位置
        g1, g2 = dg.get_positions()
        dg.disconnect()
        
        msg = f"✓ 成功！夹爪1位置={g1}, 夹爪2位置={g2}"
        print(f"  {msg}")
        return True, msg
        
    except Exception as e:
        msg = f"✗ 失败: {str(e)[:100]}"
        print(f"  {msg}")
        return False, msg


def diagnose_single_port_dual_slaves(port: str, id1: int = 1, id2: int = 2) -> Tuple[bool, str]:
    """
    诊断模式 B: 单一串口，两个不同 Slave ID（主从级联模式）
    
    Args:
        port: 串口路径
        id1: 第一个夹爪的 Slave ID
        id2: 第二个夹爪的 Slave ID
    
    Returns:
        (success, message)
    """
    print(f"\n[诊断] 模式 B: 单串口主从级联")
    print(f"       port={port}")
    print(f"       Slave 1 ID={id1}, Slave 2 ID={id2}")
    
    try:
        cfg = DualGripperConfig(
            port1=port,
            port2=port,  # 相同串口
            slave_id1=id1,
            slave_id2=id2,
        )
        dg = DualGripper(cfg)
        dg.connect()
        
        # 尝试读取位置
        g1, g2 = dg.get_positions()
        dg.disconnect()
        
        msg = f"✓ 成功！夹爪1位置={g1}, 夹爪2位置={g2}"
        print(f"  {msg}")
        return True, msg
        
    except Exception as e:
        msg = f"✗ 失败: {str(e)[:100]}"
        print(f"  {msg}")
        return False, msg


def diagnose_single_gripper(port: str, slave_id: int = 1) -> Tuple[bool, str]:
    """
    诊断模式 C：检查单个串口是否连接了至少一只夹爪
    
    Returns:
        (success, message)
    """
    print(f"\n[诊断] 模式 C: 单串口单夹爪检查")
    print(f"       port={port} (Slave {slave_id})")
    
    try:
        # 使用 minimalmodbus 直接测试
        import minimalmodbus
        instr = minimalmodbus.Instrument(port, slave_id)
        instr.baudrate = 115200
        instr.timeout = 1.0
        
        # 读取位置（寄存器 0x0100）
        pos = instr.read_register(0x0100, number_of_decimals=0)
        instr.close()
        
        msg = f"✓ 成功！夹爪位置={pos}"
        print(f"  {msg}")
        return True, msg
        
    except Exception as e:
        msg = f"✗ 失败: {str(e)[:100]}"
        print(f"  {msg}")
        return False, msg


# ===========================================================================
# 主诊断流程
# ===========================================================================

def main():
    print("=" * 70)
    print("  夹爪连接诊断工具")
    print("=" * 70)
    
    # 步骤 1: 扫描可用串口
    print("\n[步骤 1] 扫描可用串口...")
    all_ports = list_serial_ports()
    print(f"  发现的串口: {all_ports}")
    
    available_ports = filter_available_ports(all_ports)
    if not available_ports:
        print("  ✗ 未发现可用的串口设备")
        print("\n  故障排查:")
        print("    1. 确认 USB 夹爪已插入")
        print("    2. 运行: dmesg | grep -i serial  (查看内核日志)")
        print("    3. 运行: ls -la /dev/ttyUSB*     (列举 USB 串口)")
        return
    
    print(f"  ✓ 发现 {len(available_ports)} 个可用串口: {available_ports}")
    
    # 步骤 2: 尝试诊断模式
    print("\n[步骤 2] 尝试诊断连接模式...")
    print("  " + "-" * 66)
    
    results: Dict[str, List[Tuple[bool, str]]] = {
        "dual_independent": [],
        "single_port_dual_slaves": [],
        "single_gripper": [],
    }
    
    # 模式 A: 双独立串口
    if len(available_ports) >= 2:
        for i in range(len(available_ports) - 1):
            port1 = available_ports[i]
            port2 = available_ports[i + 1]
            success, msg = diagnose_dual_independent(port1, port2)
            results["dual_independent"].append((success, msg, port1, port2))
            if success:
                break
    
    # 模式 B: 单串口主从
    for port in available_ports:
        for id1, id2 in [(1, 2), (1, 3), (2, 1), (3, 1)]:
            success, msg = diagnose_single_port_dual_slaves(port, id1, id2)
            results["single_port_dual_slaves"].append((success, msg, port, id1, id2))
            if success:
                break
        if any(r[0] for r in results["single_port_dual_slaves"]):
            break
    
    # 模式 C: 单夹爪（初步检测）
    if not any(r[0] for r in results["dual_independent"]) and \
       not any(r[0] for r in results["single_port_dual_slaves"]):
        print("\n[信息] 双夹爪模式均失败，尝试检测单夹爪...")
        for port in available_ports:
            for slave_id in [1, 2, 3]:
                success, msg = diagnose_single_gripper(port, slave_id)
                results["single_gripper"].append((success, msg, port, slave_id))
                if success:
                    break
            if any(r[0] for r in results["single_gripper"]):
                break
    
    # 步骤 3: 输出诊断结果
    print("\n" + "=" * 70)
    print("  诊断结果")
    print("=" * 70)
    
    # 检查是否成功
    dual_ind_success = any(r[0] for r in results["dual_independent"])
    single_dual_success = any(r[0] for r in results["single_port_dual_slaves"])
    single_success = any(r[0] for r in results["single_gripper"])
    
    if dual_ind_success:
        # 提取成功的配置
        for r in results["dual_independent"]:
            if r[0]:
                _, msg, port1, port2 = r
                print(f"\n✓ 诊断成功！")
                print(f"\n  推荐配置（modes/dual_independent）:")
                print(f"    GRIPPER_PORT1 = '{port1}'")
                print(f"    GRIPPER_PORT2 = '{port2}'")
                print(f"    SLAVE_ID_1 = 1")
                print(f"    SLAVE_ID_2 = 1")
                print(f"\n  配置方法:")
                print(f"    编辑 src/execute_skill.py 中的这些常量即可")
                break
    
    elif single_dual_success:
        for r in results["single_port_dual_slaves"]:
            if r[0]:
                _, msg, port, id1, id2 = r
                print(f"\n✓ 诊断成功！")
                print(f"\n  推荐配置（mode: single_port_dual_slaves）:")
                print(f"    GRIPPER_PORT1 = '{port}'")
                print(f"    GRIPPER_PORT2 = '{port}'")
                print(f"    SLAVE_ID_1 = {id1}")
                print(f"    SLAVE_ID_2 = {id2}")
                print(f"\n  配置方法:")
                print(f"    编辑 src/execute_skill.py 中的这些常量即可")
                break
    
    elif single_success:
        for r in results["single_gripper"]:
            if r[0]:
                _, msg, port, slave_id = r
                print(f"\n⚠ 仅检测到单只夹爪！")
                print(f"\n  检测到的夹爪:")
                print(f"    port={port}, slave_id={slave_id}")
                print(f"\n  故障排查:")
                print(f"    1. 检查第二只夹爪是否已正确连接")
                print(f"    2. 确认硬件 Modbus 地址配置")
                print(f"    3. 尝试手动设置常量进行测试")
                break
    
    else:
        print(f"\n✗ 诊断失败：未能连接任何夹爪")
        print(f"\n  故障排查步骤:")
        print(f"    1. 确认 USB 夹爪已插入并上电")
        print(f"    2. 运行命令查看串口:")
        print(f"       $ ls -la /dev/ttyUSB*")
        print(f"       $ dmesg | tail -20")
        print(f"    3. 检查串口权限:")
        print(f"       $ sudo chmod 666 /dev/ttyUSB*")
        print(f"    4. 尝试手动列举可用串口:")
        print(f"       $ python -c 'import serial; print([p.device for p in serial.tools.list_ports.comports()])'")
        print(f"    5. 检查硬件连接和波特率（应为 115200）")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
