"""
技能点位采集工具 (Skill Waypoint Recorder)

启动后记录基准位姿，然后实时输出当前位置相对于基准的偏移量，
格式与 skills.py 的 waypoints 完全一致，可直接复制粘贴使用。

用法:
    python src/core/get_skill.py

操作说明:
    - 启动后自动记录当前位姿为基准点（原点）
    - 手动拖动机械臂到目标位置（需先开启拖动示教模式）
    - 实时终端输出相对偏移，格式为 [dx, dy, dz, drx, dry, drz]
    - 按 Enter 键保存当前位置为一个路点
    - 按 q + Enter 退出并打印完整的 waypoints 列表
    - 按 r + Enter 重置基准点为当前位置
    - 按 d + Enter 删除最后一个已保存的路点
"""

import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Robotic_Arm.rm_robot_interface import *
from src.core.demo_project import RobotArmController

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROBOT_IP   = "169.254.128.19"
ROBOT_PORT = 8080
REFRESH_HZ = 10          # 实时刷新频率（次/秒）
PRINT_DIGITS = 4         # 输出小数位数
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


def calc_offset(base: list[float], current: list[float]) -> list[float]:
    """计算相对于基准的偏移量"""
    return [round(current[i] - base[i], PRINT_DIGITS) for i in range(6)]


def format_waypoint(offset: list[float]) -> str:
    """格式化为 skills.py waypoint 格式"""
    parts = ", ".join(f"{v:.{PRINT_DIGITS}f}" for v in offset)
    return f"[{parts}]"


def clear_line():
    """清除当前行"""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def print_saved_waypoints(waypoints: list[list[float]]):
    """打印已保存的路点列表"""
    print("\n" + "=" * 60)
    print("  已保存的 waypoints（可直接粘贴到 skills.py）:")
    print("=" * 60)
    print("  \"waypoints\": [")
    for i, wp in enumerate(waypoints):
        comma = "," if i < len(waypoints) - 1 else ""
        print(f"      {format_waypoint(wp)}{comma}  # 路点 {i+1}")
    print("  ],")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  技能点位采集工具 (Skill Waypoint Recorder)")
    print("=" * 60)

    # ── 连接机械臂 ──
    print(f"\n正在连接机械臂 {ROBOT_IP}:{ROBOT_PORT} ...")
    rc = RobotArmController(ROBOT_IP, ROBOT_PORT, level=3)
    robot = rc.robot


    # while True:
    #     print("绝对位置：", get_current_pose(robot))

    # ── 记录基准位姿 ──
    base_pose = [0.090005, 0.376255, -0.182519, 3.08, 0.112, -1.897]
    if base_pose is None:
        print("❌ 无法获取当前位姿，请检查连接后重试。")
        rc.disconnect()
        return

    print(f"\n✓ 基准位姿已记录（原点）:")
    print(f"  {format_waypoint(base_pose)}")
    print(f"\n操作说明:")
    print(f"  Enter      → 保存当前偏移为路点")
    print(f"  r + Enter  → 重置基准点为当前位置")
    print(f"  d + Enter  → 删除最后一个路点")
    print(f"  q + Enter  → 退出并输出完整 waypoints")
    print(f"\n{'─' * 60}")

    saved_waypoints: list[list[float]] = []
    stop_flag = threading.Event()
    last_offset = [0.0] * 6

    # ── 实时刷新线程 ──
    def refresh_loop():
        nonlocal last_offset, base_pose
        while not stop_flag.is_set():
            pose = get_current_pose(robot)
            if pose is not None:
                offset = calc_offset(base_pose, pose)
                last_offset = offset
                wp_str = format_waypoint(offset)
                count = len(saved_waypoints)
                clear_line()
                sys.stdout.write(
                    f"  实时偏移: {wp_str}  "
                    f"[已保存 {count} 个路点]  "
                    f"| Enter保存 r重置 d删除 q退出"
                )
                sys.stdout.flush()
            time.sleep(1.0 / REFRESH_HZ)

    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    # ── 主循环：处理键盘输入 ──
    try:
        while True:
            cmd = input()   # 阻塞等待用户输入

            if cmd.strip().lower() == "q":
                stop_flag.set()
                break

            elif cmd.strip().lower() == "r":
                new_base = get_current_pose(robot)
                if new_base is not None:
                    base_pose = new_base
                    clear_line()
                    print(f"\n  ✓ 基准点已重置为: {format_waypoint(base_pose)}")
                else:
                    clear_line()
                    print("\n  ⚠️  重置失败，无法获取当前位姿")

            elif cmd.strip().lower() == "d":
                if saved_waypoints:
                    removed = saved_waypoints.pop()
                    clear_line()
                    print(f"\n  🗑  已删除路点 {len(saved_waypoints)+1}: {format_waypoint(removed)}")
                else:
                    clear_line()
                    print("\n  ⚠️  没有可删除的路点")

            else:
                # 空输入（直接按 Enter）→ 保存当前偏移
                wp = last_offset[:]
                saved_waypoints.append(wp)
                clear_line()
                print(f"\n  ✓ 路点 {len(saved_waypoints)} 已保存: {format_waypoint(wp)}")

    except KeyboardInterrupt:
        stop_flag.set()
        print("\n\n  (Ctrl+C 中断)")

    stop_flag.set()

    # ── 输出结果 ──
    if saved_waypoints:
        print_saved_waypoints(saved_waypoints)
    else:
        print("\n  未保存任何路点。")

    rc.disconnect()


if __name__ == "__main__":
    main()


