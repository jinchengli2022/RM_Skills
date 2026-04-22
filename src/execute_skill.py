import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.skills import execute_skill, SkillExecutor
from core.demo_project import RobotArmController
from core.dual_gripper import DualGripper, DualGripperConfig

# ── 双夹爪串口配置（不使用夹爪时将两个端口均设为 None）──────────────────────
GRIPPER_PORT1 = "/dev/ttyUSB1"   # 夹爪 1 串口
GRIPPER_PORT2 = "/dev/ttyUSB2"   # 夹爪 2 串口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 连接双夹爪（可选，失败时跳过夹爪控制）──
    dual_gripper: DualGripper | None = None
    if GRIPPER_PORT1 and GRIPPER_PORT2:
        try:
            cfg = DualGripperConfig(port1=GRIPPER_PORT1, port2=GRIPPER_PORT2)
            dual_gripper = DualGripper(cfg)
            dual_gripper.connect()
        except Exception as e:
            dual_gripper = None
            print(f"[execute_skill] ⚠️  双夹爪初始化失败，将跳过夹爪控制: {e}")

    # ── 执行技能 ──
    # execute_skill() 是便捷封装；如需传入夹爪，直接使用 SkillExecutor
    ROBOT_IP   = "169.254.128.19"
    ROBOT_PORT = 8080
    SKILL_NAME = "set_kettle"
    # 外部传入的 > 技能内置的，确保技能内不覆盖这个基准位姿
    AFFORDANCE_POSE = [0.0900, 0.3763, -0.1825, 3.0800, 0.1120, -1.8970]
    REF_POSE = [0.0900, 0.3763, -0.1825, 3.0800, 0.1120, -1.8970]

    rc = RobotArmController(ROBOT_IP, ROBOT_PORT, level=3)
    try:
        executor = SkillExecutor(rc, dual_gripper=dual_gripper)
        executor.execute(SKILL_NAME,  affordance_pose=AFFORDANCE_POSE, ref_pose=REF_POSE)
    finally:
        if dual_gripper is not None:
            dual_gripper.disconnect()
        rc.disconnect()








