from __future__ import annotations

import argparse
import sys
import os
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None
try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional runtime dependency
    rs = None

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.skills import execute_skill, SkillExecutor
from core.demo_project import RobotArmController
from core.zhixing import GripperController
from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
from src.align.test_reconstruct_and_register import reconstruct_and_register_grasp_pose


def load_camera_to_gripper_transform(handeye_path: str | Path) -> np.ndarray:
    handeye_file = Path(handeye_path)
    if not handeye_file.exists():
        raise FileNotFoundError(f"手眼标定结果不存在: {handeye_file}")

    data = json.loads(handeye_file.read_text(encoding="utf-8"))
    transform = np.asarray(data["result"]["camera_to_gripper_transform"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"camera_to_gripper_transform 形状异常: {transform.shape}")
    return transform


def get_current_pose(robot_controller: RobotArmController) -> list[float]:
    ret, state = robot_controller.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂当前位姿失败，错误码: {ret}")
    pose = state.get("pose")
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"机械臂返回的 pose 格式异常: {state}")
    return [float(v) for v in pose]


def matrix_from_rm_pose(robot_controller: RobotArmController, pose: list[float]) -> np.ndarray:
    rm_matrix = robot_controller.robot.rm_algo_pos2matrix(pose)
    values = [float(v) for v in rm_matrix.data]
    return np.array(values, dtype=np.float64).reshape(4, 4)


def rm_pose_from_matrix(robot_controller: RobotArmController, matrix: np.ndarray) -> list[float]:
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot_controller.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(v) for v in pose]


def compute_affordance_pose_in_base(
    robot_controller: RobotArmController,
    camera_to_gripper_transform: np.ndarray,
    camera_affordance_pose: np.ndarray,
) -> list[float]:
    current_gripper_pose = get_current_pose(robot_controller)
    base_to_gripper_transform = matrix_from_rm_pose(robot_controller, current_gripper_pose)
    gripper_to_camera_transform = np.linalg.inv(camera_to_gripper_transform)
    base_to_affordance_transform = base_to_gripper_transform @ gripper_to_camera_transform @ camera_affordance_pose
    return rm_pose_from_matrix(robot_controller, base_to_affordance_transform)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行技能，并可选实时可视化 affordance_pose 与当前夹爪姿态")
    parser.add_argument("--robot-ip", default="169.254.128.18", help="机械臂 IP")
    parser.add_argument("--robot-port", type=int, default=8080, help="机械臂端口")
    parser.add_argument("--skill-name", default="goto_affordance", help="技能名称")
    parser.add_argument("--camera-serial", default="348522075148", help="RealSense 相机序列号")
    parser.add_argument(
        "--source-ply",
        type=Path,
        default=Path("/home/rm/ljc/RM_Skills/outputs/kettle_test5/point_cloud.ply"),
        help="带 label 的 source 点云路径",
    )
    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("/home/rm/ljc/RM_Skills/handeye_output/handeye_result.json"),
        help="手眼标定结果 JSON 路径",
    )
    parser.add_argument("--vis", action="store_true", help="实时显示相机深度图及 affordance 投影，执行前等待确认")
    return parser.parse_args()


def _format_pose(pose: list[float] | None) -> str:
    if pose is None:
        return "None"
    return "[" + ", ".join(f"{value:.4f}" for value in pose) + "]"


def run_depth_display(
    camera_serial: str,
    camera_affordance_pose: np.ndarray,
    stop_event: threading.Event,
) -> None:
    """实时显示彩色图（左）和深度图（右），并将 affordance 3D 坐标（相机系）投影到彩色图上。
    按 q 键可提前停止。
    """
    if cv2 is None:
        print("[depth_display] 未安装 opencv-python，跳过深度图显示")
        return
    if rs is None:
        print("[depth_display] 未安装 pyrealsense2，跳过深度图显示")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camera_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    for _ in range(5):  # 预热
        pipeline.wait_for_frames()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy, cx_i, cy_i = intr.fx, intr.fy, intr.ppx, intr.ppy

    # 将 affordance 原点（相机坐标系）投影到像素坐标
    pt = camera_affordance_pose[:3, 3]
    aff_px = (int(fx * pt[0] / pt[2] + cx_i), int(fy * pt[1] / pt[2] + cy_i))

    # 姿态轴端点（相机坐标系，长 5 cm）
    rot = camera_affordance_pose[:3, :3]
    axis_tips = [rot[:, i] * 0.05 + pt for i in range(3)]
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X红 Y绿 Z蓝

    def project(p3d: np.ndarray):
        if p3d[2] <= 0:
            return None
        return (int(fx * p3d[0] / p3d[2] + cx_i), int(fy * p3d[1] / p3d[2] + cy_i))

    align = rs.align(rs.stream.color)

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
            )

            # 投影点 + 圆圈标记
            cv2.drawMarker(color_img, aff_px, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(color_img, aff_px, 12, (0, 255, 0), 2)
            label = f"Aff ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}m)"
            cv2.putText(color_img, label, (aff_px[0] + 14, aff_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 姿态轴箭头
            for tip_3d, color in zip(axis_tips, axis_colors):
                tip_2d = project(tip_3d)
                if tip_2d is not None:
                    cv2.arrowedLine(color_img, aff_px, tip_2d, color, 2, tipLength=0.3)

            # 深度图也标注位置
            cv2.drawMarker(depth_colormap, aff_px, (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

            combined = np.hstack([color_img, depth_colormap])
            cv2.imshow("Color + Depth | Affordance Projection", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

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

def main() -> None:
    args = parse_args()
    gripper_modbus_port = 1
    gripper_device_id = 1

    gripper_executor = SingleGripperViaZhixing(
                host=args.robot_ip,
                tcp_port=args.robot_port,
                modbus_port=gripper_modbus_port,
                device_id=gripper_device_id,
            )

    # ── 执行技能 ──
    # execute_skill() 是便捷封装；如需传入夹爪，直接使用 SkillExecutor

    # 外部传入的 > 技能内置的，确保技能内不覆盖这个基准位姿
    AFFORDANCE_POSE = None
    REF_POSE = None

    rc = RobotArmController(args.robot_ip, args.robot_port, level=3)

    # 连接夹爪（在 rc 创建之后，纳入 try-finally 保护范围之内）
    gripper_executor.connect()

    try:
        camera_affordance_pose = reconstruct_and_register_grasp_pose(
            source_ply=args.source_ply,
            camera_serial=args.camera_serial,
        )
        print("camera affordance pose matrix:")
        print(json.dumps(camera_affordance_pose.tolist(), indent=2, ensure_ascii=False))

        camera_to_gripper_transform = load_camera_to_gripper_transform(args.handeye_result_path)
        AFFORDANCE_POSE = compute_affordance_pose_in_base(
            robot_controller=rc,
            camera_to_gripper_transform=camera_to_gripper_transform,
            camera_affordance_pose=camera_affordance_pose,
        )

        print(f"base affordance_pose: {AFFORDANCE_POSE}")

        executor = SkillExecutor(rc, dual_gripper=gripper_executor)
        error_holder: dict[str, BaseException | None] = {"error": None}

        def run_skill() -> None:
            try:
                executor.execute(args.skill_name, affordance_pose=AFFORDANCE_POSE, ref_pose=REF_POSE)
            except BaseException as exc:
                error_holder["error"] = exc

        if args.vis:
            depth_stop = threading.Event()
            depth_thread = threading.Thread(
                target=run_depth_display,
                kwargs={
                    "camera_serial": args.camera_serial,
                    "camera_affordance_pose": camera_affordance_pose,
                    "stop_event": depth_stop,
                },
                daemon=True,
            )
            depth_thread.start()
            print("深度图可视化已启动（左=彩色+Affordance投影，右=深度图）。")
            print("确认 affordance 位置后，在终端按 Enter 开始执行技能...")
            input()
            depth_stop.set()
            depth_thread.join(timeout=2.0)

        run_skill()

        if error_holder["error"] is not None:
            raise error_holder["error"]
    finally:
        if gripper_executor is not None:
            gripper_executor.disconnect()
        rc.disconnect()


if __name__ == "__main__":
    main()
