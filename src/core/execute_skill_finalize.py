from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_project import RobotArmController
from labeling import (
    CameraIntrinsics,
    build_object_point_cloud,
    build_selection_from_arrays,
    load_base_to_camera_transform,
    make_ee_to_gripper_transform,
    require_cv2,
    require_open3d,
    require_pyvista,
    show_registration_pyvista_preview,
)
from skills import SkillExecutor
from zhixing import GripperController
from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
from src.align.ply_registration import register_point_clouds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize execute-skill session and run the matched skill.")
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


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


def rm_pose_from_matrix(robot_controller: RobotArmController, matrix: np.ndarray) -> list[float]:
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot_controller.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(v) for v in pose]


def compute_gripper_tcp_and_ee_pose_in_base(
    robot_controller: RobotArmController,
    base_to_camera_transform: np.ndarray,
    camera_gripper_tcp_pose: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    ee_to_gripper_tcp_transform = make_ee_to_gripper_transform()
    base_to_gripper_tcp_transform = base_to_camera_transform @ camera_gripper_tcp_pose
    base_to_ee_target_transform = base_to_gripper_tcp_transform @ np.linalg.inv(ee_to_gripper_tcp_transform)
    return base_to_gripper_tcp_transform, rm_pose_from_matrix(robot_controller, base_to_ee_target_transform)


def grasp_result_to_camera_pose_matrix(grasp_result: dict[str, object]) -> np.ndarray:
    rotation = np.asarray(grasp_result["gripper_rotation_matrix"], dtype=np.float64)
    translation = np.asarray(grasp_result["gripper_contact_center"], dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"gripper_rotation_matrix 形状异常: {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"gripper_contact_center 形状异常: {translation.shape}")
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation
    return pose_matrix


def wait_for_execute_confirmation() -> None:
    while True:
        user_input = input("输入 confirm 执行技能，或 Ctrl-C 取消: ").strip().lower()
        if user_input == "confirm":
            return
        print("未检测到 confirm，当前不会执行技能。")


def main() -> None:
    args = parse_args()
    require_cv2()
    require_open3d()

    session_dir = args.session_dir.resolve()
    manifest_path = session_dir / "session.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"session.json 不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_dir = Path(manifest["source_dir"]).resolve()
    source_ply = source_dir / "point_cloud.ply"
    source_label = source_dir / "label.json"
    if not source_dir.exists():
        raise FileNotFoundError(f"source 目录不存在: {source_dir}")
    if not source_ply.exists():
        raise FileNotFoundError(f"source 点云不存在: {source_ply}")
    if not source_label.exists():
        raise FileNotFoundError(f"source label 不存在: {source_label}")

    color_bgr = cv2.imread(str(session_dir / "capture_color.png"), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError(f"读取临时彩色图失败: {session_dir / 'capture_color.png'}")
    depth_image = np.load(session_dir / "capture_depth.npy")
    mask_image = cv2.imread(str(session_dir / "session_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise RuntimeError(f"读取临时 SAM mask 失败: {session_dir / 'session_mask.png'}")

    intr = manifest["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        width=int(intr["width"]),
        height=int(intr["height"]),
    )
    prompt = manifest["prompt_pixel"]
    selection = build_selection_from_arrays(
        color_bgr=color_bgr,
        depth_image=depth_image,
        intrinsics=intrinsics,
        depth_scale=float(manifest["depth_scale"]),
        mask=mask_image > 0,
        prompt_pixel=(int(prompt[0]), int(prompt[1])),
    )
    point_cloud = build_object_point_cloud(selection)

    import open3d as o3d

    target_ply = session_dir / "target_object_point_cloud.ply"
    if not o3d.io.write_point_cloud(str(target_ply), point_cloud, write_ascii=False):
        raise RuntimeError(f"保存临时目标点云失败: {target_ply}")
    if not target_ply.exists():
        raise RuntimeError(f"临时目标点云未成功落盘: {target_ply}")

    registration_output = register_point_clouds(
        source=source_ply,
        target=target_ply,
        no_vis=True,
        return_registration_summary=True,
    )
    if registration_output is None:
        raise RuntimeError(f"source 目录下缺少可用 grasp label: {source_dir}")
    grasp_result, registration_summary = registration_output
    if grasp_result is None:
        raise RuntimeError(f"source 目录下缺少可用 grasp label: {source_dir}")

    camera_gripper_tcp_pose = grasp_result_to_camera_pose_matrix(grasp_result)
    print("camera gripper TCP pose matrix:")
    print(json.dumps(camera_gripper_tcp_pose.tolist(), indent=2, ensure_ascii=False))

    gripper_modbus_port = 1
    gripper_device_id = 1
    rc = RobotArmController(str(manifest["robot_ip"]), int(manifest["robot_port"]), level=3)
    gripper_executor = SingleGripperViaZhixing(
        host=str(manifest["robot_ip"]),
        tcp_port=int(manifest["robot_port"]),
        modbus_port=gripper_modbus_port,
        device_id=gripper_device_id,
    )
    gripper_executor.connect()

    try:
        base_to_camera_transform = load_base_to_camera_transform(Path(manifest["handeye_result_path"]))
        base_to_gripper_tcp_transform, affordance_pose = compute_gripper_tcp_and_ee_pose_in_base(
            robot_controller=rc,
            base_to_camera_transform=base_to_camera_transform,
            camera_gripper_tcp_pose=camera_gripper_tcp_pose,
        )
        camera_capture_to_ee_grasp = camera_gripper_tcp_pose @ np.linalg.inv(make_ee_to_gripper_transform())

        print("base gripper TCP pose matrix:")
        print(json.dumps(base_to_gripper_tcp_transform.tolist(), indent=2, ensure_ascii=False))
        print(f"base EE affordance_pose for robot (局部坐标系原点与朝向): {affordance_pose}")

        if bool(manifest.get("vis", False)):
            require_pyvista()
            source_point_cloud = o3d.io.read_point_cloud(str(source_ply))
            if source_point_cloud.is_empty():
                raise RuntimeError(f"读取 source 点云失败或为空: {source_ply}")
            show_registration_pyvista_preview(
                source_point_cloud=source_point_cloud,
                target_point_cloud=point_cloud,
                transformation=registration_summary.transformation,
                gripper_center_in_capture_camera=np.asarray(grasp_result["gripper_contact_center"], dtype=np.float64),
                gripper_rotation_in_capture_camera=np.asarray(grasp_result["gripper_rotation_matrix"], dtype=np.float64),
                gripper_width=float(grasp_result["gripper_width"]),
                camera_capture_to_ee_grasp=camera_capture_to_ee_grasp,
            )
            wait_for_execute_confirmation()

        executor = SkillExecutor(rc, dual_gripper=gripper_executor)
        executor.execute(str(manifest["skill_name"]), affordance_pose=affordance_pose, ref_pose=None)
    finally:
        gripper_executor.disconnect()
        rc.disconnect()


if __name__ == "__main__":
    main()
