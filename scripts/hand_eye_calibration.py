from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency on target machine
    cv2 = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
from src.core.demo_project import RobotArmController

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency on target machine
    rs = None


WINDOW_NAME = "RM Hand-Eye Calibration"
DEFAULT_ROBOT_PORT = 8080
DEFAULT_OUTPUT_DIR = Path("/home/rm/ljc/RM_Skills/handeye_output")
SUPPORTED_MODES = ("eye_in_hand", "eye_to_hand")
AUTO_TRANSLATION_BY_MODEL_MM = {
    "RM_65": 30.0,
    "RM_75": 35.0,
    "RML_63": 30.0,
    "ECO_65": 25.0,
    "GEN_72": 35.0,
    "ECO_63": 25.0,
}
AUTO_ROTATION_BY_MODEL_DEG = {
    "RM_65": 12.0,
    "RM_75": 15.0,
    "RML_63": 12.0,
    "ECO_65": 10.0,
    "GEN_72": 15.0,
    "ECO_63": 10.0,
}
SAFE_WORKSPACE_BY_MODEL = {
    "RM_65": ((-0.55, -0.45, 0.18), (0.55, 0.45, 0.65)),
    "RM_75": ((-0.70, -0.55, 0.18), (0.70, 0.55, 0.72)),
    "RML_63": ((-0.60, -0.50, 0.18), (0.65, 0.50, 0.70)),
    "ECO_65": ((-0.55, -0.45, 0.16), (0.55, 0.45, 0.60)),
    "GEN_72": ((-0.60, -0.55, 0.20), (0.75, 0.55, 0.80)),
    "ECO_63": ((-0.60, -0.45, 0.18), (0.65, 0.45, 0.72)),
}
DEFAULT_WORKSPACE_MIN = np.array([-0.60, -0.50, 0.15], dtype=np.float64)
DEFAULT_WORKSPACE_MAX = np.array([0.70, 0.50, 0.75], dtype=np.float64)
FRAME_MARGIN_RATIO = 0.08
MIN_MARKER_AREA_RATIO = 0.010
MAX_MARKER_AREA_RATIO = 0.30
MAX_CENTER_OFFSET_RATIO = 0.28
MIN_NORMAL_Z = 0.35
MIN_DEPTH_M = 0.12
MAX_DEPTH_M = 1.80


@dataclass
class Sample:
    index: int
    timestamp: float
    gripper_pose: list[float]
    gripper_rvec: list[float]
    gripper_tvec: list[float]
    target_rvec: list[float]
    target_tvec: list[float]


@dataclass
class AutoMoveConfig:
    translation_step_m: float
    rotation_step_rad: float
    workspace_min_xyz: np.ndarray
    workspace_max_xyz: np.ndarray
    reference_workspace_min_xyz: np.ndarray
    reference_workspace_max_xyz: np.ndarray


@dataclass
class ArucoDetection:
    found: bool
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    preview: np.ndarray
    corners: np.ndarray | None
    center_offset_ratio: float | None
    area_ratio: float | None
    normal_z: float | None
    depth_m: float | None
    quality_ok: bool = False
    reject_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RM 手眼标定（眼在手上 / 眼在手外）")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=None, help="手眼标定模式，必须显式指定")
    parser.add_argument("--robot-ip", default=None, help="机械臂 IP")
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT, help="机械臂端口")
    parser.add_argument("--list-cameras", action="store_true", help="列出所有 RealSense 相机后退出")
    parser.add_argument("--camera-serial", default=None, help="指定 RealSense 序列号，三台相机时推荐用这个")
    parser.add_argument("--camera-index", type=int, default=0, help="按检测顺序选择第几台相机，从 0 开始")
    parser.add_argument("--width", type=int, default=1280, help="彩色图宽度")
    parser.add_argument("--height", type=int, default=720, help="彩色图高度")
    parser.add_argument("--fps", type=int, default=30, help="彩色图帧率")
    parser.add_argument("--board-cols", type=int, default=None, help="棋盘格内角点列数")
    parser.add_argument("--board-rows", type=int, default=None, help="棋盘格内角点行数")
    parser.add_argument("--square-size-mm", type=float, default=None, help="棋盘格每格边长，单位 mm")
    parser.add_argument("--aruco", action="store_true", help="使用 ArUco 码替代棋盘格进行手眼标定")
    parser.add_argument(
        "--aruco-dict",
        default="DICT_4X4_50",
        choices=[
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_4X4_1000",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_5X5_1000",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_7X7_50",
            "DICT_7X7_100",
            "DICT_7X7_250",
            "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL",
        ],
        help="ArUco 字典名称（仅 --aruco 模式有效）",
    )
    parser.add_argument("--marker-id", type=int, default=0, help="目标 ArUco marker ID（仅 --aruco 模式有效）")
    parser.add_argument("--marker-size-mm", type=float, default=None, help="ArUco marker 边长，单位 mm（仅 --aruco 模式有效）")
    parser.add_argument("--min-samples", type=int, default=10, help="最少采样数")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
        help="OpenCV 手眼标定算法",
    )
    parser.add_argument("--auto-move", action="store_true", help="自动移动机械臂并采样")
    parser.add_argument("--move-speed", type=int, default=10, help="自动移动速度，建议 1-20")
    parser.add_argument("--settle-time", type=float, default=1.2, help="每次运动后的稳定等待时间，单位秒")
    parser.add_argument("--detect-retries", type=int, default=15, help="每个自动位姿的目标检测重试次数")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.list_cameras:
        return

    if args.mode is None:
        raise RuntimeError("--mode 必须显式指定，可选值: eye_in_hand, eye_to_hand")
    if args.robot_ip is None:
        raise RuntimeError("非 --list-cameras 模式下必须提供 --robot-ip")

    if args.mode == "eye_to_hand":
        if not args.aruco:
            raise RuntimeError("eye_to_hand 模式仅支持 ArUco，请添加 --aruco")
        if args.marker_size_mm is None:
            raise RuntimeError("eye_to_hand 模式需要提供 --marker-size-mm")
        if not args.auto_move:
            raise RuntimeError("eye_to_hand 模式仅支持自动采样，请添加 --auto-move")
        return

    if args.aruco:
        if args.marker_size_mm is None:
            raise RuntimeError("ArUco 模式需要提供 --marker-size-mm")
    elif args.board_cols is None or args.board_rows is None or args.square_size_mm is None:
        raise RuntimeError("eye_in_hand 棋盘格模式需要提供 --board-cols、--board-rows 和 --square-size-mm")


def require_realsense() -> None:
    if rs is None:
        raise RuntimeError("未检测到 pyrealsense2。请先安装，例如：pip install pyrealsense2")


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未检测到 OpenCV。请先安装，例如：pip install opencv-python")


def list_realsense_devices() -> list[dict[str, str]]:
    require_realsense()
    ctx = rs.context()
    devices: list[dict[str, str]] = []
    for index, device in enumerate(ctx.query_devices()):
        devices.append(
            {
                "index": index,
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
        )
    return devices


def select_camera(devices: list[dict[str, str]], camera_serial: str | None, camera_index: int) -> dict[str, str]:
    if not devices:
        raise RuntimeError("未发现任何 RealSense 相机")

    if camera_serial:
        for device in devices:
            if device["serial"] == camera_serial:
                return device
        available = ", ".join(device["serial"] for device in devices)
        raise RuntimeError(f"找不到序列号为 {camera_serial} 的相机，当前可用序列号: {available}")

    if camera_index < 0 or camera_index >= len(devices):
        raise RuntimeError(f"camera-index={camera_index} 超出范围，当前共有 {len(devices)} 台相机")

    return devices[camera_index]


class RealSenseCamera:
    def __init__(self, device: dict[str, str], width: int, height: int, fps: int):
        require_realsense()
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

    def start(self) -> None:
        config = rs.config()
        config.enable_device(self.device["serial"])
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(config)

        for _ in range(10):
            self.pipeline.wait_for_frames()

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.camera_matrix = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            raise RuntimeError("未读取到彩色图像帧")
        return np.asanyarray(color_frame.get_data())


def make_board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * square_size_m
    return object_points


def matrix_from_rm_pose(robot: RobotArmController, pose: list[float]) -> np.ndarray:
    rm_matrix = robot.robot.rm_algo_pos2matrix(pose)
    values = [float(v) for v in rm_matrix.data]
    return np.array(values, dtype=np.float64).reshape(4, 4)


def rm_pose_from_matrix(robot: RobotArmController, matrix: np.ndarray) -> list[float]:
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(v) for v in pose]


def get_current_pose(robot: RobotArmController) -> list[float]:
    ret, state = robot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂当前位姿失败，错误码: {ret}")
    pose = state.get("pose")
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"机械臂返回的 pose 格式异常: {state}")
    return [float(v) for v in pose]


def rad_from_deg(value: float) -> float:
    return float(np.deg2rad(value))


def build_auto_move_config(arm_model: str, base_pose: list[float] | None = None) -> AutoMoveConfig:
    translation_step_m = AUTO_TRANSLATION_BY_MODEL_MM.get(arm_model, 30.0) / 1000.0
    rotation_step_rad = rad_from_deg(AUTO_ROTATION_BY_MODEL_DEG.get(arm_model, 12.0))
    workspace = SAFE_WORKSPACE_BY_MODEL.get(arm_model)
    if workspace is None:
        reference_workspace_min_xyz = DEFAULT_WORKSPACE_MIN.copy()
        reference_workspace_max_xyz = DEFAULT_WORKSPACE_MAX.copy()
    else:
        reference_workspace_min_xyz = np.array(workspace[0], dtype=np.float64)
        reference_workspace_max_xyz = np.array(workspace[1], dtype=np.float64)

    workspace_min_xyz = reference_workspace_min_xyz.copy()
    workspace_max_xyz = reference_workspace_max_xyz.copy()
    if base_pose is not None:
        base_xyz = np.array(base_pose[:3], dtype=np.float64)
        dynamic_margin = np.array(
            [
                max(translation_step_m * 2.2, 0.08),
                max(translation_step_m * 2.2, 0.08),
                max(translation_step_m * 1.8, 0.08),
            ],
            dtype=np.float64,
        )
        workspace_min_xyz = np.minimum(reference_workspace_min_xyz, base_xyz - dynamic_margin)
        workspace_max_xyz = np.maximum(reference_workspace_max_xyz, base_xyz + dynamic_margin)
    return AutoMoveConfig(
        translation_step_m=translation_step_m,
        rotation_step_rad=rotation_step_rad,
        workspace_min_xyz=workspace_min_xyz,
        workspace_max_xyz=workspace_max_xyz,
        reference_workspace_min_xyz=reference_workspace_min_xyz,
        reference_workspace_max_xyz=reference_workspace_max_xyz,
    )


def apply_pose_offset(base_pose: list[float], offset: list[float]) -> list[float]:
    return [float(base_pose[i] + offset[i]) for i in range(6)]


def is_pose_in_workspace(target_pose: list[float], config: AutoMoveConfig) -> tuple[bool, str]:
    xyz = np.array(target_pose[:3], dtype=np.float64)
    if np.any(xyz < config.workspace_min_xyz) or np.any(xyz > config.workspace_max_xyz):
        return False, "超出内置安全工作空间"
    return True, "ok"


def generate_auto_offsets(arm_model: str, mode: str) -> list[list[float]]:
    config = build_auto_move_config(arm_model)
    step_t = config.translation_step_m
    step_r = config.rotation_step_rad
    step_r2 = step_r * 2.0
    offsets = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [step_t, 0.0, 0.0, 0.0, step_r, 0.0],
        [-step_t, 0.0, 0.0, 0.0, -step_r, 0.0],
        [0.0, step_t, 0.0, step_r, 0.0, 0.0],
        [0.0, -step_t, 0.0, -step_r, 0.0, 0.0],
        [0.0, 0.0, step_t, 0.0, 0.0, step_r],
        [0.0, 0.0, -step_t, 0.0, 0.0, -step_r],
        [step_t, step_t, 0.0, step_r, step_r, 0.0],
        [step_t, -step_t, 0.0, step_r, -step_r, 0.0],
        [-step_t, step_t, 0.0, -step_r, step_r, 0.0],
        [-step_t, -step_t, 0.0, -step_r, -step_r, 0.0],
        [0.0, 0.0, step_t, step_r2, 0.0, 0.0],
        [0.0, 0.0, step_t, 0.0, step_r2, 0.0],
        [0.0, 0.0, step_t, 0.0, 0.0, step_r2],
    ]
    if mode == "eye_to_hand":
        offsets.extend(
            [
                [step_t * 1.5, 0.0, step_t, 0.0, step_r2, 0.0],
                [-step_t * 1.5, 0.0, step_t, 0.0, -step_r2, 0.0],
                [0.0, step_t * 1.5, step_t, step_r2, 0.0, step_r],
                [0.0, -step_t * 1.5, step_t, -step_r2, 0.0, -step_r],
                [step_t, 0.0, step_t * 1.5, 0.0, step_r, step_r2],
                [-step_t, 0.0, step_t * 1.5, 0.0, -step_r, -step_r2],
            ]
        )
    return offsets


def move_robot_to_pose(robot: RobotArmController, target_pose: list[float], speed: int) -> None:
    ret = robot.robot.rm_movel(target_pose, speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"机械臂 movel 失败，错误码: {ret}，目标位姿: {target_pose}")


def detect_board_pose(
    image: np.ndarray,
    board_size: tuple[int, int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray]:
    require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    preview = image.copy()

    if not found:
        return False, None, None, preview

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    success, rvec, tvec = cv2.solvePnP(object_points, corners_subpix, camera_matrix, dist_coeffs)
    cv2.drawChessboardCorners(preview, board_size, corners_subpix, found)

    if not success:
        return False, None, None, preview

    return True, rvec, tvec, preview


def _aruco_dict_id(name: str) -> int:
    require_cv2()
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    return mapping[name]


def evaluate_aruco_detection(
    corners: np.ndarray,
    rotation: np.ndarray,
    tvec: np.ndarray,
    image_shape: tuple[int, int, int],
) -> tuple[bool, str, float, float, float, float]:
    height, width = image_shape[:2]
    points = corners.reshape(4, 2)
    center = points.mean(axis=0)
    center_offset_ratio = float(np.linalg.norm(center - np.array([width / 2.0, height / 2.0])) / np.linalg.norm([width / 2.0, height / 2.0]))
    area = cv2.contourArea(points.astype(np.float32))
    area_ratio = float(area / float(width * height))
    margin_x = width * FRAME_MARGIN_RATIO
    margin_y = height * FRAME_MARGIN_RATIO
    normal_z = float(abs(rotation[2, 2]))
    depth_m = float(tvec.reshape(3)[2])

    if np.any(points[:, 0] < margin_x) or np.any(points[:, 0] > width - margin_x):
        return False, "marker 超出左右安全边界", center_offset_ratio, area_ratio, normal_z, depth_m
    if np.any(points[:, 1] < margin_y) or np.any(points[:, 1] > height - margin_y):
        return False, "marker 超出上下安全边界", center_offset_ratio, area_ratio, normal_z, depth_m
    if center_offset_ratio > MAX_CENTER_OFFSET_RATIO:
        return False, "marker 偏离图像中心过多", center_offset_ratio, area_ratio, normal_z, depth_m
    if area_ratio < MIN_MARKER_AREA_RATIO:
        return False, "marker 投影过小", center_offset_ratio, area_ratio, normal_z, depth_m
    if area_ratio > MAX_MARKER_AREA_RATIO:
        return False, "marker 投影过大", center_offset_ratio, area_ratio, normal_z, depth_m
    if depth_m < MIN_DEPTH_M or depth_m > MAX_DEPTH_M:
        return False, "marker 深度不在有效范围", center_offset_ratio, area_ratio, normal_z, depth_m
    if normal_z < MIN_NORMAL_Z:
        return False, "marker 视角过斜", center_offset_ratio, area_ratio, normal_z, depth_m
    return True, "ok", center_offset_ratio, area_ratio, normal_z, depth_m


def detect_aruco_pose(
    image: np.ndarray,
    aruco_dict_name: str,
    marker_id: int,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    quality_gate: bool,
) -> ArucoDetection:
    require_cv2()
    aruco_dict = cv2.aruco.getPredefinedDictionary(_aruco_dict_id(aruco_dict_name))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    preview = image.copy()

    if ids is None:
        return ArucoDetection(False, None, None, preview, None, None, None, None, None, False, "未识别到 ArUco")

    half = marker_size_m / 2.0
    obj_pts = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )

    for i, mid in enumerate(ids.flatten()):
        if int(mid) != marker_id:
            continue
        marker_corners = corners[i]
        cv2.aruco.drawDetectedMarkers(preview, [marker_corners], np.array([[marker_id]]))
        img_pts = marker_corners.reshape(4, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        if not success:
            return ArucoDetection(False, None, None, preview, marker_corners, None, None, None, None, False, "solvePnP 失败")
        rotation, _ = cv2.Rodrigues(rvec)
        ok, reason, center_offset_ratio, area_ratio, normal_z, depth_m = evaluate_aruco_detection(
            marker_corners,
            rotation,
            tvec,
            image.shape,
        )
        return ArucoDetection(
            True,
            rvec,
            tvec,
            preview,
            marker_corners,
            center_offset_ratio,
            area_ratio,
            normal_z,
            depth_m,
            ok if quality_gate else True,
            reason if quality_gate else "",
        )

    return ArucoDetection(False, None, None, preview, None, None, None, None, None, False, f"未找到 marker id={marker_id}")


def handeye_method(name: str) -> int:
    require_cv2()
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    return methods[name]


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def build_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3].reshape(3, 1)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = (-rotation.T @ translation).reshape(3)
    return inverse


def draw_overlay(
    image: np.ndarray,
    camera_name: str,
    camera_serial: str,
    sample_count: int,
    min_samples: int,
    found: bool,
    mode_text: str,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    status = "FOUND" if found else "SEARCHING"
    color = (0, 220, 0) if found else (0, 180, 255)
    lines = [
        f"camera: {camera_name}",
        f"serial: {camera_serial}",
        f"samples: {sample_count}/{min_samples}",
        f"target: {status}",
        f"mode: {mode_text}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("keys: s=sample  c=calibrate  q=quit")
    overlay = image.copy()
    y = 30
    for line in lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 30
    return overlay


def compute_eye_in_hand(
    robot: RobotArmController,
    samples: list[Sample],
    method_name: str,
) -> dict[str, object]:
    if len(samples) < 3:
        raise RuntimeError("至少需要 3 组样本才能进行计算")

    r_gripper2base = [np.array(sample.gripper_rvec, dtype=np.float64).reshape(3, 1) for sample in samples]
    t_gripper2base = [np.array(sample.gripper_tvec, dtype=np.float64).reshape(3, 1) for sample in samples]
    r_target2cam = [np.array(sample.target_rvec, dtype=np.float64).reshape(3, 1) for sample in samples]
    t_target2cam = [np.array(sample.target_tvec, dtype=np.float64).reshape(3, 1) for sample in samples]

    rotation_cam2gripper, translation_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=handeye_method(method_name),
    )

    transform_cam2gripper = build_transform(rotation_cam2gripper, translation_cam2gripper)
    pose_cam2gripper = rm_pose_from_matrix(robot, transform_cam2gripper)
    quaternion_cam2gripper = rotation_matrix_to_quaternion(rotation_cam2gripper)

    return {
        "method": method_name,
        "sample_count": len(samples),
        "camera_to_gripper_transform": transform_cam2gripper.tolist(),
        "camera_to_gripper_pose_rm": pose_cam2gripper,
        "camera_to_gripper_quaternion_xyzw": quaternion_cam2gripper,
        "translation_m": translation_cam2gripper.reshape(3).astype(float).tolist(),
        "rotation_matrix": rotation_cam2gripper.astype(float).tolist(),
    }


def compute_eye_to_hand(samples: list[Sample], method_name: str) -> np.ndarray:
    if len(samples) < 3:
        raise RuntimeError("至少需要 3 组样本才能进行计算")

    r_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    r_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []

    for sample in samples:
        base_to_gripper = build_transform(
            cv2.Rodrigues(np.array(sample.gripper_rvec, dtype=np.float64).reshape(3, 1))[0],
            np.array(sample.gripper_tvec, dtype=np.float64).reshape(3, 1),
        )
        gripper_to_base = invert_transform(base_to_gripper)
        target_to_camera = build_transform(
            cv2.Rodrigues(np.array(sample.target_rvec, dtype=np.float64).reshape(3, 1))[0],
            np.array(sample.target_tvec, dtype=np.float64).reshape(3, 1),
        )
        r_gripper2base.append(gripper_to_base[:3, :3])
        t_gripper2base.append(gripper_to_base[:3, 3].reshape(3, 1))
        r_target2cam.append(target_to_camera[:3, :3])
        t_target2cam.append(target_to_camera[:3, 3].reshape(3, 1))

    rotation_base_to_camera, translation_base_to_camera = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=handeye_method(method_name),
    )
    return build_transform(rotation_base_to_camera, translation_base_to_camera)


def append_sample(
    robot: RobotArmController,
    samples: list[Sample],
    gripper_pose: list[float],
    target_rvec: np.ndarray,
    target_tvec: np.ndarray,
) -> None:
    base_to_gripper = matrix_from_rm_pose(robot, gripper_pose)
    gripper_rvec, _ = cv2.Rodrigues(base_to_gripper[:3, :3])
    sample = Sample(
        index=len(samples),
        timestamp=time.time(),
        gripper_pose=gripper_pose,
        gripper_rvec=gripper_rvec.reshape(3).astype(float).tolist(),
        gripper_tvec=base_to_gripper[:3, 3].reshape(3).astype(float).tolist(),
        target_rvec=target_rvec.reshape(3).astype(float).tolist(),
        target_tvec=target_tvec.reshape(3).astype(float).tolist(),
    )
    samples.append(sample)


def try_detect_board(
    camera_stream: RealSenseCamera,
    board_size: tuple[int, int],
    object_points: np.ndarray,
    retries: int,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray]:
    preview = camera_stream.read()
    for _ in range(max(retries, 1)):
        frame = camera_stream.read()
        found, target_rvec, target_tvec, preview = detect_board_pose(
            frame,
            board_size,
            object_points,
            camera_stream.camera_matrix,
            camera_stream.dist_coeffs,
        )
        if found:
            return True, target_rvec, target_tvec, preview
    return False, None, None, preview


def try_detect_aruco(
    camera_stream: RealSenseCamera,
    aruco_dict_name: str,
    marker_id: int,
    marker_size_m: float,
    retries: int,
    quality_gate: bool,
) -> ArucoDetection:
    preview = camera_stream.read()
    last_detection = ArucoDetection(False, None, None, preview, None, None, None, None, None, False, "未开始检测")
    for _ in range(max(retries, 1)):
        frame = camera_stream.read()
        detection = detect_aruco_pose(
            frame,
            aruco_dict_name,
            marker_id,
            marker_size_m,
            camera_stream.camera_matrix,
            camera_stream.dist_coeffs,
            quality_gate,
        )
        last_detection = detection
        if detection.found and detection.rvec is not None and detection.tvec is not None and detection.quality_ok:
            return detection
    return last_detection


def run_auto_sampling(
    robot: RobotArmController,
    camera: dict[str, str],
    camera_stream: RealSenseCamera,
    board_size: tuple[int, int],
    object_points: np.ndarray,
    args: argparse.Namespace,
    samples: list[Sample],
    arm_model: str,
) -> np.ndarray | None:
    base_pose = get_current_pose(robot)
    config = build_auto_move_config(arm_model, base_pose=base_pose)
    offsets = generate_auto_offsets(arm_model, args.mode)
    last_preview: np.ndarray | None = None
    marker_size_m = (args.marker_size_mm or 0.0) / 1000.0

    print("\n开始自动移动采样")
    print(f"当前机械臂型号: {arm_model}")
    print(f"初始位姿: {base_pose}")
    print(f"自动位姿数量: {len(offsets)}，速度={args.move_speed}")
    base_xyz = np.array(base_pose[:3], dtype=np.float64)
    if np.any(base_xyz < config.reference_workspace_min_xyz) or np.any(base_xyz > config.reference_workspace_max_xyz):
        print(
            "注意：当前起始位姿超出了预设绝对工作空间，"
            "已自动切换为基于当前位姿的动态安全范围。"
        )
    print(
        "本次有效工作空间: "
        f"min={config.workspace_min_xyz.round(4).tolist()}, "
        f"max={config.workspace_max_xyz.round(4).tolist()}"
    )

    for index, offset in enumerate(offsets, start=1):
        if len(samples) >= args.min_samples:
            break

        target_pose = apply_pose_offset(base_pose, offset)
        safe, reason = is_pose_in_workspace(target_pose, config)
        if not safe:
            print(f"跳过自动位姿 {index}/{len(offsets)}: {reason} -> {target_pose}")
            continue

        print(f"自动位姿 {index}/{len(offsets)}: 移动到 {target_pose}")
        move_robot_to_pose(robot, target_pose, args.move_speed)
        time.sleep(args.settle_time)

        overlay = draw_overlay(
            camera_stream.read(),
            camera["name"],
            camera["serial"],
            len(samples),
            args.min_samples,
            False,
            f"{args.mode} auto {index}/{len(offsets)}",
        )
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("自动采样被用户终止")
            break

        if args.aruco:
            detection = try_detect_aruco(
                camera_stream,
                args.aruco_dict,
                args.marker_id,
                marker_size_m,
                args.detect_retries,
                quality_gate=(args.mode == "eye_to_hand"),
            )
            found = detection.found and detection.rvec is not None and detection.tvec is not None and detection.quality_ok
            preview = detection.preview
            target_rvec = detection.rvec
            target_tvec = detection.tvec
            extra_lines = []
            if detection.center_offset_ratio is not None:
                extra_lines.append(f"center_offset={detection.center_offset_ratio:.3f}")
            if detection.area_ratio is not None:
                extra_lines.append(f"area_ratio={detection.area_ratio:.3f}")
            if detection.normal_z is not None:
                extra_lines.append(f"normal_z={detection.normal_z:.3f}")
            if detection.reject_reason and not detection.quality_ok:
                extra_lines.append(f"reject={detection.reject_reason}")
        else:
            found, target_rvec, target_tvec, preview = try_detect_board(
                camera_stream,
                board_size,
                object_points,
                args.detect_retries,
            )
            extra_lines = []

        last_preview = preview
        overlay = draw_overlay(
            preview,
            camera["name"],
            camera["serial"],
            len(samples),
            args.min_samples,
            found,
            f"{args.mode} auto {index}/{len(offsets)}",
            extra_lines=extra_lines,
        )
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("自动采样被用户终止")
            break

        if not found or target_rvec is None or target_tvec is None:
            print(f"自动位姿 {index}/{len(offsets)} 目标无效，跳过")
            continue

        gripper_pose = get_current_pose(robot)
        append_sample(robot, samples, gripper_pose, target_rvec, target_tvec)
        print(f"自动采样成功 {len(samples)} / {args.min_samples}")

    print("自动采样结束")

    try:
        move_robot_to_pose(robot, base_pose, args.move_speed)
    except Exception as exc:  # pragma: no cover - hardware side effect
        print(f"回到初始位姿失败: {exc}")

    return last_preview


def save_eye_in_hand_result(
    output_dir: Path,
    args: argparse.Namespace,
    camera: dict[str, str],
    samples: list[Sample],
    result: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "eye_in_hand",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "robot": {
            "ip": args.robot_ip,
            "port": args.robot_port,
        },
        "camera": camera,
        "board": {
            "cols": args.board_cols,
            "rows": args.board_rows,
            "square_size_mm": args.square_size_mm,
        },
        "result": result,
        "samples": [sample.__dict__ for sample in samples],
    }
    output_path = output_dir / "handeye_result.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_eye_to_hand_result(output_dir: Path, base_to_camera_transform: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "这是头部相机相对于机械臂 base 的标定结果，矩阵方向为相机坐标系到机械臂 base 坐标系。",
        "base_to_camera_transform": base_to_camera_transform.astype(float).tolist(),
    }
    output_path = output_dir / "handeye_result.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run_manual_eye_in_hand(
    robot: RobotArmController,
    camera: dict[str, str],
    camera_stream: RealSenseCamera,
    board_size: tuple[int, int],
    object_points: np.ndarray,
    args: argparse.Namespace,
    samples: list[Sample],
) -> None:
    while True:
        frame = camera_stream.read()
        if args.aruco:
            detection = detect_aruco_pose(
                frame,
                args.aruco_dict,
                args.marker_id,
                args.marker_size_mm / 1000.0,
                camera_stream.camera_matrix,
                camera_stream.dist_coeffs,
                quality_gate=False,
            )
            found = detection.found and detection.rvec is not None and detection.tvec is not None
            preview = detection.preview
            target_rvec = detection.rvec
            target_tvec = detection.tvec
            extra_lines = []
        else:
            found, target_rvec, target_tvec, preview = detect_board_pose(
                frame,
                board_size,
                object_points,
                camera_stream.camera_matrix,
                camera_stream.dist_coeffs,
            )
            extra_lines = []

        overlay = draw_overlay(
            preview,
            camera["name"],
            camera["serial"],
            len(samples),
            args.min_samples,
            found,
            "eye_in_hand manual",
            extra_lines=extra_lines,
        )
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            if not found or target_rvec is None or target_tvec is None:
                print("未检测到目标，当前帧不能采样")
                continue
            gripper_pose = get_current_pose(robot)
            append_sample(robot, samples, gripper_pose, target_rvec, target_tvec)
            print(f"已采样 {len(samples)} / {args.min_samples}: gripper_pose={gripper_pose}")
            continue
        if key == ord("c"):
            if len(samples) < args.min_samples:
                print(f"样本数不足，当前 {len(samples)}，至少需要 {args.min_samples}")
                continue
            result = compute_eye_in_hand(robot, samples, args.method)
            output_path = save_eye_in_hand_result(Path(args.output_dir), args, camera, samples, result)
            print("\n标定完成")
            print(f"camera_to_gripper_pose_rm = {result['camera_to_gripper_pose_rm']}")
            print(f"结果已保存到: {output_path}")


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.list_cameras:
        for device in list_realsense_devices():
            print(
                f'index={device["index"]}  serial={device["serial"]}  '
                f'name={device["name"]}  firmware={device["firmware"]}'
            )
        return 0

    require_cv2()

    devices = list_realsense_devices()
    camera = select_camera(devices, args.camera_serial, args.camera_index)

    if args.aruco:
        board_size = (0, 0)
        object_points = np.zeros((0, 3), dtype=np.float32)
    else:
        square_size_m = args.square_size_mm / 1000.0
        board_size = (args.board_cols, args.board_rows)
        object_points = make_board_points(args.board_cols, args.board_rows, square_size_m)

    robot = RobotArmController(args.robot_ip, args.robot_port, level=3)
    camera_stream = RealSenseCamera(camera, args.width, args.height, args.fps)
    samples: list[Sample] = []

    print("=" * 72)
    print("RM 手眼标定")
    print("=" * 72)
    print(f"模式: {args.mode}")
    print(f"机械臂 IP: {args.robot_ip}")
    print(f'当前相机: index={camera["index"]}, serial={camera["serial"]}, name={camera["name"]}')
    if args.aruco:
        print(f"ArUco 字典: {args.aruco_dict}，marker ID: {args.marker_id}，边长: {args.marker_size_mm:.3f} mm")
    else:
        print(f"棋盘格内角点: {args.board_cols} x {args.board_rows}")
        print(f"棋盘格边长: {args.square_size_mm:.3f} mm")
    print(f"最少样本数: {args.min_samples}")
    print(f"输出目录: {Path(args.output_dir).resolve()}")
    if args.auto_move:
        print("运行模式: 自动移动采样")
    elif args.mode == "eye_in_hand":
        print("按键说明: s=采样, c=计算标定, q=退出")

    try:
        camera_stream.start()
        arm_model = robot.get_arm_model() or "UNKNOWN"

        if args.auto_move:
            preview = run_auto_sampling(robot, camera, camera_stream, board_size, object_points, args, samples, arm_model)
            if preview is not None:
                overlay = draw_overlay(
                    preview,
                    camera["name"],
                    camera["serial"],
                    len(samples),
                    args.min_samples,
                    len(samples) > 0,
                    f"{args.mode} auto",
                )
                cv2.imshow(WINDOW_NAME, overlay)
                cv2.waitKey(10)

            if len(samples) < args.min_samples:
                print(f"自动采样完成，但样本数不足: {len(samples)} / {args.min_samples}")
                return 1

            if args.mode == "eye_to_hand":
                base_to_camera_transform = compute_eye_to_hand(samples, args.method)
                output_path = save_eye_to_hand_result(Path(args.output_dir), base_to_camera_transform)
                print("\n标定完成")
                print("base_to_camera_transform =")
                print(np.array2string(base_to_camera_transform, precision=6, suppress_small=True))
                print(f"结果已保存到: {output_path}")
                return 0

            result = compute_eye_in_hand(robot, samples, args.method)
            output_path = save_eye_in_hand_result(Path(args.output_dir), args, camera, samples, result)
            print("\n标定完成")
            print(f"camera_to_gripper_pose_rm = {result['camera_to_gripper_pose_rm']}")
            print(f"结果已保存到: {output_path}")
            return 0

        if args.mode == "eye_to_hand":
            raise RuntimeError("eye_to_hand 模式仅支持自动采样，请添加 --auto-move")

        run_manual_eye_in_hand(robot, camera, camera_stream, board_size, object_points, args, samples)
    finally:
        camera_stream.stop()
        robot.disconnect()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
