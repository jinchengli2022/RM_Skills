from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency on target machine
    cv2 = None

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - runtime dependency on target machine
    pv = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency on target machine
    rs = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
from src.core.demo_project import RobotArmController


WINDOW_NAME = "Pick And Move Pose"
DEFAULT_SPHERE_DIAMETER_CM = 10.0
DEFAULT_AXIS_LENGTH_CM = 12.0
DEFAULT_DEPTH_STEP_CM = 1.0
ROTATION_STEP_DEG = 5.0
GRIPPER_TCP_OFFSET_IN_EE = np.asarray([0.0, 0.0, 0.19], dtype=np.float64)
POINT_CLOUD_STRIDE = 3
POINT_CLOUD_MAX_POINTS = 40000


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class SelectedPoint:
    pixel_x: int
    pixel_y: int
    raw_xyz: np.ndarray
    adjusted_xyz: np.ndarray


@dataclass
class AppState:
    intrinsics: CameraIntrinsics | None = None
    selected_point: SelectedPoint | None = None
    current_rotation: np.ndarray | None = None
    pending_click: tuple[int, int] | None = None
    phase: str = "pick"
    last_message: str = "左键点击 RGB 图中的一个点"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在头部相机 RGB 图上取点、调深度、选姿态并驱动机械臂到位")
    parser.add_argument("--camera-serial", default=None, help="指定 RealSense 序列号")
    parser.add_argument("--camera-index", type=int, default=0, help="按检测顺序选择第几台相机，从 0 开始")
    parser.add_argument("--width", type=int, default=1280, help="彩色图宽度")
    parser.add_argument("--height", type=int, default=720, help="彩色图高度")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument(
        "--sphere-diameter-cm",
        type=float,
        default=DEFAULT_SPHERE_DIAMETER_CM,
        help=f"小球直径，单位 cm，默认 {DEFAULT_SPHERE_DIAMETER_CM:g}",
    )
    parser.add_argument(
        "--axis-length-cm",
        type=float,
        default=DEFAULT_AXIS_LENGTH_CM,
        help=f"坐标轴长度，单位 cm，默认 {DEFAULT_AXIS_LENGTH_CM:g}",
    )
    parser.add_argument(
        "--depth-step-cm",
        type=float,
        default=DEFAULT_DEPTH_STEP_CM,
        help=f"深度调节步长，单位 cm，默认 {DEFAULT_DEPTH_STEP_CM:g}",
    )
    parser.add_argument("--robot-ip", default="169.254.128.18", help="机械臂 IP")
    parser.add_argument("--robot-port", type=int, default=8080, help="机械臂端口")
    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("/home/rm/ljc/RM_Skills/handeye_output/handeye_result.json"),
        help="手眼标定结果 JSON 路径",
    )
    parser.add_argument("--move-speed", type=int, default=20, help="机械臂 movel 速度百分比，1-100")
    return parser.parse_args()


def require_runtime() -> None:
    if cv2 is None:
        raise RuntimeError("未检测到 OpenCV。请先安装，例如：pip install opencv-python")
    if pv is None:
        raise RuntimeError("未检测到 PyVista。请先安装，例如：pip install pyvista")
    if rs is None:
        raise RuntimeError("未检测到 pyrealsense2。请先安装，例如：pip install pyrealsense2")


def get_device_info(device: rs.device, info: rs.camera_info) -> str | None:
    if not device.supports(info):
        return None
    value = device.get_info(info)
    return value or None


def list_available_realsense_devices() -> list[dict[str, str | int | None]]:
    ctx = rs.context()
    devices: list[dict[str, str | int | None]] = []
    for index, device in enumerate(ctx.query_devices()):
        devices.append(
            {
                "index": index,
                "name": get_device_info(device, rs.camera_info.name),
                "serial": get_device_info(device, rs.camera_info.serial_number),
                "ip": get_device_info(device, rs.camera_info.ip_address),
                "firmware": get_device_info(device, rs.camera_info.firmware_version),
                "connection_type": get_device_info(device, rs.camera_info.connection_type),
            }
        )
    return devices


def select_realsense_camera(
    devices: list[dict[str, str | int | None]],
    camera_serial: str | None,
    camera_index: int,
) -> dict[str, str | int | None]:
    if not devices:
        raise RuntimeError("未发现任何 RealSense 相机")
    if camera_serial:
        for device in devices:
            if device.get("serial") == camera_serial:
                return device
        available = ", ".join(str(device.get("serial", "unknown")) for device in devices)
        raise RuntimeError(f"找不到序列号为 {camera_serial} 的 RealSense 相机，当前可用序列号: {available}")
    if camera_index < 0 or camera_index >= len(devices):
        raise RuntimeError(f"camera-index={camera_index} 超出范围，当前共有 {len(devices)} 台相机")
    return devices[camera_index]


def skew_symmetric(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    k = skew_symmetric(axis)
    return np.eye(3, dtype=np.float64) + math.sin(angle_rad) * k + (1.0 - math.cos(angle_rad)) * (k @ k)


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def build_pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = orthonormalize_rotation(rotation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def project_point(point_xyz: np.ndarray, intr: CameraIntrinsics) -> tuple[int, int] | None:
    x, y, z = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    if z <= 0.0:
        return None
    u = int(round(intr.fx * x / z + intr.cx))
    v = int(round(intr.fy * y / z + intr.cy))
    return (u, v)


def sphere_radius_pixels(center_xyz: np.ndarray, intr: CameraIntrinsics, sphere_radius_m: float) -> int:
    z = float(center_xyz[2])
    if z <= 0.0:
        return 1
    px = ((intr.fx + intr.fy) * 0.5) * sphere_radius_m / z
    return max(1, int(round(px)))


def is_point_in_image(point_px: tuple[int, int] | None, width: int, height: int) -> bool:
    if point_px is None:
        return False
    x, y = point_px
    return 0 <= x < width and 0 <= y < height


def deproject_pixel_to_xyz(
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    pixel_x: int,
    pixel_y: int,
) -> np.ndarray | None:
    depth_m = float(depth_frame.get_distance(pixel_x, pixel_y))
    if depth_m <= 0.0:
        return None
    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel_x), float(pixel_y)], depth_m)
    return np.asarray(xyz, dtype=np.float64)


def recompute_xyz_on_same_camera_ray(original_xyz: np.ndarray, new_z: float) -> np.ndarray:
    original_z = float(original_xyz[2])
    if original_z <= 0.0:
        raise ValueError("原始 z 非法，无法沿相机射线重算坐标")
    scale = float(new_z / original_z)
    return np.asarray(original_xyz, dtype=np.float64) * scale


def load_base_to_camera_transform(handeye_path: str | Path) -> np.ndarray:
    handeye_file = Path(handeye_path)
    if not handeye_file.exists():
        raise FileNotFoundError(f"手眼标定结果不存在: {handeye_file}")
    data = json.loads(handeye_file.read_text(encoding="utf-8"))
    if "base_to_camera_transform" not in data:
        raise ValueError("手眼标定结果缺少 base_to_camera_transform，仅支持最新格式")
    transform = np.asarray(data["base_to_camera_transform"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"base_to_camera_transform 形状异常: {transform.shape}")
    return transform


def rm_pose_from_matrix(robot_controller: RobotArmController, matrix: np.ndarray) -> list[float]:
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot_controller.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(v) for v in pose]


def make_ee_to_gripper_tcp_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = GRIPPER_TCP_OFFSET_IN_EE
    return transform


def compute_gripper_tcp_and_ee_pose_in_base(
    robot_controller: RobotArmController,
    base_to_camera_transform: np.ndarray,
    camera_gripper_tcp_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    ee_to_gripper_tcp_transform = make_ee_to_gripper_tcp_transform()
    base_to_gripper_tcp_transform = base_to_camera_transform @ camera_gripper_tcp_pose
    base_to_ee_target_transform = base_to_gripper_tcp_transform @ np.linalg.inv(ee_to_gripper_tcp_transform)
    base_to_ee_target_pose = rm_pose_from_matrix(robot_controller, base_to_ee_target_transform)
    return base_to_gripper_tcp_transform, base_to_ee_target_transform, base_to_ee_target_pose


def draw_text_lines(image: np.ndarray, lines: list[str], start_x: int = 15, start_y: int = 28) -> None:
    y = start_y
    for line in lines:
        cv2.putText(image, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26


def draw_axis_legend(image: np.ndarray, rotation: np.ndarray) -> None:
    legend_x = image.shape[1] - 265
    legend_y = 28
    cv2.putText(image, "Axis Orientation", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    axis_specs = [("X", (0, 0, 255), rotation[:, 0]), ("Y", (0, 255, 0), rotation[:, 1]), ("Z", (255, 0, 0), rotation[:, 2])]
    row_y = legend_y + 28
    for label, color, axis_vec in axis_specs:
        cv2.line(image, (legend_x, row_y - 5), (legend_x + 18, row_y - 5), color, 3, cv2.LINE_AA)
        text = f"{label}: [{axis_vec[0]:+.3f}, {axis_vec[1]:+.3f}, {axis_vec[2]:+.3f}]"
        cv2.putText(image, text, (legend_x + 26, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        row_y += 24


def draw_overlay(
    color_image: np.ndarray,
    state: AppState,
    selected_camera: dict[str, str | int | None],
    sphere_radius_m: float,
    axis_length_m: float,
) -> np.ndarray:
    overlay = color_image.copy()
    lines = [
        f"camera: {selected_camera.get('name', 'unknown')}",
        f"serial: {selected_camera.get('serial', 'unknown')}",
        f"phase: {state.phase}",
        state.last_message,
    ]

    if state.phase == "pick":
        lines.extend(["left click to pick a point", "Esc quit"])
    elif state.phase == "depth_adjust" and state.selected_point is not None:
        point = state.selected_point.adjusted_xyz
        lines.extend(
            [
                f"point xyz(m): [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]",
                "w/s: increase/decrease depth on same camera ray",
                "Enter: open PyVista 3D orientation editor",
                "Esc quit",
            ]
        )

    if state.pending_click is not None:
        cv2.circle(overlay, state.pending_click, 6, (0, 165, 255), -1, cv2.LINE_AA)

    if state.selected_point is not None:
        click_px = (state.selected_point.pixel_x, state.selected_point.pixel_y)
        cv2.circle(overlay, click_px, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, click_px, 5, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(overlay, click_px, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 3)
        cv2.drawMarker(overlay, click_px, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(overlay, "clicked pixel", (click_px[0] + 8, click_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, "clicked pixel", (click_px[0] + 8, click_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
        if state.current_rotation is not None:
            draw_axis_legend(overlay, state.current_rotation)

    draw_text_lines(overlay, lines)
    return overlay


def update_rotation(state: AppState, axis_index: int, angle_deg: float) -> None:
    if state.current_rotation is None:
        return
    local_axis = state.current_rotation[:, axis_index]
    delta = rotation_from_axis_angle(local_axis, math.radians(angle_deg))
    state.current_rotation = orthonormalize_rotation(delta @ state.current_rotation)


def adjust_depth(state: AppState, delta_z_m: float) -> bool:
    if state.selected_point is None:
        return False
    current_z = float(state.selected_point.adjusted_xyz[2])
    next_z = current_z + delta_z_m
    if next_z <= 0.0:
        state.last_message = "新的 z 非法，不能小于等于 0"
        return False
    state.selected_point.adjusted_xyz = recompute_xyz_on_same_camera_ray(state.selected_point.raw_xyz, next_z)
    state.last_message = f"已调整深度到 {next_z:.4f} m，按 Enter 锁定点位"
    return True


def handle_key(state: AppState, key: int, depth_step_m: float) -> str | None:
    if key == 27:
        return "quit"

    if state.phase == "depth_adjust":
        if key == ord("w"):
            adjust_depth(state, depth_step_m)
        elif key == ord("s"):
            adjust_depth(state, -depth_step_m)
        elif key in (13, 10):
            state.last_message = "已锁定点位，准备打开 PyVista 3D 视图"
            return "depth_locked"
        return None
    return None


def on_mouse(event: int, x: int, y: int, _flags: int, param: AppState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and param.phase == "pick":
        param.pending_click = (int(x), int(y))
        param.last_message = f"已点击像素 ({x}, {y})，等待读取深度"


def print_pose_result(center_xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = build_pose_matrix(center_xyz, rotation)
    print("\n相机坐标系下的小球中心坐标 [x, y, z] (m):")
    print(np.array2string(center_xyz, precision=6, suppress_small=True))
    print("\n相机坐标系下的三轴方向向量 (rotation matrix columns):")
    print(np.array2string(rotation, precision=6, suppress_small=True))
    print("\n4x4 pose matrix in camera frame:")
    print(np.array2string(pose, precision=6, suppress_small=True))
    return pose


def build_point_cloud_from_frames(
    depth_frame: rs.depth_frame,
    color_frame: rs.video_frame,
    color_bgr: np.ndarray,
    stride: int = POINT_CLOUD_STRIDE,
    max_points: int = POINT_CLOUD_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    pointcloud = rs.pointcloud()
    pointcloud.map_to(color_frame)
    points = pointcloud.calculate(depth_frame)

    vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    texcoords = np.asarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    valid_mask = np.isfinite(vertices).all(axis=1)
    valid_mask &= vertices[:, 2] > 0.0
    vertices = vertices[valid_mask].astype(np.float64, copy=False)
    texcoords = texcoords[valid_mask]

    if stride > 1:
        vertices = vertices[::stride]
        texcoords = texcoords[::stride]

    if len(vertices) > max_points:
        sample_idx = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int32)
        vertices = vertices[sample_idx]
        texcoords = texcoords[sample_idx]

    color_rgb = color_bgr[:, :, ::-1]
    image_height, image_width = color_rgb.shape[:2]
    u = np.clip(np.rint(texcoords[:, 0] * (image_width - 1)).astype(np.int32), 0, image_width - 1)
    v = np.clip(np.rint(texcoords[:, 1] * (image_height - 1)).astype(np.int32), 0, image_height - 1)
    colors = color_rgb[v, u].astype(np.uint8, copy=False)
    return vertices, colors


def make_axis_arrow(start: np.ndarray, direction: np.ndarray, length: float, scale: float) -> "pv.PolyData":
    axis = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / norm
    return pv.Arrow(
        start=np.asarray(start, dtype=np.float64).reshape(3),
        direction=axis,
        scale=length,
        shaft_radius=max(scale * 0.08, 1e-4),
        tip_radius=max(scale * 0.18, 2e-4),
        tip_length=max(scale * 0.28, 5e-4),
    )


class PoseEditor3D:
    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        center_xyz: np.ndarray,
        axis_length_m: float,
        sphere_radius_m: float,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.uint8)
        self.center_xyz = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        self.axis_length_m = float(axis_length_m)
        self.sphere_radius_m = float(sphere_radius_m)
        self.rotation = np.eye(3, dtype=np.float64)
        self.confirmed = False
        self.plotter = pv.Plotter(window_size=(1400, 900))
        self._pose_actor_names = ("pose_sphere", "pose_x", "pose_y", "pose_z")

    def _refresh_pose_actors(self) -> None:
        for name in self._pose_actor_names:
            self.plotter.remove_actor(name, render=False)

        sphere = pv.Sphere(radius=self.sphere_radius_m, center=self.center_xyz)
        self.plotter.add_mesh(
            sphere,
            color="#ffd24a",
            smooth_shading=True,
            name="pose_sphere",
            render=False,
        )

        axis_specs = (
            ("pose_x", "#ff4d4d", self.rotation[:, 0]),
            ("pose_y", "#53d769", self.rotation[:, 1]),
            ("pose_z", "#4da3ff", self.rotation[:, 2]),
        )
        for name, color, axis_vec in axis_specs:
            arrow = make_axis_arrow(self.center_xyz, axis_vec, self.axis_length_m, self.axis_length_m)
            self.plotter.add_mesh(arrow, color=color, name=name, render=False)

        self.plotter.render()

    def _rotate_local(self, axis_index: int, angle_deg: float) -> None:
        local_axis = self.rotation[:, axis_index]
        delta = rotation_from_axis_angle(local_axis, math.radians(angle_deg))
        self.rotation = orthonormalize_rotation(delta @ self.rotation)
        self._refresh_pose_actors()

    def _reset(self) -> None:
        self.rotation = np.eye(3, dtype=np.float64)
        self._refresh_pose_actors()

    def _finish_interaction(self) -> None:
        interactor = getattr(self.plotter, "iren", None)
        if interactor is not None:
            interactor.terminate_app()

    def _confirm(self) -> None:
        self.confirmed = True
        self._finish_interaction()

    def _cancel(self) -> None:
        self.confirmed = False
        self._finish_interaction()

    def run(self) -> tuple[np.ndarray, bool]:
        print("\nPyVista 交互说明：")
        print("  鼠标左键拖拽旋转视角，中键平移，滚轮缩放")
        print("  j/l: 绕局部 X 轴 +/-5deg")
        print("  i/k: 绕局部 Y 轴 +/-5deg")
        print("  u/o: 绕局部 Z 轴 +/-5deg")
        print("  r: 重置姿态")
        print("  Space: 确认输出并执行")
        print("  Escape: 取消退出")

        cloud = pv.PolyData(self.points)
        cloud["rgb"] = self.colors
        self.plotter.set_background("#101418")
        self.plotter.add_points(cloud, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)
        self.plotter.add_axes(line_width=3, labels_off=False)
        self.plotter.show_grid(color="#666666")
        self.plotter.add_text(
            "Mouse: orbit/pan/zoom | j/l X | i/k Y | u/o Z | r reset | Space confirm | Esc cancel",
            position="upper_left",
            font_size=10,
            color="white",
        )

        self.plotter.add_key_event("j", lambda: self._rotate_local(0, ROTATION_STEP_DEG))
        self.plotter.add_key_event("l", lambda: self._rotate_local(0, -ROTATION_STEP_DEG))
        self.plotter.add_key_event("i", lambda: self._rotate_local(1, ROTATION_STEP_DEG))
        self.plotter.add_key_event("k", lambda: self._rotate_local(1, -ROTATION_STEP_DEG))
        self.plotter.add_key_event("u", lambda: self._rotate_local(2, ROTATION_STEP_DEG))
        self.plotter.add_key_event("o", lambda: self._rotate_local(2, -ROTATION_STEP_DEG))
        self.plotter.add_key_event("r", self._reset)
        self.plotter.add_key_event("space", self._confirm)
        self.plotter.add_key_event("Escape", self._cancel)

        self._refresh_pose_actors()
        self.plotter.camera_position = "iso"
        try:
            self.plotter.show()
        finally:
            self.plotter.close()
        return orthonormalize_rotation(self.rotation), self.confirmed


def execute_robot_move(args: argparse.Namespace, camera_gripper_tcp_pose: np.ndarray) -> None:
    base_to_camera_transform = load_base_to_camera_transform(args.handeye_result_path)
    move_speed = int(args.move_speed)
    if move_speed < 1 or move_speed > 100:
        raise RuntimeError("--move-speed 必须在 1 到 100 之间")
    rc = RobotArmController(args.robot_ip, args.robot_port, level=3)
    try:
        base_to_gripper_tcp_transform, base_to_ee_target_transform, base_to_ee_target_pose = compute_gripper_tcp_and_ee_pose_in_base(
            robot_controller=rc,
            base_to_camera_transform=base_to_camera_transform,
            camera_gripper_tcp_pose=camera_gripper_tcp_pose,
        )
        print("\nbase gripper TCP pose matrix:")
        print(np.array2string(base_to_gripper_tcp_transform, precision=6, suppress_small=True))
        print("\nbase EE target transform:")
        print(np.array2string(base_to_ee_target_transform, precision=6, suppress_small=True))
        print(f"\nbase EE target RM pose: {base_to_ee_target_pose}")
        rc.movel(base_to_ee_target_pose, v=move_speed, block=1)
    finally:
        rc.disconnect()


def main() -> int:
    require_runtime()
    args = parse_args()
    if args.sphere_diameter_cm <= 0.0:
        raise RuntimeError("--sphere-diameter-cm 必须大于 0")
    if args.axis_length_cm <= 0.0:
        raise RuntimeError("--axis-length-cm 必须大于 0")
    if args.depth_step_cm <= 0.0:
        raise RuntimeError("--depth-step-cm 必须大于 0")

    sphere_radius_m = float(args.sphere_diameter_cm) * 0.005
    axis_length_m = float(args.axis_length_cm) * 0.01
    depth_step_m = float(args.depth_step_cm) * 0.01

    devices = list_available_realsense_devices()
    selected_camera = select_realsense_camera(devices, args.camera_serial, args.camera_index)
    camera_serial = selected_camera.get("serial")
    if not camera_serial:
        raise RuntimeError(f"选中的 RealSense 相机缺少序列号，无法启动: {selected_camera}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(camera_serial))
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    align = rs.align(rs.stream.color)
    state = AppState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    captured_points: np.ndarray | None = None
    captured_colors: np.ndarray | None = None

    pipeline.start(config)
    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            color_intr = color_frame.profile.as_video_stream_profile().intrinsics
            state.intrinsics = CameraIntrinsics(
                fx=float(color_intr.fx),
                fy=float(color_intr.fy),
                cx=float(color_intr.ppx),
                cy=float(color_intr.ppy),
                width=int(color_intr.width),
                height=int(color_intr.height),
            )

            if state.pending_click is not None and state.phase == "pick":
                px, py = state.pending_click
                if px < 0 or py < 0 or px >= color_image.shape[1] or py >= color_image.shape[0]:
                    state.pending_click = None
                    state.last_message = "点击超出图像范围，请重新点击"
                else:
                    depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
                    xyz = deproject_pixel_to_xyz(depth_frame, depth_intr, px, py)
                    if xyz is None:
                        state.pending_click = None
                        state.last_message = "点击处无有效 depth，请重新点击"
                    else:
                        state.selected_point = SelectedPoint(
                            pixel_x=px,
                            pixel_y=py,
                            raw_xyz=xyz,
                            adjusted_xyz=xyz.copy(),
                        )
                        state.current_rotation = np.eye(3, dtype=np.float64)
                        state.phase = "depth_adjust"
                        state.pending_click = None
                        state.last_message = "已进入深度调整模式，w/s 调整深度，Enter 锁定"

            overlay = draw_overlay(color_image, state, selected_camera, sphere_radius_m, axis_length_m)
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            action = handle_key(state, key, depth_step_m)
            if action == "quit":
                break
            if action == "depth_locked":
                captured_points, captured_colors = build_point_cloud_from_frames(depth_frame, color_frame, color_image)
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if state.selected_point is None or captured_points is None or captured_colors is None:
        return 0

    editor = PoseEditor3D(
        points=captured_points,
        colors=captured_colors,
        center_xyz=state.selected_point.adjusted_xyz,
        axis_length_m=axis_length_m,
        sphere_radius_m=sphere_radius_m,
    )
    rotation, confirmed = editor.run()
    if not confirmed:
        return 0

    camera_gripper_tcp_pose = print_pose_result(
        state.selected_point.adjusted_xyz,
        rotation,
    )
    execute_robot_move(args, camera_gripper_tcp_pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
