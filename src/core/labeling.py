from __future__ import annotations

import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None


GRIPPER_TCP_OFFSET_IN_EE = np.asarray([0.0, 0.0, 0.19], dtype=np.float64)
HEAD_CAMERA_SERIAL = "344322073674"
WINDOW_NAME = "Label With SAM"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 15
DEFAULT_WARMUP_FRAMES = 30
DEFAULT_DEPTH_TRUNC_M = 3.0
CONSISTENCY_TOL = 1e-6
SAM_MODEL_TYPE = "vit_h"


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class CapturedSelection:
    color_bgr: np.ndarray
    color_rgb: np.ndarray
    depth_image: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale: float
    mask: np.ndarray
    prompt_pixel: tuple[int, int]


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法显示 2D 标注界面")


def require_open3d() -> None:
    if o3d is None:
        raise RuntimeError("未安装 open3d，无法保存点云")


def require_pyvista() -> None:
    if pv is None:
        raise RuntimeError("未安装 pyvista，无法显示标注结果三维预览")


def camera_role_name(serial: str | None) -> str | None:
    if serial == HEAD_CAMERA_SERIAL:
        return "头部相机"
    return None


def format_camera_serial(serial: str | None) -> str:
    if serial is None:
        return "unknown"
    role = camera_role_name(serial)
    if role is None:
        return str(serial)
    return f"{serial} ({role})"


def draw_prompt_overlay(
    image_bgr: np.ndarray,
    prompt_pixel: tuple[int, int] | None,
    selected: bool,
) -> np.ndarray:
    require_cv2()
    overlay = image_bgr.copy()
    if prompt_pixel is not None:
        cv2.drawMarker(overlay, prompt_pixel, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.circle(overlay, prompt_pixel, 5, (255, 255, 255), -1)
    lines = [
        "Point selected, running SAM next" if selected else "Click target object to run SAM",
        "r: reset | q: quit" if selected else "Click once to segment | r: reset | q: quit",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (16, 32 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_selection_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    prompt_pixel: tuple[int, int] | None,
    selected: bool,
) -> np.ndarray:
    require_cv2()
    overlay = image_bgr.copy()
    if mask is not None:
        mask_bool = mask.astype(bool)
        tint = np.zeros_like(overlay)
        tint[:, :] = (0, 255, 0)
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.4, tint, 0.6, 0.0)[mask_bool]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    if prompt_pixel is not None:
        cv2.drawMarker(overlay, prompt_pixel, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.circle(overlay, prompt_pixel, 5, (255, 255, 255), -1)
    lines = [
        "SAM mask ready, confirm or reset" if selected else "Waiting for SAM result",
        "c: confirm current mask | r: repick | q: quit" if selected else "q: quit",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (16, 32 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def build_mask_cutout(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    require_cv2()
    cutout = np.zeros_like(image_bgr)
    mask_bool = mask.astype(bool)
    cutout[mask_bool] = image_bgr[mask_bool]
    return cutout


def draw_review_panel(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt_pixel: tuple[int, int],
) -> np.ndarray:
    require_cv2()
    overlay = draw_selection_overlay(
        image_bgr=image_bgr,
        mask=mask,
        prompt_pixel=prompt_pixel,
        selected=True,
    )
    cutout = build_mask_cutout(image_bgr, mask)

    h, w = image_bgr.shape[:2]
    panel = np.zeros((h, w * 2, 3), dtype=np.uint8)
    panel[:, :w] = overlay
    panel[:, w:] = cutout

    cv2.putText(
        panel,
        "Overlay",
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Masked Object",
        (w + 16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_review_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt_pixel: tuple[int, int],
) -> np.ndarray:
    require_cv2()
    return draw_selection_overlay(
        image_bgr=image_bgr,
        mask=mask,
        prompt_pixel=prompt_pixel,
        selected=True,
    )


def build_selection_from_arrays(
    color_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    mask: np.ndarray,
    prompt_pixel: tuple[int, int],
) -> CapturedSelection:
    return CapturedSelection(
        color_bgr=color_bgr,
        color_rgb=color_bgr[:, :, ::-1].copy(),
        depth_image=depth_image,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        mask=mask.astype(bool),
        prompt_pixel=(int(prompt_pixel[0]), int(prompt_pixel[1])),
    )


def build_object_point_cloud(selection: CapturedSelection) -> Any:
    require_open3d()
    depth_m = selection.depth_image.astype(np.float32) * selection.depth_scale
    valid = selection.mask.astype(bool)
    valid &= np.isfinite(depth_m)
    valid &= depth_m > 0.0
    valid &= depth_m <= DEFAULT_DEPTH_TRUNC_M
    if not np.any(valid):
        raise RuntimeError("SAM 目标区域没有有效深度，无法生成目标点云")

    v_coords, u_coords = np.nonzero(valid)
    z = depth_m[v_coords, u_coords].astype(np.float64)
    x = (u_coords.astype(np.float64) - selection.intrinsics.cx) / selection.intrinsics.fx * z
    y = (v_coords.astype(np.float64) - selection.intrinsics.cy) / selection.intrinsics.fy * z
    points = np.column_stack([x, y, z])
    colors = selection.color_rgb[v_coords, u_coords].astype(np.float64) / 255.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    cleaned = point_cloud.remove_non_finite_points()
    if isinstance(cleaned, tuple):
        point_cloud = cleaned[0]
    elif cleaned is not None:
        point_cloud = cleaned
    if len(point_cloud.points) == 0:
        raise RuntimeError("目标点云为空，无法继续保存标签")
    return point_cloud


def build_saved_overlay_image(selection: CapturedSelection) -> np.ndarray:
    return draw_selection_overlay(selection.color_bgr, selection.mask, selection.prompt_pixel, selected=True)


def save_metadata(
    output_dir: Path,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    device: dict[str, str | int | None],
) -> None:
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "camera": device.get("name", "Intel RealSense"),
        "camera_serial": device.get("serial"),
        "camera_ip": device.get("ip"),
        "connection_type": device.get("connection_type"),
        "image_size": {
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
        "depth_scale_m_per_unit": depth_scale,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_base_to_camera_transform(handeye_path: Path) -> np.ndarray:
    if not handeye_path.exists():
        raise FileNotFoundError(f"手眼标定结果不存在: {handeye_path}")
    data = json.loads(handeye_path.read_text(encoding="utf-8"))
    if "base_to_camera_transform" not in data:
        raise ValueError("手眼标定结果缺少 base_to_camera_transform，仅支持最新格式")
    transform = np.asarray(data["base_to_camera_transform"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"base_to_camera_transform 形状异常: {transform.shape}")
    return transform


def get_current_pose(robot_controller: Any) -> list[float]:
    ret, state = robot_controller.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"读取机械臂当前位姿失败，错误码: {ret}")
    pose = state.get("pose")
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"机械臂返回 pose 格式异常: {state}")
    return [float(value) for value in pose]


def matrix_from_rm_pose(robot_controller: Any, pose: list[float]) -> np.ndarray:
    rm_matrix = robot_controller.robot.rm_algo_pos2matrix(pose)
    values = [float(value) for value in rm_matrix.data]
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def rm_pose_from_matrix(robot_controller: Any, matrix: np.ndarray) -> list[float]:
    from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t

    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot_controller.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(value) for value in pose]


def make_ee_to_gripper_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = GRIPPER_TCP_OFFSET_IN_EE
    return transform


def check_rotation_matrix(rotation: np.ndarray, name: str, tol: float = 1e-5) -> None:
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} 旋转矩阵维度错误: {rotation.shape}")
    identity = np.eye(3, dtype=np.float64)
    orth_err = np.linalg.norm(rotation.T @ rotation - identity)
    det_val = np.linalg.det(rotation)
    if orth_err > tol or abs(det_val - 1.0) > tol:
        raise ValueError(f"{name} 不是有效旋转矩阵: orth_err={orth_err:.3e}, det={det_val:.6f}")


def wait_until_confirm(prompt: str = "输入 confirm 继续: ") -> None:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "confirm":
            return
        print("未检测到 confirm，请在确认机械臂状态后输入 confirm。")


def toggle_gripper_state(robot_controller: Any, is_open: bool | None) -> bool:
    target_open = not bool(is_open) if is_open is not None else False
    target_position = 1000 if target_open else 0
    action_text = "打开" if target_open else "闭合"
    ret = robot_controller.robot.rm_set_gripper_position(target_position, False, 0)
    if ret != 0:
        raise RuntimeError(f"{action_text}夹爪失败，错误码: {ret}")
    time.sleep(1.0)
    ret, state = robot_controller.robot.rm_get_gripper_state()
    if ret == 0:
        position = state.get("actpos")
        print(f"夹爪已{action_text}，当前位置: {position}")
    else:
        print(f"夹爪已发送{action_text}命令，但读取状态失败，错误码: {ret}")
    return target_open


def wait_until_confirm_with_gripper_toggle(robot_controller: Any) -> None:
    prompt = "请手动将机械臂移动到 grasp 位姿。输入 confirm 并回车继续，按 g 切换夹爪开闭。"
    print(prompt)
    if not sys.stdin.isatty():
        print("当前不是交互式终端，无法监听 g 热键，仅支持输入 confirm。")
        wait_until_confirm()
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer = ""
    gripper_is_open: bool | None = None
    print("> ", end="", flush=True)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            char = os.read(fd, 1).decode(errors="ignore")
            if not char:
                continue
            if char in ("\n", "\r"):
                text = buffer.strip().lower()
                print()
                if text == "confirm":
                    return
                if text:
                    print("未检测到 confirm。按 g 切换夹爪，或输入 confirm 后回车继续。")
                buffer = ""
                print("> ", end="", flush=True)
                continue
            if char == "\x03":
                raise KeyboardInterrupt
            if char in ("\x7f", "\b"):
                if buffer:
                    buffer = buffer[:-1]
                    print("\b \b", end="", flush=True)
                continue
            if char.lower() == "g":
                print()
                gripper_is_open = toggle_gripper_state(robot_controller, gripper_is_open)
                buffer = ""
                print("> ", end="", flush=True)
                continue
            if char.isprintable():
                buffer += char
                print(char, end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(matrix, precision=6, suppress_small=False))


def save_label(
    label_path: Path,
    point_cloud_path: Path,
    mask_path: Path,
    overlay_path: Path,
    gripper_center_in_capture_camera: np.ndarray,
    gripper_rotation_in_capture_camera: np.ndarray,
    gripper_width: float,
    ee_to_gripper_transform: np.ndarray,
    base_to_camera_at_capture: np.ndarray,
    base_to_ee_at_capture: np.ndarray,
    base_to_ee_at_grasp: np.ndarray,
    base_to_gripper_at_grasp: np.ndarray,
    base_to_camera_transform: np.ndarray,
    camera_capture_to_ee_grasp: np.ndarray,
    camera_capture_to_gripper_grasp: np.ndarray,
    robot_controller: Any,
    handeye_result_path: Path,
    selection: CapturedSelection,
    sam_checkpoint_path: str,
) -> None:
    gripper_direction = gripper_rotation_in_capture_camera[:, 2]
    payload = {
        "ply_path": str(point_cloud_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "sam_overlay_path": str(overlay_path.resolve()),
        "gripper_contact_center": gripper_center_in_capture_camera.tolist(),
        "gripper_rotation_matrix": gripper_rotation_in_capture_camera.tolist(),
        "gripper_direction": gripper_direction.tolist(),
        "gripper_width": float(gripper_width),
        "arrow_position": gripper_center_in_capture_camera.tolist(),
        "arrow_direction": gripper_direction.tolist(),
        "gripper_tcp_offset_in_ee": GRIPPER_TCP_OFFSET_IN_EE.tolist(),
        "ee_to_gripper_transform": ee_to_gripper_transform.tolist(),
        "base_to_camera_at_capture": base_to_camera_at_capture.tolist(),
        "base_to_camera_at_capture_pose_rm": rm_pose_from_matrix(robot_controller, base_to_camera_at_capture),
        "base_to_ee_at_capture": base_to_ee_at_capture.tolist(),
        "base_to_ee_at_grasp": base_to_ee_at_grasp.tolist(),
        "base_to_gripper_at_grasp": base_to_gripper_at_grasp.tolist(),
        "base_to_gripper_at_grasp_pose_rm": rm_pose_from_matrix(robot_controller, base_to_gripper_at_grasp),
        "base_to_camera_transform": base_to_camera_transform.tolist(),
        "camera_capture_to_ee_grasp": camera_capture_to_ee_grasp.tolist(),
        "camera_capture_to_ee_grasp_pose_rm": rm_pose_from_matrix(robot_controller, camera_capture_to_ee_grasp),
        "camera_capture_to_gripper_grasp": camera_capture_to_gripper_grasp.tolist(),
        "camera_capture_to_gripper_grasp_pose_rm": rm_pose_from_matrix(robot_controller, camera_capture_to_gripper_grasp),
        "sam_prompt_pixel": [int(selection.prompt_pixel[0]), int(selection.prompt_pixel[1])],
        "sam_model_type": SAM_MODEL_TYPE,
        "sam_checkpoint_path": sam_checkpoint_path,
        "sam_mask_area_px": int(np.count_nonzero(selection.mask)),
        "handeye_result_path": str(handeye_result_path.resolve()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def show_pyvista_preview(
    point_cloud: Any,
    gripper_center_in_capture_camera: np.ndarray,
    gripper_rotation_in_capture_camera: np.ndarray,
    gripper_width: float,
    camera_capture_to_ee_grasp: np.ndarray,
) -> None:
    require_pyvista()
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise RuntimeError("点云为空，无法显示 PyVista 预览")
    colors = np.asarray(point_cloud.colors)
    if colors.shape[0] != points.shape[0]:
        colors = np.repeat(np.asarray([[95.0, 141.0, 211.0]], dtype=np.float64), len(points), axis=0)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    frame_size = max(diag * 0.18, 0.045)
    tcp_radius = max(frame_size * 0.08, 0.004)
    finger_length = max(float(GRIPPER_TCP_OFFSET_IN_EE[2]) * 0.35, 0.05)

    x_axis = gripper_rotation_in_capture_camera[:, 0]
    y_axis = gripper_rotation_in_capture_camera[:, 1]
    z_axis = gripper_rotation_in_capture_camera[:, 2]
    half_width = float(gripper_width) * 0.5
    p_left = gripper_center_in_capture_camera - x_axis * half_width
    p_right = gripper_center_in_capture_camera + x_axis * half_width
    p_left_back = p_left - z_axis * finger_length
    p_right_back = p_right - z_axis * finger_length
    p_left_outer = p_left + y_axis * half_width * 0.45
    p_right_outer = p_right + y_axis * half_width * 0.45
    p_left_back_outer = p_left_back + y_axis * half_width * 0.45
    p_right_back_outer = p_right_back + y_axis * half_width * 0.45

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("#0f1720")
    cloud = pv.PolyData(points)
    cloud["rgb"] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    plotter.add_points(cloud, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)
    plotter.add_axes(line_width=3, labels_off=False)
    plotter.show_grid(color="#6b7280")
    plotter.add_text("PyVista Label Preview | Target point cloud + gripper pose", position="upper_left", font_size=11, color="white")

    for name, color, axis_vec in (
        ("gripper_x", "#ff4d4d", x_axis),
        ("gripper_y", "#53d769", y_axis),
        ("gripper_z", "#4da3ff", z_axis),
    ):
        plotter.add_mesh(
            pv.Line(gripper_center_in_capture_camera.astype(float), (gripper_center_in_capture_camera + axis_vec * frame_size).astype(float)),
            color=color,
            line_width=5,
            name=name,
            render_lines_as_tubes=True,
        )

    for idx, (start, end) in enumerate((
        (p_left, p_right),
        (p_left, p_left_back),
        (p_right, p_right_back),
        (p_left_back, p_right_back),
        (p_left_outer, p_left_back_outer),
        (p_right_outer, p_right_back_outer),
        (p_left_back_outer, p_right_back_outer),
    )):
        plotter.add_mesh(
            pv.Line(start.astype(float), end.astype(float)),
            color="#00d7ff",
            line_width=4,
            name=f"gripper_segment_{idx}",
            render_lines_as_tubes=True,
        )

    plotter.add_mesh(
        pv.Sphere(radius=tcp_radius, center=gripper_center_in_capture_camera.astype(float)),
        color="#ff3b30",
        smooth_shading=True,
        name="gripper_tcp",
    )

    if camera_capture_to_ee_grasp.shape == (4, 4):
        ee_center = camera_capture_to_ee_grasp[:3, 3]
        plotter.add_mesh(
            pv.Sphere(radius=tcp_radius * 0.75, center=ee_center.astype(float)),
            color="#6b7280",
            smooth_shading=True,
            name="ee_center",
        )
        plotter.add_mesh(
            pv.Line(ee_center.astype(float), gripper_center_in_capture_camera.astype(float)),
            color="#9ca3af",
            line_width=3,
            name="ee_to_tcp",
            render_lines_as_tubes=True,
        )

    plotter.camera_position = "iso"
    plotter.show()


def show_registration_pyvista_preview(
    source_point_cloud: Any,
    target_point_cloud: Any,
    transformation: np.ndarray,
    gripper_center_in_capture_camera: np.ndarray,
    gripper_rotation_in_capture_camera: np.ndarray,
    gripper_width: float,
    camera_capture_to_ee_grasp: np.ndarray,
) -> None:
    require_pyvista()
    source_points = np.asarray(source_point_cloud.points)
    target_points = np.asarray(target_point_cloud.points)
    if source_points.size == 0 or target_points.size == 0:
        raise RuntimeError("点云为空，无法显示 ICP PyVista 预览")

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    aligned_source_points = source_points @ rotation.T + translation

    combined_points = np.vstack([aligned_source_points, target_points])
    bbox_min = combined_points.min(axis=0)
    bbox_max = combined_points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    frame_size = max(diag * 0.18, 0.045)
    tcp_radius = max(frame_size * 0.08, 0.004)
    finger_length = max(float(GRIPPER_TCP_OFFSET_IN_EE[2]) * 0.35, 0.05)

    x_axis = gripper_rotation_in_capture_camera[:, 0]
    y_axis = gripper_rotation_in_capture_camera[:, 1]
    z_axis = gripper_rotation_in_capture_camera[:, 2]
    half_width = float(gripper_width) * 0.5
    p_left = gripper_center_in_capture_camera - x_axis * half_width
    p_right = gripper_center_in_capture_camera + x_axis * half_width
    p_left_back = p_left - z_axis * finger_length
    p_right_back = p_right - z_axis * finger_length
    p_left_outer = p_left + y_axis * half_width * 0.45
    p_right_outer = p_right + y_axis * half_width * 0.45
    p_left_back_outer = p_left_back + y_axis * half_width * 0.45
    p_right_back_outer = p_right_back + y_axis * half_width * 0.45

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("#0f1720")
    plotter.add_axes(line_width=3, labels_off=False)
    plotter.show_grid(color="#6b7280")
    plotter.add_text("PyVista ICP Preview | Orange: aligned source | Blue: target", position="upper_left", font_size=11, color="white")

    aligned_source_cloud = pv.PolyData(aligned_source_points)
    target_cloud = pv.PolyData(target_points)
    plotter.add_points(
        aligned_source_cloud,
        color="#f59e0b",
        point_size=3,
        render_points_as_spheres=True,
        name="aligned_source_cloud",
    )
    plotter.add_points(
        target_cloud,
        color="#38bdf8",
        point_size=3,
        render_points_as_spheres=True,
        name="target_cloud",
    )

    for name, color, axis_vec in (
        ("gripper_x", "#ff4d4d", x_axis),
        ("gripper_y", "#53d769", y_axis),
        ("gripper_z", "#4da3ff", z_axis),
    ):
        plotter.add_mesh(
            pv.Line(gripper_center_in_capture_camera.astype(float), (gripper_center_in_capture_camera + axis_vec * frame_size).astype(float)),
            color=color,
            line_width=5,
            name=name,
            render_lines_as_tubes=True,
        )

    for idx, (start, end) in enumerate((
        (p_left, p_right),
        (p_left, p_left_back),
        (p_right, p_right_back),
        (p_left_back, p_right_back),
        (p_left_outer, p_left_back_outer),
        (p_right_outer, p_right_back_outer),
        (p_left_back_outer, p_right_back_outer),
    )):
        plotter.add_mesh(
            pv.Line(start.astype(float), end.astype(float)),
            color="#00d7ff",
            line_width=4,
            name=f"gripper_segment_{idx}",
            render_lines_as_tubes=True,
        )

    plotter.add_mesh(
        pv.Sphere(radius=tcp_radius, center=gripper_center_in_capture_camera.astype(float)),
        color="#ff3b30",
        smooth_shading=True,
        name="gripper_tcp",
    )

    if camera_capture_to_ee_grasp.shape == (4, 4):
        ee_center = camera_capture_to_ee_grasp[:3, 3]
        plotter.add_mesh(
            pv.Sphere(radius=tcp_radius * 0.75, center=ee_center.astype(float)),
            color="#6b7280",
            smooth_shading=True,
            name="ee_center",
        )
        plotter.add_mesh(
            pv.Line(ee_center.astype(float), gripper_center_in_capture_camera.astype(float)),
            color="#9ca3af",
            line_width=3,
            name="ee_to_tcp",
            render_lines_as_tubes=True,
        )

    plotter.camera_position = "iso"
    plotter.show()
