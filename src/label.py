from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional runtime dependency
    rs = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional runtime dependency
    o3d = None

from core.demo_project import RobotArmController
from align.reconstruct import capture_single_frame, list_realsense_devices

try:
    from src.Robotic_Arm.rm_ctypes_wrap import rm_matrix_t
except ImportError:
    from Robotic_Arm.rm_ctypes_wrap import rm_matrix_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重建点云并生成抓取标签（拍摄坐标系下的末端位姿）"
    )
    parser.add_argument("--robot-ip", default="169.254.128.18", help="机械臂 IP")
    parser.add_argument("--robot-port", type=int, default=8080, help="机械臂端口")

    parser.add_argument("--camera-serial", default=None, help="指定 RealSense 相机序列号")
    parser.add_argument("--camera-index", type=int, default=0, help="按检测顺序选择相机，从 0 开始")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用相机后退出")
    parser.add_argument("--no-preview", action="store_true", help="跳过拍摄前实时预览")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="重建输出目录")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度")
    parser.add_argument("--height", type=int, default=720, help="图像高度")
    parser.add_argument("--fps", type=int, default=15, help="采集帧率")
    parser.add_argument("--warmup-frames", type=int, default=30, help="预热帧数")
    parser.add_argument("--depth-trunc", type=float, default=3.0, help="重建最大深度（米）")
    parser.add_argument("--preview-width", type=int, default=640, help="预览图像宽度")
    parser.add_argument("--preview-height", type=int, default=480, help="预览图像高度")
    parser.add_argument("--preview-fps", type=int, default=30, help="预览帧率")

    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        default=[-0.2, -0.2, 0.3],
        metavar=("X", "Y", "Z"),
        help="点云裁剪包围盒最小值",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        default=[0.2, 0.2, 0.7],
        metavar=("X", "Y", "Z"),
        help="点云裁剪包围盒最大值",
    )

    parser.add_argument("--no-ground-removal", action="store_true", help="关闭地面去除")
    parser.add_argument("--ground-distance-threshold", type=float, default=0.01, help="地面分割距离阈值")
    parser.add_argument("--ground-ransac-n", type=int, default=3, help="地面分割 RANSAC 采样点数")
    parser.add_argument("--ground-num-iterations", type=int, default=1000, help="地面分割迭代次数")

    parser.add_argument("--no-outlier-removal", action="store_true", help="关闭离群点去除")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=30, help="离群点去除邻居数")
    parser.add_argument("--outlier-std-ratio", type=float, default=1.5, help="离群点标准差倍率")
    parser.add_argument("--vis", action="store_true", help="保存离线叠加预览图（PNG）")

    parser.add_argument(
        "--handeye-result-path",
        type=Path,
        default=Path("handeye_output") / "handeye_result.json",
        help="手眼标定结果 JSON 路径",
    )
    parser.add_argument("--gripper-width", type=float, default=0.02, help="保存到标签中的夹爪开度（米）")
    parser.add_argument(
        "--consistency-tol",
        type=float,
        default=1e-6,
        help="坐标链路一致性检查阈值（Frobenius 范数）",
    )
    return parser.parse_args()


def load_camera_to_gripper_transform(handeye_path: Path) -> np.ndarray:
    if not handeye_path.exists():
        raise FileNotFoundError(f"手眼标定结果不存在: {handeye_path}")

    data = json.loads(handeye_path.read_text(encoding="utf-8"))
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
        raise RuntimeError(f"机械臂返回 pose 格式异常: {state}")
    return [float(value) for value in pose]


def matrix_from_rm_pose(robot_controller: RobotArmController, pose: list[float]) -> np.ndarray:
    rm_matrix = robot_controller.robot.rm_algo_pos2matrix(pose)
    values = [float(value) for value in rm_matrix.data]
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def rm_pose_from_matrix(robot_controller: RobotArmController, matrix: np.ndarray) -> list[float]:
    rm_mat = rm_matrix_t(data=matrix.tolist())
    pose = robot_controller.robot.rm_algo_matrix2pos(rm_mat, 1)
    return [float(value) for value in pose]


def check_rotation_matrix(rotation: np.ndarray, name: str, tol: float = 1e-5) -> None:
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} 旋转矩阵维度错误: {rotation.shape}")
    identity = np.eye(3, dtype=np.float64)
    orth_err = np.linalg.norm(rotation.T @ rotation - identity)
    det_val = np.linalg.det(rotation)
    if orth_err > tol or abs(det_val - 1.0) > tol:
        raise ValueError(
            f"{name} 不是有效旋转矩阵: orth_err={orth_err:.3e}, det={det_val:.6f}"
        )


def wait_until_confirm(prompt: str = "输入 confirm 继续: ") -> None:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "confirm":
            return
        print("未检测到 confirm，请在确认机械臂状态后输入 confirm。")


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(matrix, precision=6, suppress_small=False))


def resolve_camera_serial(camera_serial: str | None, camera_index: int) -> str:
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("未发现任何 RealSense 相机")

    if camera_serial:
        for device in devices:
            if device.get("serial") == camera_serial:
                return str(camera_serial)
        available = ", ".join(str(d.get("serial", "unknown")) for d in devices)
        raise RuntimeError(f"找不到序列号为 {camera_serial} 的相机，当前可用: {available}")

    if camera_index < 0 or camera_index >= len(devices):
        raise RuntimeError(f"camera-index={camera_index} 超出范围，当前共有 {len(devices)} 台相机")

    selected_serial = devices[camera_index].get("serial")
    if not selected_serial:
        raise RuntimeError(f"第 {camera_index} 台相机缺少序列号，无法继续")
    return str(selected_serial)


def preview_camera_until_confirm(
    camera_serial: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法显示实时预览")
    if rs is None:
        raise RuntimeError("未安装 pyrealsense2，无法显示实时预览")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camera_serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    window_name = "Label Preview (press c to capture, q to quit)"
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.putText(
                color_image,
                "Move robot to capture view | Press c to continue, q to abort",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                return
            if key == ord("q"):
                raise RuntimeError("用户取消了拍摄前预览流程")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def save_label(
    label_path: Path,
    point_cloud_path: Path,
    gripper_center_in_capture_camera: np.ndarray,
    gripper_rotation_in_capture_camera: np.ndarray,
    gripper_width: float,
    base_to_camera_at_capture: np.ndarray,
    base_to_ee_at_capture: np.ndarray,
    base_to_ee_at_grasp: np.ndarray,
    camera_to_gripper_transform: np.ndarray,
    camera_capture_to_ee_grasp: np.ndarray,
    robot_controller: RobotArmController,
    handeye_result_path: Path,
) -> None:
    gripper_direction = gripper_rotation_in_capture_camera[:, 2]
    payload = {
        "ply_path": str(point_cloud_path.resolve()),
        "gripper_contact_center": gripper_center_in_capture_camera.tolist(),
        "gripper_rotation_matrix": gripper_rotation_in_capture_camera.tolist(),
        "gripper_direction": gripper_direction.tolist(),
        "gripper_width": float(gripper_width),
        "arrow_position": gripper_center_in_capture_camera.tolist(),
        "arrow_direction": gripper_direction.tolist(),
        "base_to_camera_at_capture": base_to_camera_at_capture.tolist(),
        "base_to_camera_at_capture_pose_rm": rm_pose_from_matrix(
            robot_controller, base_to_camera_at_capture
        ),
        "base_to_ee_at_capture": base_to_ee_at_capture.tolist(),
        "base_to_ee_at_grasp": base_to_ee_at_grasp.tolist(),
        "camera_to_gripper_transform": camera_to_gripper_transform.tolist(),
        "camera_capture_to_ee_grasp": camera_capture_to_ee_grasp.tolist(),
        "camera_capture_to_ee_grasp_pose_rm": rm_pose_from_matrix(
            robot_controller, camera_capture_to_ee_grasp
        ),
        "handeye_result_path": str(handeye_result_path.resolve()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    label_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_offline_preview_from_label(label_path: Path) -> Path | None:
    if o3d is None:
        print("未安装 open3d，跳过离线渲染")
        return None

    if not label_path.exists():
        print(f"label 文件不存在，无法可视化: {label_path}")
        return None

    payload = json.loads(label_path.read_text(encoding="utf-8"))
    point_cloud_path = Path(payload["ply_path"])
    gripper_center_in_capture_camera = np.asarray(
        payload["gripper_contact_center"], dtype=np.float64
    )
    gripper_rotation_in_capture_camera = np.asarray(
        payload["gripper_rotation_matrix"], dtype=np.float64
    )
    gripper_width = float(payload.get("gripper_width", 0.02))

    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    if len(point_cloud.points) == 0:
        print(f"点云为空，跳过离线渲染: {point_cloud_path}")
        return None

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    n_orig = points.shape[0]
    if n_orig > 30000:
        sample_idx = np.random.default_rng(0).choice(n_orig, size=30000, replace=False)
        points = points[sample_idx]
        if colors.shape[0] == n_orig:
            colors = colors[sample_idx]

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    axis_len = max(diag * 0.12, 0.04)

    x_axis = gripper_rotation_in_capture_camera[:, 0]
    y_axis = gripper_rotation_in_capture_camera[:, 1]
    z_axis = gripper_rotation_in_capture_camera[:, 2]

    preview_path = label_path.with_name("label_preview.png")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if colors.shape[0] == points.shape[0] and colors.shape[1] == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.0, alpha=0.7)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#5f8dd3", s=1.0, alpha=0.7)

    cx, cy, cz = gripper_center_in_capture_camera
    ax.scatter([cx], [cy], [cz], c="red", s=80, label="gripper_contact_center")

    # 抓取局部坐标轴
    ax.plot([cx, cx + x_axis[0] * axis_len], [cy, cy + x_axis[1] * axis_len], [cz, cz + x_axis[2] * axis_len], c="r", linewidth=2, label="gripper_x")
    ax.plot([cx, cx + y_axis[0] * axis_len], [cy, cy + y_axis[1] * axis_len], [cz, cz + y_axis[2] * axis_len], c="g", linewidth=2, label="gripper_y")
    ax.plot([cx, cx + z_axis[0] * axis_len], [cy, cy + z_axis[1] * axis_len], [cz, cz + z_axis[2] * axis_len], c="b", linewidth=2, label="gripper_z")

    # 夹爪开度线（沿局部X）
    half_width = gripper_width * 0.5
    p_left = gripper_center_in_capture_camera - x_axis * half_width
    p_right = gripper_center_in_capture_camera + x_axis * half_width
    ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], [p_left[2], p_right[2]], c="c", linewidth=2, label="gripper_width")

    ax.set_title("PLY + Label (Capture Camera Frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=24, azim=35)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=200)
    plt.close(fig)

    return preview_path


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
            return
        for device in devices:
            print(
                "index={index} name={name} serial={serial} ip={ip} firmware={firmware} connection={connection_type}".format(
                    **device
                )
            )
        return

    camera_to_gripper_transform = load_camera_to_gripper_transform(args.handeye_result_path)
    selected_camera_serial = resolve_camera_serial(args.camera_serial, args.camera_index)

    robot_controller = RobotArmController(args.robot_ip, args.robot_port, level=3)
    label_path_to_visualize: Path | None = None
    try:
        if not args.no_preview:
            print("打开实时预览中，请在预览窗口按 c 锁定视角并继续。")
            preview_camera_until_confirm(
                camera_serial=selected_camera_serial,
                width=args.preview_width,
                height=args.preview_height,
                fps=args.preview_fps,
            )
        else:
            print("已跳过实时预览，请将机械臂移动到拍摄视角后按 Enter。")
            input()

        capture_pose_rm = get_current_pose(robot_controller)
        base_to_ee_at_capture = matrix_from_rm_pose(robot_controller, capture_pose_rm)

        point_cloud_path = capture_single_frame(
            bbox_min=args.bbox_min,
            bbox_max=args.bbox_max,
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_serial=selected_camera_serial,
            camera_index=args.camera_index,
            warmup_frames=args.warmup_frames,
            depth_trunc=args.depth_trunc,
            ground_removal=not args.no_ground_removal,
            ground_distance_threshold=args.ground_distance_threshold,
            ground_ransac_n=args.ground_ransac_n,
            ground_num_iterations=args.ground_num_iterations,
            outlier_removal=not args.no_outlier_removal,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            vis=False,
        )

        base_to_camera_at_capture = base_to_ee_at_capture @ np.linalg.inv(camera_to_gripper_transform)

        print(
            "请手动将机械臂移动到 grasp 位姿。"
            "移动完成后，在终端输入 confirm 并回车继续。"
        )
        wait_until_confirm()

        grasp_pose_rm = get_current_pose(robot_controller)
        base_to_ee_at_grasp = matrix_from_rm_pose(robot_controller, grasp_pose_rm)

        camera_capture_to_ee_grasp = (
            np.linalg.inv(base_to_camera_at_capture) @ base_to_ee_at_grasp
        )

        camera_capture_to_ee_grasp_equivalent = (
            camera_to_gripper_transform
            @ np.linalg.inv(base_to_ee_at_capture)
            @ base_to_ee_at_grasp
        )
        chain_error = np.linalg.norm(
            camera_capture_to_ee_grasp - camera_capture_to_ee_grasp_equivalent
        )
        if chain_error > args.consistency_tol:
            raise RuntimeError(
                "坐标链路一致性检查失败，"
                f"error={chain_error:.3e} > tol={args.consistency_tol:.3e}"
            )

        gripper_rotation_in_capture_camera = camera_capture_to_ee_grasp[:3, :3]
        gripper_center_in_capture_camera = camera_capture_to_ee_grasp[:3, 3]

        check_rotation_matrix(gripper_rotation_in_capture_camera, "camera_capture_to_ee_grasp")
        check_rotation_matrix(base_to_camera_at_capture[:3, :3], "base_to_camera_at_capture")

        label_path = point_cloud_path.parent / "label.json"
        save_label(
            label_path=label_path,
            point_cloud_path=point_cloud_path,
            gripper_center_in_capture_camera=gripper_center_in_capture_camera,
            gripper_rotation_in_capture_camera=gripper_rotation_in_capture_camera,
            gripper_width=args.gripper_width,
            base_to_camera_at_capture=base_to_camera_at_capture,
            base_to_ee_at_capture=base_to_ee_at_capture,
            base_to_ee_at_grasp=base_to_ee_at_grasp,
            camera_to_gripper_transform=camera_to_gripper_transform,
            camera_capture_to_ee_grasp=camera_capture_to_ee_grasp,
            robot_controller=robot_controller,
            handeye_result_path=args.handeye_result_path,
        )

        print(f"坐标链路一致性误差: {chain_error:.3e}")
        print_matrix("base_to_camera_at_capture", base_to_camera_at_capture)
        print_matrix("camera_capture_to_ee_grasp", camera_capture_to_ee_grasp)
        print(f"point_cloud: {point_cloud_path}")
        print(f"label: {label_path}")

        if args.vis:
            label_path_to_visualize = label_path
    finally:
        robot_controller.disconnect()

    if args.vis and label_path_to_visualize is not None:
        preview_path = render_offline_preview_from_label(label_path_to_visualize)
        if preview_path is not None:
            print(f"offline preview: {preview_path}")


if __name__ == "__main__":
    main()
