from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency on target machine
    rs = None


def require_realsense() -> None:
    if rs is None:
        raise RuntimeError(
            "未检测到 pyrealsense2。请先安装 RealSense Python SDK，例如：pip install pyrealsense2"
        )


def get_device_info(device: rs.device, info: rs.camera_info) -> str | None:
    if not device.supports(info):
        return None
    value = device.get_info(info)
    return value or None


def list_realsense_devices() -> list[dict[str, str | int | None]]:
    require_realsense()
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


def select_camera(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集一帧 RealSense RGB-D 并重建点云")
    parser.add_argument("--camera-serial", default=None, help="手动指定 RealSense 相机序列号")
    parser.add_argument("--camera-index", type=int, default=0, help="按检测顺序选择第几台相机，从 0 开始")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用 RealSense 设备后退出")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度")
    parser.add_argument("--height", type=int, default=720, help="图像高度")
    parser.add_argument("--fps", type=int, default=15, help="采集帧率")
    parser.add_argument("--warmup-frames", type=int, default=30, help="预热帧数")
    parser.add_argument("--depth-trunc", type=float, default=3.0, help="点云重建最大深度，单位米")
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
    parser.add_argument("--vis", action="store_true", help="显示点云")
    return parser.parse_args()

def save_metadata(
    output_dir: Path,
    intrinsics: rs.intrinsics,
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
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
        },
        "depth_scale_m_per_unit": depth_scale,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def build_point_cloud_from_realsense(
    depth_frame: rs.depth_frame,
    color_frame: rs.video_frame,
    color_rgb: np.ndarray,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    pointcloud = rs.pointcloud()
    pointcloud.map_to(color_frame)
    points = pointcloud.calculate(depth_frame)

    vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    texcoords = np.asarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    valid_mask = np.isfinite(vertices).all(axis=1)
    valid_mask &= vertices[:, 2] > 0.0
    valid_mask &= vertices[:, 2] <= depth_trunc
    if not np.any(valid_mask):
        return o3d.geometry.PointCloud()

    vertices = vertices[valid_mask].astype(np.float64, copy=False)
    texcoords = texcoords[valid_mask]

    image_height, image_width = color_rgb.shape[:2]
    u = np.clip(np.rint(texcoords[:, 0] * (image_width - 1)).astype(np.int32), 0, image_width - 1)
    v = np.clip(np.rint(texcoords[:, 1] * (image_height - 1)).astype(np.int32), 0, image_height - 1)
    colors = color_rgb[v, u].astype(np.float64) / 255.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def apply_bbox_filter(
    point_cloud: o3d.geometry.PointCloud,
    bbox_min: list[float] | None,
    bbox_max: list[float] | None,
) -> o3d.geometry.PointCloud:
    if bbox_min is None or bbox_max is None:
        raise ValueError("--bbox-min and --bbox-max must be provided together.")

    min_bound = np.asarray(bbox_min, dtype=np.float64)
    max_bound = np.asarray(bbox_max, dtype=np.float64)
    if np.any(min_bound >= max_bound):
        raise ValueError("Each value in --bbox-min must be smaller than --bbox-max.")

    points = np.asarray(point_cloud.points)
    if points.size == 0:
        return point_cloud

    mask = np.all(points >= min_bound, axis=1) & np.all(points <= max_bound, axis=1)
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(points[mask])

    colors = np.asarray(point_cloud.colors)
    if colors.shape[0] == points.shape[0]:
        filtered_cloud.colors = o3d.utility.Vector3dVector(colors[mask])

    normals = np.asarray(point_cloud.normals)
    if normals.shape[0] == points.shape[0]:
        filtered_cloud.normals = o3d.utility.Vector3dVector(normals[mask])

    return filtered_cloud


def apply_ground_removal(
    point_cloud: o3d.geometry.PointCloud,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> o3d.geometry.PointCloud:
    if len(point_cloud.points) < ransac_n:
        return point_cloud

    _, inlier_indices = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if not inlier_indices:
        return point_cloud
    return point_cloud.select_by_index(inlier_indices, invert=True)


def apply_outlier_removal(
    point_cloud: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    if len(point_cloud.points) < nb_neighbors:
        return point_cloud
    filtered_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered_cloud


def capture_single_frame(
    bbox_min: list[float] = [-0.2, -0.2, 0.3],
    bbox_max: list[float] = [0.2, 0.2, 0.7],
    output_dir: Path = Path("outputs"),
    width: int = 1280,
    height: int = 720,
    fps: int = 15,
    camera_serial: str | None = None,
    camera_index: int = 0,
    warmup_frames: int = 30,
    depth_trunc: float = 3.0,
    ground_removal: bool = True,
    ground_distance_threshold: float = 0.01,
    ground_ransac_n: int = 3,
    ground_num_iterations: int = 1000,
    outlier_removal: bool = True,
    outlier_nb_neighbors: int = 30,
    outlier_std_ratio: float = 1.5,
    vis: bool = False,
) -> Path:
    """Capture one aligned RGB-D frame from a D435i and reconstruct a point cloud.

    Returns the path to the saved .ply file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = list_realsense_devices()
    camera = select_camera(devices, camera_serial, camera_index)
    selected_camera_serial = camera.get("serial")
    if not selected_camera_serial:
        raise RuntimeError(f"选中的 RealSense 相机缺少序列号，无法启动: {camera}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(selected_camera_serial))
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    align = rs.align(rs.stream.color)
    started = False

    try:
        profile = pipeline.start(config)
        started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture synchronized color and depth frames.")

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1].copy()
        depth_image = np.asanyarray(depth_frame.get_data())

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_image)
        point_cloud = build_point_cloud_from_realsense(
            depth_frame=depth_frame,
            color_frame=color_frame,
            color_rgb=color_rgb,
            depth_trunc=depth_trunc,
        )
        cleaned = point_cloud.remove_non_finite_points()
        if isinstance(cleaned, tuple):
            point_cloud = cleaned[0]
        elif cleaned is not None:
            point_cloud = cleaned

        point_cloud = apply_bbox_filter(point_cloud, bbox_min, bbox_max)

        if ground_removal:
            point_cloud = apply_ground_removal(
                point_cloud=point_cloud,
                distance_threshold=ground_distance_threshold,
                ransac_n=ground_ransac_n,
                num_iterations=ground_num_iterations,
            )

        if outlier_removal:
            point_cloud = apply_outlier_removal(
                point_cloud=point_cloud,
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )

        color_path = out_dir / "color.png"
        depth_path = out_dir / "depth.png"
        point_cloud_path = out_dir / "point_cloud.ply"

        o3d.io.write_image(str(color_path), o3d_color)
        o3d.io.write_image(str(depth_path), o3d_depth)
        o3d.io.write_point_cloud(str(point_cloud_path), point_cloud, write_ascii=False)
        save_metadata(out_dir, intrinsics, depth_scale, camera)

        print(f"Using RealSense: name={camera.get('name')} serial={selected_camera_serial} ip={camera.get('ip')}")
        print(f"Saved color image: {color_path}")
        print(f"Saved depth image: {depth_path}")
        print(f"Saved point cloud: {point_cloud_path}")
        print(f"Final point count: {len(point_cloud.points)}")

        if vis:
            o3d.visualization.draw_geometries([point_cloud], window_name="D435i Single Frame Point Cloud")

        return point_cloud_path
    finally:
        if started:
            pipeline.stop()


if __name__ == "__main__":
    args = parse_args()
    if args.list_cameras:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
        else:
            for device in devices:
                print(
                    "index={index} name={name} serial={serial} ip={ip} firmware={firmware} connection={connection_type}".format(
                        **device
                    )
                )
        raise SystemExit(0)

    ply_path = capture_single_frame(
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_serial=args.camera_serial,
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
        vis=args.vis,
    )
    print(f"Done: {ply_path}")
