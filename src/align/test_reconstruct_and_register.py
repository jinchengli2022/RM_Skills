from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.align.ply_registration import register_point_clouds
    from src.align.reconstruct import capture_single_frame
else:
    from .ply_registration import register_point_clouds
    from .reconstruct import capture_single_frame


DEFAULT_SOURCE = Path("/home/rm/ljc/RM_Skills/outputs/kettle_source_right/point_cloud.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="先重建点云，再输出抓取姿态在相机坐标系下的 4x4 齐次矩阵")
    parser.add_argument("--camera-serial", required=True, help="手动指定 RealSense 相机序列号")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="source 点云路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="重建输出目录")
    parser.add_argument("--bbox-min", type=float, nargs=3, default=[-0.2, -0.2, 0.3], metavar=("X", "Y", "Z"))
    parser.add_argument("--bbox-max", type=float, nargs=3, default=[0.2, 0.2, 0.7], metavar=("X", "Y", "Z"))
    parser.add_argument("--voxel-size", type=float, default=0.01, help="配准下采样体素大小")
    parser.add_argument("--icp-distance-factor", type=float, default=2.0, help="ICP 对应点距离倍数")
    return parser.parse_args()


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


def reconstruct_and_register_grasp_pose(
    source_ply: str | Path,
    camera_serial: str,
    *,
    output_dir: str | Path = "outputs",
    bbox_min: list[float] | tuple[float, float, float] = (-0.2, -0.2, 0.3),
    bbox_max: list[float] | tuple[float, float, float] = (0.2, 0.2, 0.7),
    voxel_size: float = 0.01,
    icp_distance_factor: float = 2.0,
) -> np.ndarray:
    source_path = Path(source_ply)
    if not source_path.exists():
        raise FileNotFoundError(f"source 点云不存在: {source_path}")

    target_ply = capture_single_frame(
        bbox_min=list(bbox_min),
        bbox_max=list(bbox_max),
        output_dir=Path(output_dir),
        camera_serial=camera_serial,
        vis=False,
    )

    grasp_result = register_point_clouds(
        source=source_path,
        target=target_ply,
        voxel_size=voxel_size,
        icp_distance_factor=icp_distance_factor,
        no_vis=True,
    )
    if grasp_result is None:
        raise ValueError(f"source 点云目录下缺少可用的 label.json: {source_path.parent}")

    return grasp_result_to_camera_pose_matrix(grasp_result)


def main() -> None:
    args = parse_args()
    pose_matrix = reconstruct_and_register_grasp_pose(
        source_ply=args.source,
        camera_serial=args.camera_serial,
        output_dir=args.output_dir,
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        voxel_size=args.voxel_size,
        icp_distance_factor=args.icp_distance_factor,
    )
    print("camera grasp pose matrix:")
    print(json.dumps(pose_matrix.tolist(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
