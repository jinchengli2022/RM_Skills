from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    matplotlib = None
    plt = None


@dataclass
class RegistrationSummary:
    fitness: float
    inlier_rmse: float
    transformation: np.ndarray


def _sample_line(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 1:
        return start.reshape(1, 3)
    weights = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    return start[None, :] * (1.0 - weights) + end[None, :] * weights



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register two PLY point clouds with RANSAC + ICP and visualize the result.",
    )
    parser.add_argument("source", type=Path, help="Path to the source PLY point cloud.")
    parser.add_argument("target", type=Path, help="Path to the target PLY point cloud.")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size for downsampling.")
    parser.add_argument(
        "--icp-distance-factor",
        type=float,
        default=2.0,
        help="ICP correspondence distance multiplier based on voxel size.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip Open3D visualization windows.",
    )
    return parser.parse_args()


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {path}")

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {path}")

    cleaned = point_cloud.remove_non_finite_points()
    if isinstance(cleaned, tuple):
        point_cloud = cleaned[0]
    elif cleaned is not None:
        point_cloud = cleaned

    if point_cloud.is_empty():
        raise ValueError(f"Point cloud contains no valid points after cleaning: {path}")

    return point_cloud


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.copy()
    return vector / norm


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)

    x, y, z = axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    one_minus_cos = 1.0 - cos_theta
    return np.array(
        [
            [
                cos_theta + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_theta,
                x * z * one_minus_cos + y * sin_theta,
            ],
            [
                y * x * one_minus_cos + z * sin_theta,
                cos_theta + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_theta,
            ],
            [
                z * x * one_minus_cos - y * sin_theta,
                z * y * one_minus_cos + x * sin_theta,
                cos_theta + z * z * one_minus_cos,
            ],
        ],
        dtype=np.float64,
    )


def rotation_from_z(direction: np.ndarray) -> np.ndarray:
    src = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dst = normalize(direction)

    dot_val = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if abs(dot_val - 1.0) < 1e-8:
        return np.eye(3)
    if abs(dot_val + 1.0) < 1e-8:
        return axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), np.pi)

    axis = np.cross(src, dst)
    angle = float(np.arccos(dot_val))
    return axis_angle_to_matrix(axis, angle)


def load_grasp_label(source_path: Path) -> tuple[dict[str, object] | None, Path]:
    label_path = source_path.parent / "label.json"
    if not label_path.exists():
        return None, label_path

    with label_path.open("r", encoding="utf-8") as file:
        label = json.load(file)
    if not isinstance(label, dict):
        raise ValueError(f"Grasp label JSON must contain an object: {label_path}")

    return label, label_path


def build_grasp_points(label: dict[str, object]) -> np.ndarray:
    center_values = label.get("gripper_contact_center", label.get("arrow_position"))
    if center_values is None:
        raise ValueError("Missing grasp center in label JSON.")
    center = np.asarray(center_values, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("Grasp center must be a 3D vector.")

    rotation_values = label.get("gripper_rotation_matrix")
    if rotation_values is not None:
        rotation = np.asarray(rotation_values, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("Gripper rotation matrix must be 3x3.")
    else:
        direction_values = label.get("gripper_direction", label.get("arrow_direction"))
        if direction_values is None:
            raise ValueError("Missing gripper rotation matrix or direction in label JSON.")
        direction = np.asarray(direction_values, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError("Gripper direction must be a 3D vector.")
        rotation = rotation_from_z(direction)

    width = float(label.get("gripper_width", 0.03))
    depth = float(label.get("gripper_depth", 0.08))
    finger_height = float(label.get("finger_height", 0.01))

    finger_len = depth * 0.55
    bridge_len = depth * 0.2
    handle_len = depth * 0.45

    left_tip = np.array([-width / 2.0, 0.0, 0.0], dtype=np.float64)
    right_tip = np.array([width / 2.0, 0.0, 0.0], dtype=np.float64)
    left_base = np.array([-width / 2.0, 0.0, -finger_len], dtype=np.float64)
    right_base = np.array([width / 2.0, 0.0, -finger_len], dtype=np.float64)
    bridge_left = np.array([-width / 2.0, 0.0, -finger_len - bridge_len], dtype=np.float64)
    bridge_right = np.array([width / 2.0, 0.0, -finger_len - bridge_len], dtype=np.float64)
    handle_end = np.array([0.0, 0.0, -finger_len - bridge_len - handle_len], dtype=np.float64)

    offset_y = max(finger_height * 0.4, 0.002)
    segments = [
        _sample_line(left_tip + [0.0, -offset_y, 0.0], left_base + [0.0, -offset_y, 0.0], 40),
        _sample_line(left_tip + [0.0, offset_y, 0.0], left_base + [0.0, offset_y, 0.0], 40),
        _sample_line(right_tip + [0.0, -offset_y, 0.0], right_base + [0.0, -offset_y, 0.0], 40),
        _sample_line(right_tip + [0.0, offset_y, 0.0], right_base + [0.0, offset_y, 0.0], 40),
        _sample_line(left_base, right_base, 28),
        _sample_line(bridge_left, bridge_right, 28),
        _sample_line((left_base + bridge_left) / 2.0, (right_base + bridge_right) / 2.0, 28),
        _sample_line((left_base + right_base) / 2.0, handle_end, 36),
    ]
    local_points = np.vstack(segments)
    return local_points @ rotation.T + center


def preprocess_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    downsampled = point_cloud.voxel_down_sample(voxel_size)
    if downsampled.is_empty():
        raise ValueError("Point cloud became empty after voxel downsampling. Reduce --voxel-size.")

    normal_radius = voxel_size * 2.0
    feature_radius = voxel_size * 5.0

    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return downsampled, fpfh


def execute_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    distance_threshold = voxel_size * 3.0
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000),
    )


def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    voxel_size: float,
    icp_distance_factor: float,
) -> RegistrationSummary:
    source_tensor = o3d.t.geometry.PointCloud.from_legacy(source)
    target_tensor = o3d.t.geometry.PointCloud.from_legacy(target)

    normal_radius = voxel_size * 2.0
    source_tensor.estimate_normals(max_nn=30, radius=normal_radius)
    target_tensor.estimate_normals(max_nn=30, radius=normal_radius)

    result = o3d.t.pipelines.registration.icp(
        source_tensor,
        target_tensor,
        max_correspondence_distance=voxel_size * icp_distance_factor,
        init_source_to_target=o3d.core.Tensor(initial_transformation, dtype=o3d.core.Dtype.Float64),
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=60,
        ),
    )
    return RegistrationSummary(
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        transformation=result.transformation.numpy(),
    )


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    window_name: str,
    source_grasp: np.ndarray | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib 未安装，无法显示配准结果。请先安装：pip install matplotlib")

    figure = _create_registration_figure(
        source=source,
        target=target,
        transformation=transformation,
        source_grasp=source_grasp,
        title=window_name,
    )
    plt.show()
    plt.close(figure)


def save_registration_screenshot(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    output_path: Path,
    source_grasp: np.ndarray | None = None,
) -> None:
    figure = _create_registration_figure(
        source=source,
        target=target,
        transformation=transformation,
        source_grasp=source_grasp,
        title="Aligned Registration",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Screenshot saved: {output_path}")


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


def _transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    return points @ rotation.T + translation


def _prepare_plot_bounds(point_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([points for points in point_sets if points.size > 0])
    min_bound = stacked.min(axis=0)
    max_bound = stacked.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    half_extent = np.max(max_bound - min_bound) / 2.0
    half_extent = max(half_extent * 1.1, 1e-3)
    return center - half_extent, center + half_extent


def _draw_projection_panel(
    axis: plt.Axes,
    title: str,
    points: np.ndarray,
    dims: tuple[int, int],
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    color: str,
    label: str,
    point_size: float,
) -> None:
    axis.set_title(title, fontsize=12)
    axis.set_xlim(min_bound[dims[0]], max_bound[dims[0]])
    axis.set_ylim(min_bound[dims[1]], max_bound[dims[1]])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)

    labels = ("X", "Y", "Z")
    axis.set_xlabel(labels[dims[0]])
    axis.set_ylabel(labels[dims[1]])

    if points.size > 0:
        axis.scatter(
            points[:, dims[0]],
            points[:, dims[1]],
            s=point_size,
            c=color,
            alpha=0.65,
            linewidths=0,
            label=label,
        )


def _create_registration_figure(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    source_grasp: np.ndarray | None,
    title: str,
    max_points_per_cloud: int = 25000,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("matplotlib 未安装，无法保存离线对齐预览图。请先安装：pip install matplotlib")

    source_points = _sample_points(
        _transform_points(np.asarray(source.points), transformation),
        max_points_per_cloud,
    )
    target_points = _sample_points(np.asarray(target.points), max_points_per_cloud)

    grasp_points = np.empty((0, 3), dtype=np.float64)
    if source_grasp is not None and source_grasp.size > 0:
        grasp_points = _sample_points(_transform_points(source_grasp, transformation), 12000)

    all_points = [pts for pts in (source_points, target_points, grasp_points) if pts.size > 0]
    if not all_points:
        raise ValueError("No points available for registration preview.")

    min_bound, max_bound = _prepare_plot_bounds(all_points)

    figure, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=180)
    panels = [
        (axes[0, 0], "XY Overlay", (0, 1)),
        (axes[0, 1], "XZ Overlay", (0, 2)),
        (axes[1, 0], "YZ Overlay", (1, 2)),
        (axes[1, 1], "XY Zoom Overlay", (0, 1)),
    ]
    point_size = 1.0 if max(source_points.shape[0], target_points.shape[0]) > 12000 else 2.5

    for axis, panel_title, dims in panels:
        _draw_projection_panel(axis, panel_title, target_points, dims, min_bound, max_bound, "#3a923a", "target", point_size)
        _draw_projection_panel(axis, panel_title, source_points, dims, min_bound, max_bound, "#3050d8", "aligned source", point_size)
        if grasp_points.size > 0:
            _draw_projection_panel(axis, panel_title, grasp_points, dims, min_bound, max_bound, "#e0911b", "grasp", point_size * 1.4)

    zoom_axis = axes[1, 1]
    zoom_center = (min_bound[[0, 1]] + max_bound[[0, 1]]) / 2.0
    zoom_half = (max_bound[[0, 1]] - min_bound[[0, 1]]) * 0.35
    zoom_axis.set_xlim(zoom_center[0] - zoom_half[0], zoom_center[0] + zoom_half[0])
    zoom_axis.set_ylim(zoom_center[1] - zoom_half[1], zoom_center[1] + zoom_half[1])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(3, max(1, len(unique))), frameon=False)
    figure.suptitle(title, fontsize=18)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return figure


def save_registration_offline_preview(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    output_path: Path,
    source_grasp: np.ndarray | None = None,
    max_points_per_cloud: int = 25000,
) -> None:
    figure = _create_registration_figure(
        source=source,
        target=target,
        transformation=transformation,
        source_grasp=source_grasp,
        title="ICP Alignment Preview",
        max_points_per_cloud=max_points_per_cloud,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Offline preview saved: {output_path}")


def print_result(name: str, result: RegistrationSummary | o3d.pipelines.registration.RegistrationResult) -> None:
    print(f"{name} fitness: {result.fitness:.6f}")
    print(f"{name} inlier_rmse: {result.inlier_rmse:.6f}")
    print(f"{name} transformation:\n{result.transformation}")


def transform_grasp_label(
    label: dict[str, object],
    transformation: np.ndarray,
) -> dict[str, object]:
    """Apply a 4x4 transformation to grasp label fields and return updated values."""
    R = transformation[:3, :3]
    t = transformation[:3, 3]

    center = np.asarray(label["gripper_contact_center"], dtype=np.float64)
    new_center = R @ center + t

    rot_mat = np.asarray(label["gripper_rotation_matrix"], dtype=np.float64)
    new_rot_mat = R @ rot_mat

    direction = np.asarray(label["gripper_direction"], dtype=np.float64)
    new_direction = R @ direction

    return {
        "gripper_contact_center": new_center.tolist(),
        "gripper_direction": new_direction.tolist(),
        "gripper_rotation_matrix": new_rot_mat.tolist(),
        "gripper_width": float(label["gripper_width"]),
    }


def register_point_clouds(
    source: Path,
    target: Path,
    voxel_size: float = 0.01,
    icp_distance_factor: float = 2.0,
    no_vis: bool = True,
) -> dict[str, object] | None:
    """Register source PLY to target PLY using RANSAC + ICP.

    Saves aligned matplotlib preview images next to the target file.
    Returns transformed grasp label fields in the target coordinate frame,
    or None if no label.json is found next to source.

    Args:
        source: Path to the source PLY point cloud.
        target: Path to the target PLY point cloud.
        voxel_size: Voxel size for downsampling.
        icp_distance_factor: ICP correspondence distance multiplier.
        no_vis: If True, skip interactive visualization windows.

    Returns:
        Dict with keys gripper_contact_center, gripper_direction,
        gripper_rotation_matrix, gripper_width in the target coordinate frame,
        or None if label.json was not found.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    if icp_distance_factor <= 0:
        raise ValueError("icp_distance_factor must be positive.")

    source_pc = load_point_cloud(source)
    target_pc = load_point_cloud(target)

    source_grasp = None
    grasp_label, grasp_label_path = load_grasp_label(source)
    if grasp_label is not None:
        try:
            source_grasp = build_grasp_points(grasp_label)
            print(f"Loaded grasp label: {grasp_label_path}")
        except (TypeError, ValueError) as exc:
            print(f"Warning: failed to visualize grasp label {grasp_label_path}: {exc}")
    else:
        print(f"No grasp label found next to source PLY: {grasp_label_path}")

    print(f"Loaded source points: {len(source_pc.points)}")
    print(f"Loaded target points: {len(target_pc.points)}")
    print(f"Using voxel size: {voxel_size:.4f}")

    if not no_vis:
        draw_registration_result(
            source_pc,
            target_pc,
            np.eye(4),
            window_name="Before Registration",
            source_grasp=source_grasp,
        )

    source_down, source_fpfh = preprocess_point_cloud(source_pc, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pc, voxel_size)

    ransac_result = execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size,
    )
    print_result("RANSAC", ransac_result)

    icp_result = refine_registration(
        source_pc,
        target_pc,
        ransac_result.transformation,
        voxel_size,
        icp_distance_factor,
    )
    print_result("ICP", icp_result)

    offline_preview_path = target.parent / "aligned_registration_offline.png"
    save_registration_offline_preview(
        source_pc,
        target_pc,
        icp_result.transformation,
        offline_preview_path,
        source_grasp=source_grasp,
    )

    if not no_vis:
        screenshot_path = target.parent / "aligned_registration.png"
        save_registration_screenshot(
            source_pc,
            target_pc,
            icp_result.transformation,
            screenshot_path,
            source_grasp=source_grasp,
        )
        draw_registration_result(
            source_pc,
            target_pc,
            icp_result.transformation,
            window_name="After Registration (Matplotlib)",
            source_grasp=source_grasp,
        )

    if grasp_label is None:
        return None

    return transform_grasp_label(grasp_label, icp_result.transformation)


def main() -> None:
    args = parse_args()
    result = register_point_clouds(
        source=args.source,
        target=args.target,
        voxel_size=args.voxel_size,
        icp_distance_factor=args.icp_distance_factor,
        no_vis=args.no_vis,
    )
    if result is not None:
        print("\nTransformed grasp label in target coordinate frame:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
