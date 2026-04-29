from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    CONSISTENCY_TOL,
    CameraIntrinsics,
    build_object_point_cloud,
    build_saved_overlay_image,
    build_selection_from_arrays,
    check_rotation_matrix,
    get_current_pose,
    load_base_to_camera_transform,
    make_ee_to_gripper_transform,
    matrix_from_rm_pose,
    print_matrix,
    require_cv2,
    require_open3d,
    require_pyvista,
    save_label,
    save_metadata,
    show_pyvista_preview,
    wait_until_confirm_with_gripper_toggle,
)

SAM_CHECKPOINT_PATH = REPO_ROOT / "src" / "segment-anything" / "checkpoing" / "sam_vit_h_4b8939.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize label output from a completed session.")
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


def save_capture_artifacts(
    output_root: Path,
    selection,
    point_cloud,
    device: dict[str, str | int | None],
) -> tuple[Path, Path, Path, Path]:
    require_cv2()
    require_open3d()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    color_path = out_dir / "color.png"
    depth_path = out_dir / "depth.png"
    mask_path = out_dir / "mask.png"
    overlay_path = out_dir / "sam_overlay.png"
    point_cloud_path = out_dir / "point_cloud.ply"

    if not cv2.imwrite(str(color_path), selection.color_bgr):
        raise RuntimeError(f"保存彩色图失败: {color_path}")
    if not cv2.imwrite(str(depth_path), selection.depth_image):
        raise RuntimeError(f"保存深度图失败: {depth_path}")
    if not cv2.imwrite(str(mask_path), selection.mask.astype(np.uint8) * 255):
        raise RuntimeError(f"保存 SAM mask 失败: {mask_path}")
    if not cv2.imwrite(str(overlay_path), build_saved_overlay_image(selection)):
        raise RuntimeError(f"保存 SAM 叠加图失败: {overlay_path}")

    import open3d as o3d

    if not o3d.io.write_point_cloud(str(point_cloud_path), point_cloud, write_ascii=False):
        raise RuntimeError(f"保存目标点云失败: {point_cloud_path}")
    if not point_cloud_path.exists():
        raise RuntimeError(f"目标点云未成功落盘: {point_cloud_path}")

    save_metadata(out_dir, selection.intrinsics, selection.depth_scale, device)
    return out_dir, point_cloud_path, mask_path, overlay_path


def main() -> None:
    args = parse_args()
    require_cv2()
    require_open3d()
    require_pyvista()

    session_dir = args.session_dir.resolve()
    manifest_path = session_dir / "session.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"session.json 不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

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

    robot_controller = RobotArmController(str(manifest["robot_ip"]), int(manifest["robot_port"]), level=3)
    try:
        base_to_camera_transform = load_base_to_camera_transform(Path(manifest["handeye_result_path"]))
        ee_to_gripper_transform = make_ee_to_gripper_transform()
        capture_pose_rm = get_current_pose(robot_controller)
        base_to_ee_at_capture = matrix_from_rm_pose(robot_controller, capture_pose_rm)
        base_to_camera_at_capture = base_to_camera_transform.copy()

        output_dir, point_cloud_path, mask_path, overlay_path = save_capture_artifacts(
            output_root=Path(manifest["output_dir"]),
            selection=selection,
            point_cloud=point_cloud,
            device=manifest["selected_camera"],
        )

        wait_until_confirm_with_gripper_toggle(robot_controller)

        grasp_pose_rm = get_current_pose(robot_controller)
        base_to_ee_at_grasp = matrix_from_rm_pose(robot_controller, grasp_pose_rm)
        base_to_gripper_at_grasp = base_to_ee_at_grasp @ ee_to_gripper_transform

        camera_capture_to_ee_grasp = np.linalg.inv(base_to_camera_at_capture) @ base_to_ee_at_grasp
        camera_capture_to_gripper_grasp = np.linalg.inv(base_to_camera_at_capture) @ base_to_gripper_at_grasp
        camera_capture_to_gripper_grasp_equivalent = camera_capture_to_ee_grasp @ ee_to_gripper_transform

        chain_error = np.linalg.norm(camera_capture_to_ee_grasp - (np.linalg.inv(base_to_camera_at_capture) @ base_to_ee_at_grasp))
        gripper_chain_error = np.linalg.norm(camera_capture_to_gripper_grasp - camera_capture_to_gripper_grasp_equivalent)
        if chain_error > CONSISTENCY_TOL:
            raise RuntimeError(f"坐标链路一致性检查失败，error={chain_error:.3e} > tol={CONSISTENCY_TOL:.3e}")
        if gripper_chain_error > CONSISTENCY_TOL:
            raise RuntimeError(f"夹爪 TCP 坐标链路一致性检查失败，error={gripper_chain_error:.3e} > tol={CONSISTENCY_TOL:.3e}")

        gripper_rotation_in_capture_camera = camera_capture_to_gripper_grasp[:3, :3]
        gripper_center_in_capture_camera = camera_capture_to_gripper_grasp[:3, 3]
        check_rotation_matrix(gripper_rotation_in_capture_camera, "camera_capture_to_gripper_grasp")
        check_rotation_matrix(base_to_camera_at_capture[:3, :3], "base_to_camera_at_capture")

        label_path = output_dir / "label.json"
        save_label(
            label_path=label_path,
            point_cloud_path=point_cloud_path,
            mask_path=mask_path,
            overlay_path=overlay_path,
            gripper_center_in_capture_camera=gripper_center_in_capture_camera,
            gripper_rotation_in_capture_camera=gripper_rotation_in_capture_camera,
            gripper_width=float(manifest["gripper_width"]),
            ee_to_gripper_transform=ee_to_gripper_transform,
            base_to_camera_at_capture=base_to_camera_at_capture,
            base_to_ee_at_capture=base_to_ee_at_capture,
            base_to_ee_at_grasp=base_to_ee_at_grasp,
            base_to_gripper_at_grasp=base_to_gripper_at_grasp,
            base_to_camera_transform=base_to_camera_transform,
            camera_capture_to_ee_grasp=camera_capture_to_ee_grasp,
            camera_capture_to_gripper_grasp=camera_capture_to_gripper_grasp,
            robot_controller=robot_controller,
            handeye_result_path=Path(manifest["handeye_result_path"]),
            selection=selection,
            sam_checkpoint_path=str(SAM_CHECKPOINT_PATH.resolve()),
        )

        show_pyvista_preview(
            point_cloud=point_cloud,
            gripper_center_in_capture_camera=gripper_center_in_capture_camera,
            gripper_rotation_in_capture_camera=gripper_rotation_in_capture_camera,
            gripper_width=float(manifest["gripper_width"]),
            camera_capture_to_ee_grasp=camera_capture_to_ee_grasp,
        )

        print(f"目标 mask 像素数: {np.count_nonzero(selection.mask)}")
        print(f"目标点云点数: {len(point_cloud.points)}")
        print(f"坐标链路一致性误差: {chain_error:.3e}")
        print(f"夹爪 TCP 坐标链路一致性误差: {gripper_chain_error:.3e}")
        print_matrix("base_to_camera_at_capture", base_to_camera_at_capture)
        print_matrix("camera_capture_to_ee_grasp", camera_capture_to_ee_grasp)
        print_matrix("base_to_gripper_at_grasp", base_to_gripper_at_grasp)
        print_matrix("camera_capture_to_gripper_grasp", camera_capture_to_gripper_grasp)
        print(f"mask: {mask_path}")
        print(f"sam_overlay: {overlay_path}")
        print(f"point_cloud: {point_cloud_path}")
        print(f"label: {label_path}")
    finally:
        robot_controller.disconnect()


if __name__ == "__main__":
    main()
