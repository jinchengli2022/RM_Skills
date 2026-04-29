from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labeling import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_WARMUP_FRAMES,
    WINDOW_NAME,
    CameraIntrinsics,
    camera_role_name,
    draw_prompt_overlay,
    format_camera_serial,
    require_cv2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one RGB-D frame for execute-skill target selection.")
    parser.add_argument("--robot-ip", default="169.254.128.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--skill-name", default="goto_affordance")
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--source-ply", type=Path, default=None)
    parser.add_argument("--handeye-result-path", type=Path, default=Path("handeye_output") / "handeye_result.json")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--session-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_camera_device(camera_serial: str | None, camera_index: int) -> dict[str, str | int | None]:
    from src.align.reconstruct import list_realsense_devices

    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("未发现任何 RealSense 相机")
    if camera_serial:
        for device in devices:
            if device.get("serial") == camera_serial:
                return device
        available = ", ".join(str(d.get("serial", "unknown")) for d in devices)
        raise RuntimeError(f"找不到序列号为 {camera_serial} 的相机，当前可用: {available}")
    if camera_index < 0 or camera_index >= len(devices):
        raise RuntimeError(f"camera-index={camera_index} 超出范围，当前共有 {len(devices)} 台相机")
    return devices[camera_index]


def resolve_source_dir(source_dir: Path | None, source_ply: Path | None) -> Path:
    candidate = source_dir if source_dir is not None else source_ply
    if candidate is None:
        raise RuntimeError("必须提供 --source-dir；旧参数 --source-ply 仅作为兼容别名保留")
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate.parent
    return candidate


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2，无法列出相机")
        from src.align.reconstruct import list_realsense_devices

        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
            return
        for device in devices:
            serial = device.get("serial")
            role = camera_role_name(str(serial) if serial is not None else None)
            role_text = f" role={role}" if role is not None else ""
            print(
                "index={index} name={name} serial={serial} ip={ip} firmware={firmware} connection={connection_type}{role}".format(
                    role=role_text,
                    **device,
                )
            )
        return

    require_cv2()
    if rs is None:
        raise RuntimeError("未安装 pyrealsense2，无法读取 RealSense 相机")
    if args.session_dir is None:
        raise RuntimeError("--session-dir 是必需的")

    session_dir = args.session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    selected_camera = resolve_camera_device(args.camera_serial, args.camera_index)
    selected_camera_serial = selected_camera.get("serial")
    if not selected_camera_serial:
        raise RuntimeError(f"选中的相机缺少序列号，无法继续: {selected_camera}")
    camera_serial = str(selected_camera_serial)
    print(f"selected camera: {format_camera_serial(camera_serial)}")

    resolved_source_dir = resolve_source_dir(args.source_dir, args.source_ply)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camera_serial)
    config.enable_stream(rs.stream.depth, DEFAULT_WIDTH, DEFAULT_HEIGHT, rs.format.z16, DEFAULT_FPS)
    config.enable_stream(rs.stream.color, DEFAULT_WIDTH, DEFAULT_HEIGHT, rs.format.bgr8, DEFAULT_FPS)
    align = rs.align(rs.stream.color)

    state: dict[str, object] = {"prompt_pixel": None, "should_capture": False}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["prompt_pixel"] = (x, y)
            state["should_capture"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    started = False
    latest_snapshot: tuple[np.ndarray, np.ndarray, CameraIntrinsics, float] | None = None
    try:
        profile = pipeline.start(config)
        started = True
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        for _ in range(DEFAULT_WARMUP_FRAMES):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            rs_intr = depth_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = CameraIntrinsics(
                fx=float(rs_intr.fx),
                fy=float(rs_intr.fy),
                cx=float(rs_intr.ppx),
                cy=float(rs_intr.ppy),
                width=int(rs_intr.width),
                height=int(rs_intr.height),
            )
            latest_snapshot = (color_bgr, depth_image, intrinsics, depth_scale)
            prompt_pixel = state["prompt_pixel"]
            display = draw_prompt_overlay(color_bgr, prompt_pixel, selected=prompt_pixel is not None)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise RuntimeError("用户取消了执行流程")
            if key == ord("r"):
                state["prompt_pixel"] = None
                state["should_capture"] = False
            if state["should_capture"]:
                if prompt_pixel is None or latest_snapshot is None:
                    state["should_capture"] = False
                    continue
                color_bgr, depth_image, intrinsics, depth_scale = latest_snapshot
                color_path = session_dir / "capture_color.png"
                depth_path = session_dir / "capture_depth.npy"
                if not cv2.imwrite(str(color_path), color_bgr):
                    raise RuntimeError(f"保存临时彩色图失败: {color_path}")
                np.save(depth_path, depth_image)
                manifest = {
                    "robot_ip": args.robot_ip,
                    "robot_port": int(args.robot_port),
                    "skill_name": str(args.skill_name),
                    "camera_serial": camera_serial,
                    "camera_index": int(args.camera_index),
                    "source_dir": str(resolved_source_dir),
                    "handeye_result_path": str(args.handeye_result_path.resolve()),
                    "vis": bool(args.vis),
                    "selected_camera": selected_camera,
                    "depth_scale": depth_scale,
                    "prompt_pixel": [int(prompt_pixel[0]), int(prompt_pixel[1])],
                    "intrinsics": {
                        "fx": intrinsics.fx,
                        "fy": intrinsics.fy,
                        "cx": intrinsics.cx,
                        "cy": intrinsics.cy,
                        "width": intrinsics.width,
                        "height": intrinsics.height,
                    },
                }
                (session_dir / "session.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
