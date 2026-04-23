#!/usr/bin/env python
"""
手眼标定脚本（基于 RealSense + RM 机械臂）

功能:
1. 参考 tmp.py 的方式打开 RealSense 彩色/深度流并实时预览
2. 检测棋盘格角点，按 s 采集一帧样本（图像 + 机器人位姿）
3. 使用 OpenCV calibrateHandEye 计算手眼外参
4. 保存标定结果到 JSON 文件

按键:
- s: 采集当前样本（要求检测到棋盘格）
- c: 计算标定结果（样本数达到阈值后）
- q: 退出
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.demo_project import RobotArmController  # noqa: E402


@dataclass
class Sample:
    """一次手眼标定采样。"""
    timestamp: float
    pose_base_to_gripper: List[float]  # [x,y,z,rx,ry,rz], m/rad
    R_gripper2base: np.ndarray         # 3x3
    t_gripper2base: np.ndarray         # 3x1
    R_target2cam: np.ndarray           # 3x3
    t_target2cam: np.ndarray           # 3x1


def euler_xyz_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """将 [rx, ry, rz] (rad) 转换为旋转矩阵。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=np.float64)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float64)

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # 与项目中可视化逻辑保持一致: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def invert_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """求逆变换。"""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def make_chessboard_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """构建棋盘格三维点，Z=0，单位米。"""
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_m
    return objp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RealSense + RM 手眼标定')

    parser.add_argument('--robot-ip', type=str, default='169.254.128.19', help='机械臂 IP')
    parser.add_argument('--robot-port', type=int, default=8080, help='机械臂端口')
    parser.add_argument('--robot-level', type=int, default=3, help='机械臂连接 level')
    parser.add_argument('--robot-mode', type=int, default=2, help='机械臂线程模式 0/1/2')

    parser.add_argument('--width', type=int, default=640, help='彩色流宽度')
    parser.add_argument('--height', type=int, default=480, help='彩色流高度')
    parser.add_argument('--fps', type=int, default=30, help='帧率')

    parser.add_argument('--board-cols', type=int, default=8, help='棋盘格内角点列数')
    parser.add_argument('--board-rows', type=int, default=6, help='棋盘格内角点行数')
    parser.add_argument('--square-size-mm', type=float, default=25.0, help='棋盘格方格边长(mm)')

    parser.add_argument('--min-samples', type=int, default=10, help='最小样本数')
    parser.add_argument('--eye-in-hand', action='store_true', default=True,
                        help='眼在手上模式（默认开启）')
    parser.add_argument('--eye-to-hand', action='store_true',
                        help='眼在手外固定模式（若设置则覆盖 eye-in-hand）')

    parser.add_argument('--output-dir', type=str, default='handeye_output', help='输出目录')
    parser.add_argument('--save-frames', action='store_true', help='是否保存采样彩色图')
    return parser


def get_robot_pose(robot: RobotArmController) -> Optional[List[float]]:
    """读取末端位姿 [x,y,z,rx,ry,rz]。"""
    ret, state = robot.robot.rm_get_current_arm_state()
    if ret != 0:
        print(f'[错误] 获取机械臂状态失败, 错误码={ret}')
        return None
    pose = state.get('pose')
    if pose is None or len(pose) != 6:
        print(f'[错误] 机械臂 pose 数据异常: {pose}')
        return None
    return [float(x) for x in pose]


def pose_to_gripper2base(pose: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """将 [x,y,z,rx,ry,rz] 转为 gripper->base 变换。"""
    x, y, z, rx, ry, rz = pose
    R = euler_xyz_to_rot(rx, ry, rz)
    t = np.array([[x], [y], [z]], dtype=np.float64)
    return R, t


def solve_target2cam(
    corners: np.ndarray,
    obj_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """PnP 求解 target->camera 变换。"""
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_points,
        imagePoints=corners,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def calibrate_handeye(samples: List[Sample], eye_in_hand: bool) -> Tuple[np.ndarray, np.ndarray]:
    """执行手眼标定并返回旋转和平移。"""
    R_gripper2base = [s.R_gripper2base for s in samples]
    t_gripper2base = [s.t_gripper2base for s in samples]
    R_target2cam = [s.R_target2cam for s in samples]
    t_target2cam = [s.t_target2cam for s in samples]

    if eye_in_hand:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        return R_cam2gripper, t_cam2gripper

    # 眼在手外: 将 gripper2base 取逆得到 base2gripper，再喂给 calibrateHandEye
    R_base2gripper = []
    t_base2gripper = []
    for R_g2b, t_g2b in zip(R_gripper2base, t_gripper2base):
        R_b2g, t_b2g = invert_transform(R_g2b, t_g2b)
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_base2gripper,
        t_gripper2base=t_base2gripper,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    return R_cam2base, t_cam2base


def main() -> None:
    args = build_parser().parse_args()
    eye_in_hand = not args.eye_to_hand

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('手眼标定启动')
    print(f'模式: {"眼在手上" if eye_in_hand else "眼在手外"}')
    print(f'输出目录: {args.output_dir}')
    print('=' * 70)

    # 连接机械臂
    robot = RobotArmController(args.robot_ip, args.robot_port, args.robot_level, args.robot_mode)

    # 初始化 RealSense（参考 tmp.py）
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        robot.disconnect()
        raise RuntimeError('未检测到 RealSense 设备，请检查 USB 连接和供电。')

    print('检测到的 RealSense 设备:')
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f'[{i}] name={name}, serial={serial}')

    serial = devices[0].get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    print('相机内参 K:')
    print(K)
    print(f'畸变参数: {dist.tolist()}')

    board_size = (args.board_cols, args.board_rows)
    square_size_m = args.square_size_mm / 1000.0
    obj_points = make_chessboard_points(args.board_cols, args.board_rows, square_size_m)

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    samples: List[Sample] = []

    print('\n按键说明: s=采样, c=计算标定, q=退出\n')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size, None)

            vis = color_image.copy()
            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
                cv2.drawChessboardCorners(vis, board_size, corners_refined, found)
            else:
                corners_refined = None

            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            show = np.hstack((vis, depth_vis))

            status = f'samples={len(samples)} | found={found} | min={args.min_samples}'
            cv2.putText(show, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('HandEye Calibration: RGB|Depth', show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if not found or corners_refined is None:
                    print('[提示] 当前帧未检测到棋盘格，采样失败')
                    continue

                pose = get_robot_pose(robot)
                if pose is None:
                    print('[提示] 未获取到有效机械臂位姿，采样失败')
                    continue

                pnp = solve_target2cam(corners_refined, obj_points, K, dist)
                if pnp is None:
                    print('[提示] solvePnP 失败，采样失败')
                    continue

                R_t2c, t_t2c = pnp
                R_g2b, t_g2b = pose_to_gripper2base(pose)

                ts = time.time()
                sample = Sample(
                    timestamp=ts,
                    pose_base_to_gripper=pose,
                    R_gripper2base=R_g2b,
                    t_gripper2base=t_g2b,
                    R_target2cam=R_t2c,
                    t_target2cam=t_t2c,
                )
                samples.append(sample)

                print(f'[采样] 成功，第 {len(samples)} 组')
                print(f'       pose=[{", ".join(f"{x:.4f}" for x in pose)}]')

                if args.save_frames:
                    img_path = os.path.join(args.output_dir, f'sample_{len(samples):02d}_{int(ts)}.png')
                    cv2.imwrite(img_path, vis)
                    print(f'       已保存图像: {img_path}')

            elif key == ord('c'):
                if len(samples) < args.min_samples:
                    print(f'[提示] 样本不足: {len(samples)}/{args.min_samples}')
                    continue

                print('[计算] 开始手眼标定...')
                R_out, t_out = calibrate_handeye(samples, eye_in_hand=eye_in_hand)

                out = {
                    'mode': 'eye_in_hand' if eye_in_hand else 'eye_to_hand',
                    'num_samples': len(samples),
                    'camera_matrix': K.tolist(),
                    'dist_coeffs': dist.tolist(),
                    'board': {
                        'cols': args.board_cols,
                        'rows': args.board_rows,
                        'square_size_mm': args.square_size_mm,
                    },
                    'result': {
                        'rotation_matrix': R_out.tolist(),
                        'translation_m': t_out.reshape(3).tolist(),
                    },
                    'samples': [
                        {
                            'timestamp': s.timestamp,
                            'pose_base_to_gripper': s.pose_base_to_gripper,
                        }
                        for s in samples
                    ],
                }

                out_path = os.path.join(args.output_dir, 'handeye_result.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

                print('[计算] 标定完成')
                print('结果旋转矩阵:')
                print(R_out)
                print(f'结果平移(m): {t_out.reshape(3)}')
                print(f'已保存: {out_path}')

            elif key == ord('q'):
                print('[退出] 用户结束程序')
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.disconnect()


if __name__ == '__main__':
    main()
