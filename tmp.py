import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs

save_dir = "realsense_test_output"
os.makedirs(save_dir, exist_ok=True)

# 1. 枚举设备
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    raise RuntimeError("未检测到 RealSense 设备，请检查 USB 连接和供电。")

print("检测到的 RealSense 设备：")
for i, dev in enumerate(devices):
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"[{i}] name={name}, serial={serial}")

# 2. 配置流
pipeline = rs.pipeline()
config = rs.config()

# 如有多台 D435，可指定序列号
serial = devices[0].get_info(rs.camera_info.serial_number)
config.enable_device(serial)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# 对齐：把 depth 对齐到 color
align = rs.align(rs.stream.color)

# 深度尺度
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale:", depth_scale)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())     # uint16
        color_image = np.asanyarray(color_frame.get_data())     # BGR

        # 可视化深度图
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 横向拼接显示
        show_img = np.hstack((color_image, depth_vis))
        cv2.imshow("RGB | Depth", show_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("received")
            ts = int(time.time())
            color_path = os.path.join(save_dir, f"color_{ts}.png")
            depth_png_path = os.path.join(save_dir, f"depth_vis_{ts}.png")
            depth_raw_path = os.path.join(save_dir, f"depth_raw_{ts}.npy")

            cv2.imwrite(color_path, color_image)
            cv2.imwrite(depth_png_path, depth_vis)
            np.save(depth_raw_path, depth_image)

            # 示例：取中心点深度（米）
            h, w = depth_image.shape
            center_depth = depth_image[h // 2, w // 2] * depth_scale
            print(f"已保存: {color_path}, {depth_png_path}, {depth_raw_path}")
            print(f"中心点深度: {center_depth:.3f} m")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()