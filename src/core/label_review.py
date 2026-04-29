from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labeling import WINDOW_NAME, draw_review_overlay, require_cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review SAM mask and confirm or reset.")
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cv2()
    session_dir = args.session_dir.resolve()
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    color_path = session_dir / "capture_color.png"
    mask_path = session_dir / "session_mask.png"

    image_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"读取临时彩色图失败: {color_path}")
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise RuntimeError(f"读取临时 SAM mask 失败: {mask_path}")

    prompt = manifest.get("prompt_pixel")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"session.json 缺少 prompt_pixel: {session_dir / 'session.json'}")

    display = draw_review_overlay(
        image_bgr=image_bgr,
        mask=mask_image > 0,
        prompt_pixel=(int(prompt[0]), int(prompt[1])),
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, max(1280, display.shape[1]), max(720, display.shape[0]))
    print("SAM 复核窗口已打开：当前只显示叠加结果。按 c 确认，r 重选，q 退出。")
    try:
        while True:
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                return
            if key == ord("r"):
                raise SystemExit(10)
            if key == ord("q"):
                raise SystemExit(11)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
