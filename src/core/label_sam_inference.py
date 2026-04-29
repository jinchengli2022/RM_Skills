from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src" / "segment-anything"))

from segment_anything import SamPredictor, sam_model_registry
from labeling import SAM_MODEL_TYPE, WINDOW_NAME, draw_review_overlay, require_cv2

SAM_CHECKPOINT_PATH = REPO_ROOT / "src" / "segment-anything" / "checkpoing" / "sam_vit_h_4b8939.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM inference from a saved label session.")
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cv2()
    session_dir = args.session_dir.resolve()
    manifest_path = session_dir / "session.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"session.json 不存在: {manifest_path}")
    if not SAM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {SAM_CHECKPOINT_PATH}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_path = session_dir / "capture_color.png"
    if not image_path.exists():
        raise FileNotFoundError(f"临时彩色图不存在: {image_path}")

    prompt = manifest.get("prompt_pixel")
    if not isinstance(prompt, list) or len(prompt) != 2:
        raise ValueError(f"session.json 缺少 prompt_pixel: {manifest_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"读取图像失败: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    point_coords = np.asarray([[float(prompt[0]), float(prompt[1])]], dtype=np.float32)
    point_labels = np.asarray([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8) * 255

    mask_path = session_dir / "session_mask.png"
    if not cv2.imwrite(str(mask_path), best_mask):
        raise RuntimeError(f"保存 SAM mask 失败: {mask_path}")

    display = draw_review_overlay(
        image_bgr=image_bgr,
        mask=best_mask > 0,
        prompt_pixel=(int(prompt[0]), int(prompt[1])),
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, max(1280, display.shape[1]), max(720, display.shape[0]))
    print("SAM 结果已显示叠加图。按 c 确认，r 重选，q 退出。")
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
