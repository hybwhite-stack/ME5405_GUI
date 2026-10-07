from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_image(path: Path, image: np.ndarray) -> None:
    """Write through encoded bytes so Windows paths may contain Chinese text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def crop_roi(image: np.ndarray, roi: list[int]) -> tuple[np.ndarray, dict]:
    if not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError("roi 必须是四个整数 [x,y,width,height]")
    x, y, width, height = roi
    source_height, source_width = image.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > source_width or y + height > source_height:
        raise ValueError("roi 超出原图范围或尺寸无效")
    return image[y:y + height, x:x + width].copy(), {
        "roi_original_xywh": [x, y, width, height],
        "original_to_processed": [[1, 0, -x], [0, 1, -y], [0, 0, 1]],
    }


def roi_from_detections(image: np.ndarray, detections: list[dict], margin: int = 20) -> list[int] | None:
    if not detections:
        return None
    height, width = image.shape[:2]
    x = max(0, min(item["x_min"] for item in detections) - margin)
    y = max(0, min(item["y_min"] for item in detections) - margin)
    right = min(width, max(item["x_max"] for item in detections) + margin)
    bottom = min(height, max(item["y_max"] for item in detections) + margin)
    return [int(x), int(y), int(right - x), int(bottom - y)]


def image_diagnostics(image: np.ndarray) -> dict:
    gray = to_gray(image)
    scale = min(1.0, 800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    smooth = cv2.GaussianBlur(small, (0, 0), 15)
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "mean_brightness": float(gray.mean()),
        "gray_std": float(gray.std()),
        "p98_minus_p02": float(np.percentile(gray, 98) - np.percentile(gray, 2)),
        "laplacian_variance_at_max800": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        "illumination_std_at_max800": float(smooth.std()),
        "note": "Indicators only: texture, resolution and background affect these measurements.",
    }
