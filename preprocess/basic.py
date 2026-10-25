from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))

# Convert a color image to grayscale.
def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

# Crop the selected region.
def crop_roi(image: np.ndarray, roi: list[int]) -> tuple[np.ndarray, dict]:
    if not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError("roi must be four integers [x,y,width,height]")
    x, y, width, height = roi
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
