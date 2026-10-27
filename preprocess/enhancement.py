from __future__ import annotations
import cv2
import numpy as np

#Stretch the grayscale range.
def contrast_stretch(gray: np.ndarray, low_percentile: float = 2, high_percentile: float = 98) -> np.ndarray:
    low, high = np.percentile(gray, [low_percentile, high_percentile])
    return np.clip((gray.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)


#Equalize the whole image.
def equalize(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


#Equalize small image regions.
def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)


#Sharpen using a blurred copy.
def sharpen(gray: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    result = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return result, blurred
