from __future__ import annotations
import cv2
import numpy as np

#Thresholding techniques to convert grayscale images into binary images.
def otsu(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive(gray: np.ndarray, block_size: int = 31, constant: float = 11) -> np.ndarray:
    if min(gray.shape) < 3:
        return otsu(gray)
    maximum = min(gray.shape) if min(gray.shape) % 2 else min(gray.shape) - 1
    block = min(block_size, maximum)
    block = max(3, block if block % 2 else block - 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        constant,
    )
