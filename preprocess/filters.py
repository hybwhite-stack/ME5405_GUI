from __future__ import annotations

import cv2
import numpy as np


def median(gray: np.ndarray, kernel: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, kernel)


def gaussian(gray: np.ndarray, kernel: int = 3, sigma: float = 0.8) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel, kernel), sigma)


def bilateral(gray: np.ndarray, diameter: int = 5, sigma_color: float = 40, sigma_space: float = 40) -> np.ndarray:
    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)
