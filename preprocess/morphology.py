from __future__ import annotations

import cv2
import numpy as np


def apply_morphology(binary: np.ndarray, operation: str, kernel_size: tuple[int, int] = (2, 2), iterations: int = 1) -> np.ndarray:
    """Apply morphology to minority-colour foreground, then restore polarity."""
    invert = np.count_nonzero(binary) > binary.size / 2
    foreground = 255 - binary if invert else binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    code = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
    result = cv2.morphologyEx(foreground, code, kernel, iterations=iterations)
    return 255 - result if invert else result


def opening(binary: np.ndarray, kernel_size: tuple[int, int] = (2, 2), iterations: int = 1) -> np.ndarray:
    return apply_morphology(binary, "open", kernel_size, iterations)


def closing(binary: np.ndarray, kernel_size: tuple[int, int] = (2, 2), iterations: int = 1) -> np.ndarray:
    return apply_morphology(binary, "close", kernel_size, iterations)
