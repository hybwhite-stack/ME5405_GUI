from __future__ import annotations

import math

import cv2
import numpy as np

from .basic import to_gray


def validate_quad(points, width: int, height: int) -> np.ndarray:
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("corners 必须是四个 [x,y] 点")
    if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= width) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= height):
        raise ValueError("corners 超出原图范围")
    if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < 100:
        raise ValueError("四角必须按左上、右上、右下、左下顺序构成凸四边形")
    return quad


def perspective(image: np.ndarray, points) -> tuple[np.ndarray, dict]:
    quad = validate_quad(points, image.shape[1], image.shape[0])
    top_left, top_right, bottom_right, bottom_left = quad
    width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    if width < 10 or height < 10:
        raise ValueError("透视目标尺寸过小")
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad, destination)
    border = (255, 255, 255) if image.ndim == 3 else 255
    result = cv2.warpPerspective(image, matrix, (width, height), borderValue=border)
    return result, {"corners_original": quad.tolist(), "original_to_processed": matrix.tolist()}


def detect_document_quad(image: np.ndarray):
    scale = min(1.0, 1200 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    edges = cv2.Canny(cv2.GaussianBlur(to_gray(small), (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour) / (small.shape[0] * small.shape[1])
        if not 0.25 <= area <= 0.95:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        points = approx.reshape(4, 2).astype(np.float32) / scale
        center = points.mean(axis=0)
        points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
        points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
        try:
            return validate_quad(points, image.shape[1], image.shape[0])
        except ValueError:
            continue
    return None


def rotate_expanded(image: np.ndarray, angle: float) -> tuple[np.ndarray, dict]:
    if not math.isfinite(angle) or abs(angle) > 180:
        raise ValueError("angle 必须在 -180 到 180 之间")
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    sine, cosine = abs(matrix[0, 1]), abs(matrix[0, 0])
    new_width = math.ceil(height * sine + width * cosine)
    new_height = math.ceil(height * cosine + width * sine)
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    border = (255, 255, 255) if image.ndim == 3 else 255
    result = cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=border)
    return result, {
        "rotation_ccw_degrees": angle,
        "original_to_processed": np.vstack([matrix, [0, 0, 1]]).tolist(),
    }


def consistent_text_angle(detections: list[dict]) -> float | None:
    angles = []
    for item in detections:
        points = item["polygon"]
        dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
        if dx > 30:
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) <= 15:
                angles.append(angle)
    if len(angles) < 3 or np.std(angles) > 3 or abs(float(np.median(angles))) < 1:
        return None
    return float(np.median(angles))
