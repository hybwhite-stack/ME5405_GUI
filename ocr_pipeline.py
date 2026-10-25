from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def load_image(image_path: Path) -> np.ndarray:
    """Read an image through bytes so Windows paths may contain Chinese characters."""
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV 无法解码该图片")
    return image


def _font_candidates() -> Iterable[Path]:
    fonts_dir = Path("C:/Windows/Fonts")
    yield fonts_dir / "msyh.ttc"
    yield fonts_dir / "msyhbd.ttc"
    yield fonts_dir / "simhei.ttf"
    yield fonts_dir / "simsun.ttc"


def load_annotation_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Keep labels readable after large photographs are scaled down in HTML previews.
    font_size = max(26, min(48, image_height // 35))
    for font_path in _font_candidates():
        if font_path.exists():
            return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def normalize_detection(
    image_name: str,
    region_id: int,
    raw_detection: list[Any] | tuple[Any, ...],
    threshold: float,
) -> dict[str, Any]:
    polygon, text, confidence = raw_detection
    points = [[float(point[0]), float(point[1])] for point in polygon]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    confidence_value = float(confidence)
    return {
        "image_name": image_name,
        "region_id": region_id,
        "text": str(text),
        "confidence": confidence_value,
        "accepted": confidence_value >= threshold,
        "polygon": points,
        "x_min": math.floor(min(xs)),
        "y_min": math.floor(min(ys)),
        "x_max": math.ceil(max(xs)),
        "y_max": math.ceil(max(ys)),
    }


def render_annotated_image(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Return an annotated BGR image without writing it to disk."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas)
    font = load_annotation_font(canvas.height)
    line_width = max(3, min(8, canvas.height // 250))
    label_padding = max(3, line_width - 1)

    for detection in detections:
        if not detection["accepted"]:
            continue
        polygon = [(round(x), round(y)) for x, y in detection["polygon"]]
        draw.line(polygon + [polygon[0]], fill=(0, 220, 0), width=line_width)
        text = detection["text"]
        display_text = text if len(text) <= 28 else f"{text[:27]}…"
        label = (
            f'{detection["region_id"]}: {display_text} '
            f'({detection["confidence"]:.2f})'
        )
        anchor_x = max(0, detection["x_min"])
        try:
            measured = draw.textbbox((0, 0), label, font=font)
            label_height = measured[3] - measured[1]
            anchor_y = max(label_padding, detection["y_min"] - label_height - 2 * label_padding)
            box = draw.textbbox((anchor_x, anchor_y), label, font=font)
            padded_box = (
                max(0, box[0] - label_padding),
                max(0, box[1] - label_padding),
                min(canvas.width, box[2] + label_padding),
                min(canvas.height, box[3] + label_padding),
            )
            draw.rectangle(padded_box, fill=(0, 0, 0))
            draw.text((anchor_x, anchor_y), label, font=font, fill=(255, 255, 255))
        except UnicodeEncodeError:
            fallback = f'{detection["region_id"]} ({detection["confidence"]:.2f})'
            anchor_y = max(0, detection["y_min"] - 20)
            draw.text((anchor_x, anchor_y), fallback, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
