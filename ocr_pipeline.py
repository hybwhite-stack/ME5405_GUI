from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CONFIDENCE = 0.20
CSV_FIELDS = [
    "image_name",
    "region_id",
    "text",
    "confidence",
    "accepted",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


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


def annotation_filename(image_path: Path) -> str:
    suffix_name = image_path.suffix.lower().lstrip(".")
    return f"{image_path.stem}_{suffix_name}_ocr.png"


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


def save_annotated_image(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    annotated = render_annotated_image(image_bgr, detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise ValueError(f"Cannot encode annotated image: {output_path}")
    encoded.tofile(str(output_path))


def write_csv(detections: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for detection in detections:
            writer.writerow(
                {
                    "image_name": detection["image_name"],
                    "region_id": detection["region_id"],
                    "text": detection["text"],
                    "confidence": f'{detection["confidence"]:.6f}',
                    "accepted": str(detection["accepted"]).lower(),
                    "x_min": detection["x_min"],
                    "y_min": detection["y_min"],
                    "x_max": detection["x_max"],
                    "y_max": detection["y_max"],
                }
            )


def write_json(image_results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(image_results, stream, ensure_ascii=False, indent=2)


def write_summary(image_results: list[dict[str, Any]], output_path: Path) -> None:
    lines = ["ME5405 Raw Image OCR Baseline", ""]
    if not image_results:
        lines.append("No supported images were found.")

    for result in image_results:
        lines.append(f'Image: {result["image_name"]}')
        lines.append(f'Status: {result["status"]}')
        if result["status"] == "ok":
            accepted = [item for item in result["detections"] if item["accepted"]]
            lines.append(f'Detected regions: {len(result["detections"])}')
            lines.append(f'Accepted regions: {len(accepted)}')
            lines.append("Accepted text: " + " | ".join(item["text"] for item in accepted))
        else:
            lines.append(f'Error: {result["error"]}')
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    image_results: list[dict[str, Any]],
    all_detections: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    write_csv(all_detections, output_dir / "results.csv")
    write_json(image_results, output_dir / "results.json")
    write_summary(image_results, output_dir / "summary.txt")


def run_batch_ocr(
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    use_gpu: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated").mkdir(parents=True, exist_ok=True)
    images = find_images(input_dir)

    if not images:
        _write_outputs([], [], output_dir)
        print(f"未找到支持的原始图片：{input_dir}")
        print("请把 JPG、PNG、BMP 或 TIFF 图片放入该目录后重新运行。")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    print("正在初始化 EasyOCR（首次运行会下载中英文模型）...")
    reader = easyocr.Reader(
        ["ch_sim", "en"],
        gpu=use_gpu,
        model_storage_directory=str(model_dir),
        user_network_directory=str(model_dir / "user_network"),
        download_enabled=True,
        verbose=False,
    )

    image_results: list[dict[str, Any]] = []
    all_detections: list[dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] 处理：{image_path.name}")
        try:
            image_bgr = load_image(image_path)
            raw_detections = reader.readtext(image_bgr, detail=1, rotation_info=None,
                                            contrast_ths=0.0, workers=0, paragraph=False)
            detections = [
                normalize_detection(image_path.name, region_id, raw, confidence_threshold)
                for region_id, raw in enumerate(raw_detections, start=1)
            ]
            save_annotated_image(
                image_bgr,
                detections,
                output_dir / "annotated" / annotation_filename(image_path),
            )
            image_results.append(
                {
                    "image_name": image_path.name,
                    "status": "ok",
                    "detections": detections,
                }
            )
            all_detections.extend(detections)
        except Exception as exc:  # Continue the batch when one file is unreadable.
            message = f"{type(exc).__name__}: {exc}"
            print(f"警告：跳过 {image_path.name}，原因：{message}")
            image_results.append(
                {
                    "image_name": image_path.name,
                    "status": "error",
                    "error": message,
                    "detections": [],
                }
            )

    _write_outputs(image_results, all_detections, output_dir)
    succeeded = sum(result["status"] == "ok" for result in image_results)
    print(f"完成：{succeeded}/{len(image_results)} 张图片处理成功。")
    print(f"结果目录：{output_dir}")
    return 0 if succeeded > 0 else 1
