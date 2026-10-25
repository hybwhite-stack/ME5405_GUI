from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from . import enhancement, filters, thresholding
from .basic import crop_roi, roi_from_detections, to_gray
from .geometry import consistent_text_angle, detect_document_quad, perspective, rotate_expanded


class PipelineUnavailable(ValueError):
    """The requested spatial operation lacks reliable coordinates or evidence."""


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    label: str
    category: str
    description: str
    requires: str
    produces: str
    default_params: dict = field(default_factory=dict)
    spatial: bool = False


@dataclass
class PipelineResult:
    image: np.ndarray
    recipe: list[dict]
    steps: list[tuple[str, np.ndarray]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


OPERATIONS: dict[str, OperationSpec] = {
    "raw": OperationSpec("raw", "Original", "Basic", "No preprocessing.", "any", "same"),
    "roi": OperationSpec("roi", "Crop region", "Basic", "Crop the selected region.", "any", "same", spatial=True),
    "rotate": OperationSpec("rotate", "Rotate", "Geometry", "Rotate by an angle.", "any", "same", {"angle": None}, True),
    "perspective": OperationSpec("perspective", "Perspective correction", "Geometry", "Straighten a four-corner region.", "any", "same", {"corners": None}, True),
    "gray": OperationSpec("gray", "Grayscale", "Basic", "Convert to grayscale.", "color_or_gray", "gray"),
    "contrast": OperationSpec("contrast", "Contrast stretch", "Enhancement", "Stretch the grayscale range.", "gray", "gray", {"low_percentile": 2, "high_percentile": 98}),
    "equalize": OperationSpec("equalize", "Histogram equalization", "Enhancement", "Equalize the image histogram.", "gray", "gray"),
    "clahe": OperationSpec("clahe", "CLAHE", "Enhancement", "Improve local contrast.", "gray", "gray", {"clip_limit": 2.0, "tile_grid": [8, 8]}),
    "median": OperationSpec("median", "Median filter", "Filtering", "Reduce impulse noise.", "gray", "gray", {"kernel": 3}),
    "gaussian": OperationSpec("gaussian", "Gaussian filter", "Filtering", "Smooth image noise.", "gray", "gray", {"kernel": 3, "sigma": 0.8}),
    "bilateral": OperationSpec("bilateral", "Bilateral filter", "Filtering", "Smooth while keeping edges.", "gray", "gray", {"diameter": 5, "sigma_color": 40, "sigma_space": 40}),
    "otsu": OperationSpec("otsu", "Otsu threshold", "Thresholding", "Use one automatic threshold.", "gray", "binary"),
    "adaptive": OperationSpec("adaptive", "Adaptive threshold", "Thresholding", "Use local thresholds.", "gray", "binary", {"block_size": 31, "constant": 11}),
    "sharpen": OperationSpec("sharpen", "Sharpen", "Enhancement", "Increase edge contrast.", "gray", "gray", {"sigma": 1.0, "amount": 0.5}),
}


def _coerce_recipe(recipe: list[str | dict]) -> list[dict]:
    result = []
    for item in recipe:
        if isinstance(item, str):
            operation_id, params = item, {}
        elif isinstance(item, dict) and isinstance(item.get("op"), str):
            operation_id = item["op"]
            params = item.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"{operation_id}: params must be an object")
        else:
            raise ValueError("Each step must be an operation name or an op/params object")
        if operation_id not in OPERATIONS:
            raise ValueError(f"Unknown preprocessing operation: {operation_id}")
        result.append({"op": operation_id, "params": deepcopy(params)})
    return result


def normalize_recipe(recipe: list[str | dict], initial_kind: str = "color") -> list[dict]:
    """Validate an ordered recipe selected by the user."""
    items = _coerce_recipe(recipe or ["raw"])
    names = [item["op"] for item in items]
    if "raw" in names and len(names) != 1:
        raise ValueError("The original image cannot be combined with other steps")
    if sum(name in {"otsu", "adaptive"} for name in names) > 1:
        raise ValueError("Choose either Otsu or adaptive thresholding, not both")
    kind = initial_kind
    normalized: list[dict] = []
    pixel_processing_started = False
    for item in items:
        spec = OPERATIONS[item["op"]]
        if spec.spatial:
            if pixel_processing_started:
                raise ValueError(f"{spec.label} must come before grayscale, enhancement, filtering, and thresholding")
        elif item["op"] != "raw":
            pixel_processing_started = True
        if spec.requires == "gray" and kind == "color":
            raise ValueError(f"Add Grayscale before {spec.label}")
        if spec.requires == "gray" and kind == "binary":
            raise ValueError(f"{spec.label} cannot come after thresholding")
        if spec.requires == "color_or_gray" and kind == "binary":
            raise ValueError("Grayscale cannot come after thresholding")
        normalized.append(item)
        if spec.produces != "same":
            kind = spec.produces
    return normalized


def recipe_label(recipe: list[str | dict]) -> str:
    return " → ".join(OPERATIONS[item["op"]].label for item in _coerce_recipe(recipe))


def _params(spec: OperationSpec, supplied: dict) -> dict:
    params = deepcopy(spec.default_params)
    params.update(supplied)
    return params


def _apply(operation_id: str, image: np.ndarray, params: dict, context: dict) -> tuple[np.ndarray, dict, list[tuple[str, np.ndarray]]]:
    metadata: dict = {}
    extras: list[tuple[str, np.ndarray]] = []
    if operation_id == "raw":
        return image.copy(), {"operations": "none"}, extras
    if operation_id == "gray":
        return to_gray(image), metadata, extras
    if operation_id == "contrast":
        return enhancement.contrast_stretch(image, params["low_percentile"], params["high_percentile"]), params, extras
    if operation_id == "equalize":
        return enhancement.equalize(image), metadata, extras
    if operation_id == "clahe":
        grid = tuple(params["tile_grid"])
        return enhancement.clahe(image, params["clip_limit"], grid), params, extras
    if operation_id == "median":
        return filters.median(image, params["kernel"]), params, extras
    if operation_id == "gaussian":
        return filters.gaussian(image, params["kernel"], params["sigma"]), params, extras
    if operation_id == "bilateral":
        return filters.bilateral(image, params["diameter"], params["sigma_color"], params["sigma_space"]), params, extras
    if operation_id == "otsu":
        return thresholding.otsu(image), {"threshold": "otsu"}, extras
    if operation_id == "adaptive":
        return thresholding.adaptive(image, params["block_size"], params["constant"]), params, extras
    if operation_id == "sharpen":
        result, blurred = enhancement.sharpen(image, params["sigma"], params["amount"])
        extras.append(("blur", blurred))
        return result, params, extras
    config = context.get("config", {})
    detections = context.get("raw_detections", [])
    if operation_id == "roi":
        roi = params.get("roi") or config.get("roi") or roi_from_detections(image, detections)
        if roi is None:
            raise PipelineUnavailable("No crop region was set; select a region in the image window")
        result, metadata = crop_roi(image, roi)
        metadata["source"] = "manual_roi" if params.get("roi") or config.get("roi") else "raw_ocr_union_with_20px_margin"
        return result, metadata, extras
    if operation_id == "rotate":
        angle = params.get("angle")
        if angle is None:
            angle = config.get("angle")
        if angle is None:
            angle = consistent_text_angle(detections)
        if angle is None:
            raise PipelineUnavailable("No rotation angle was set or detected")
        result, metadata = rotate_expanded(image, float(angle))
        metadata["source"] = "manual_angle" if params.get("angle") is not None or "angle" in config else "automatic_text_angle"
        return result, metadata, extras
    if operation_id == "perspective":
        corners = params.get("corners") or config.get("corners")
        source = "manual_corners"
        if corners is None:
            detected = detect_document_quad(image)
            if detected is None:
                raise PipelineUnavailable("No quadrilateral was found; select its four corners in the image window")
            corners = detected.tolist()
            source = "automatic_document_quad_requires_visual_review"
        result, metadata = perspective(image, corners)
        metadata["source"] = source
        return result, metadata, extras
    raise ValueError(f"Unknown preprocessing operation: {operation_id}")


def run_pipeline(image: np.ndarray, recipe: list[str | dict], context: dict | None = None) -> PipelineResult:
    initial_kind = "color" if image.ndim == 3 else "gray"
    normalized = normalize_recipe(recipe, initial_kind=initial_kind)
    result = image.copy()
    steps: list[tuple[str, np.ndarray]] = []
    operation_metadata = []
    for index, item in enumerate(normalized, 1):
        spec = OPERATIONS[item["op"]]
        params = _params(spec, item.get("params", {}))
        result, metadata, extras = _apply(item["op"], result, params, context or {})
        for extra_name, extra_image in extras:
            steps.append((f"{len(steps) + 1:02}_{extra_name}", extra_image))
        if item["op"] != "raw":
            steps.append((f"{len(steps) + 1:02}_{item['op']}", result))
        operation_metadata.append({"op": item["op"], "label": spec.label, "params": params, **metadata})
    return PipelineResult(result, normalized, steps, {"operations": operation_metadata})
