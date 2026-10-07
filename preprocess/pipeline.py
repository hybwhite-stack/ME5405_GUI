from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import enhancement, filters, morphology, thresholding
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


@dataclass
class Candidate:
    name: str
    image: np.ndarray | None
    steps: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    skip_reason: str = ""


OPERATIONS: dict[str, OperationSpec] = {
    "raw": OperationSpec("raw", "原图", "基础", "不执行预处理。", "any", "same"),
    "roi": OperationSpec("roi", "裁剪文字区域", "基础", "裁剪目标区域以减少背景干扰。", "any", "same", spatial=True),
    "rotate": OperationSpec("rotate", "旋转校正", "几何", "按指定角度旋转并扩展画布。", "any", "same", {"angle": None}, True),
    "perspective": OperationSpec("perspective", "透视校正", "几何", "用四角将斜拍区域拉正。", "any", "same", {"corners": None}, True),
    "gray": OperationSpec("gray", "灰度化", "基础", "将彩色图转换为单通道灰度图。", "color_or_gray", "gray"),
    "contrast": OperationSpec("contrast", "对比度拉伸", "增强", "用2%和98%分位点扩展灰度范围。", "gray", "gray", {"low_percentile": 2, "high_percentile": 98}),
    "equalize": OperationSpec("equalize", "全局直方图均衡", "增强", "重新分配整幅图的灰度直方图。", "gray", "gray"),
    "clahe": OperationSpec("clahe", "CLAHE局部增强", "增强", "分块提升局部对比度并限制噪声放大。", "gray", "gray", {"clip_limit": 2.0, "tile_grid": [8, 8]}),
    "median": OperationSpec("median", "中值滤波", "滤波", "抑制脉冲噪声并尽量保持边缘。", "gray", "gray", {"kernel": 3}),
    "gaussian": OperationSpec("gaussian", "高斯滤波", "滤波", "平滑高频噪声。", "gray", "gray", {"kernel": 3, "sigma": 0.8}),
    "bilateral": OperationSpec("bilateral", "双边滤波", "滤波", "平滑纹理并尽量保持边缘。", "gray", "gray", {"diameter": 5, "sigma_color": 40, "sigma_space": 40}),
    "otsu": OperationSpec("otsu", "Otsu二值化", "二值化", "整幅图使用一个自动全局阈值。", "gray", "binary"),
    "adaptive": OperationSpec("adaptive", "自适应二值化", "二值化", "不同区域使用局部阈值。", "gray", "binary", {"block_size": 31, "constant": 11}),
    "open": OperationSpec("open", "形态学开运算", "形态学", "去除小型前景噪点。", "binary", "binary", {"kernel": [2, 2], "iterations": 1}),
    "close": OperationSpec("close", "形态学闭运算", "形态学", "连接轻微断裂笔画。", "binary", "binary", {"kernel": [2, 2], "iterations": 1}),
    "sharpen": OperationSpec("sharpen", "温和锐化", "增强", "使用反锐化掩模增强边缘，不是真正去模糊。", "gray", "gray", {"sigma": 1.0, "amount": 0.5}),
}


PRESETS: dict[str, dict] = {
    "raw": {"label": "原图基准", "recipe": ["raw"]},
    "low_contrast": {"label": "整体低对比度", "recipe": ["gray", "contrast"]},
    "global_equalize": {"label": "整体偏暗/灰度集中", "recipe": ["gray", "equalize"]},
    "uneven_light": {"label": "光照不均", "recipe": ["gray", "clahe"]},
    "uniform_binary": {"label": "均匀光照二值化", "recipe": ["gray", "median", "otsu"]},
    "uneven_binary": {"label": "不均匀光照二值化", "recipe": ["gray", "median", "adaptive"]},
    "remove_speckles": {"label": "去除小噪点", "recipe": ["gray", "median", "otsu", "open"]},
    "repair_strokes": {"label": "连接断裂笔画", "recipe": ["gray", "median", "otsu", "close"]},
    "slight_blur": {"label": "轻微模糊", "recipe": ["gray", "sharpen"]},
    "perspective_clahe": {"label": "斜拍校正并增强", "recipe": ["perspective", "gray", "clahe"]},
    "roi_clahe": {"label": "裁剪文字区并增强", "recipe": ["roi", "gray", "clahe"]},
}


LEGACY_METHODS = (
    "raw", "gray", "contrast", "equalize", "clahe", "median", "gaussian",
    "bilateral", "otsu", "adaptive", "open", "close", "sharpen", "geometry", "roi",
)

LEGACY_RECIPES = {
    "raw": ["raw"],
    "gray": ["gray"],
    "contrast": ["gray", "contrast"],
    "equalize": ["gray", "equalize"],
    "clahe": ["gray", "clahe"],
    "median": ["gray", "median"],
    "gaussian": ["gray", "gaussian"],
    "bilateral": ["gray", "bilateral"],
    "otsu": ["gray", "median", "otsu"],
    "adaptive": ["gray", "median", "adaptive"],
    "open": ["gray", "median", "otsu", "open"],
    "close": ["gray", "median", "otsu", "close"],
    "sharpen": ["gray", "sharpen"],
    "roi": ["roi", "gray", "clahe"],
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
                raise ValueError(f"{operation_id}: params 必须是对象")
        else:
            raise ValueError("流程步骤必须是操作名称或包含 op/params 的对象")
        if operation_id not in OPERATIONS:
            raise ValueError(f"未知预处理操作: {operation_id}")
        result.append({"op": operation_id, "params": deepcopy(params)})
    return result


def normalize_recipe(recipe: list[str | dict], initial_kind: str = "color", auto_gray: bool = True) -> list[dict]:
    """Validate a recipe and visibly insert required grayscale conversion."""
    items = _coerce_recipe(recipe or ["raw"])
    names = [item["op"] for item in items]
    if "raw" in names and len(names) != 1:
        raise ValueError("raw只能单独使用，不能与其他预处理步骤组合")
    if sum(name in {"otsu", "adaptive"} for name in names) > 1:
        raise ValueError("Otsu和adaptive是互斥的二值化方法，同一流程只能选择一种")
    kind = initial_kind
    normalized: list[dict] = []
    pixel_processing_started = False
    for item in items:
        spec = OPERATIONS[item["op"]]
        if spec.spatial:
            if pixel_processing_started:
                raise ValueError(f"{spec.label}必须放在灰度、增强、滤波和二值化之前")
        elif item["op"] != "raw":
            pixel_processing_started = True
        if spec.requires == "gray" and kind == "color":
            if not auto_gray:
                raise ValueError(f"{spec.label}需要先进行灰度化")
            if not normalized or normalized[-1]["op"] != "gray":
                normalized.append({"op": "gray", "params": {}, "auto_inserted": True})
            kind = "gray"
        if spec.requires == "gray" and kind == "binary":
            raise ValueError(f"{spec.label}不能放在二值化之后")
        if spec.requires == "binary" and kind != "binary":
            raise ValueError(f"{spec.label}需要先选择Otsu或adaptive二值化")
        if spec.requires == "color_or_gray" and kind == "binary":
            raise ValueError("二值化后不能再次灰度化")
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
    if operation_id == "open":
        return morphology.opening(image, tuple(params["kernel"]), params["iterations"]), params, extras
    if operation_id == "close":
        return morphology.closing(image, tuple(params["kernel"]), params["iterations"]), params, extras
    if operation_id == "sharpen":
        result, blurred = enhancement.sharpen(image, params["sigma"], params["amount"])
        extras.append(("blur", blurred))
        return result, params, extras
    config = context.get("config", {})
    detections = context.get("raw_detections", [])
    if operation_id == "roi":
        roi = params.get("roi") or config.get("roi") or roi_from_detections(image, detections)
        if roi is None:
            raise PipelineUnavailable("未设置ROI且原图没有可靠文字框；请在窗口中选择文字区域")
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
            raise PipelineUnavailable("未设置旋转角度且没有可靠一致文字倾角")
        result, metadata = rotate_expanded(image, float(angle))
        metadata["source"] = "manual_angle" if params.get("angle") is not None or "angle" in config else "automatic_text_angle"
        return result, metadata, extras
    if operation_id == "perspective":
        corners = params.get("corners") or config.get("corners")
        source = "manual_corners"
        if corners is None:
            detected = detect_document_quad(image)
            if detected is None:
                raise PipelineUnavailable("未设置四角且没有检测到可靠四边形；请在窗口中依次选择左上、右上、右下、左下")
            corners = detected.tolist()
            source = "automatic_document_quad_requires_visual_review"
        result, metadata = perspective(image, corners)
        metadata["source"] = source
        return result, metadata, extras
    raise ValueError(f"未知预处理操作: {operation_id}")


def run_pipeline(image: np.ndarray, recipe: list[str | dict], context: dict | None = None) -> PipelineResult:
    initial_kind = "color" if image.ndim == 3 else "gray"
    normalized = normalize_recipe(recipe, initial_kind=initial_kind, auto_gray=True)
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


def _legacy_geometry(image: np.ndarray, config: dict, detections: list[dict]) -> Candidate:
    try:
        if "corners" in config:
            recipe = [{"op": "perspective", "params": {"corners": config["corners"]}}]
        elif "angle" in config:
            recipe = [{"op": "rotate", "params": {"angle": config["angle"]}}]
        else:
            quad = detect_document_quad(image)
            if quad is not None:
                recipe = [{"op": "perspective", "params": {"corners": quad.tolist()}}]
            else:
                angle = consistent_text_angle(detections)
                if angle is None:
                    raise PipelineUnavailable("没有可靠四边形或一致文字倾角；可手动配置 corners/angle")
                recipe = [{"op": "rotate", "params": {"angle": angle}}]
        run = run_pipeline(image, recipe, {"config": config, "raw_detections": detections})
        metadata = dict(run.metadata)
        if run.metadata.get("operations"):
            metadata.update({key: value for key, value in run.metadata["operations"][-1].items() if key not in {"op", "label", "params"}})
        return Candidate("geometry", run.image, run.steps, metadata)
    except PipelineUnavailable as exc:
        return Candidate("geometry", None, skip_reason=str(exc))


def make_legacy_candidate(name: str, image: np.ndarray, config: dict, raw_detections: list[dict]) -> Candidate:
    if name not in LEGACY_METHODS:
        raise ValueError(f"Unknown method: {name}")
    if name == "geometry":
        return _legacy_geometry(image, config, raw_detections)
    try:
        run = run_pipeline(image, LEGACY_RECIPES[name], {"config": config, "raw_detections": raw_detections})
        metadata = dict(run.metadata)
        if name == "roi" and run.metadata.get("operations"):
            roi_meta = next((item for item in run.metadata["operations"] if item["op"] == "roi"), {})
            metadata.update({key: value for key, value in roi_meta.items() if key not in {"op", "label", "params"}})
        if name == "roi":
            metadata.setdefault("note", "Region segmentation only; auto ROI inherits missed detections from raw OCR.")
        return Candidate(name, run.image, run.steps, metadata)
    except PipelineUnavailable as exc:
        return Candidate(name, None, skip_reason=str(exc))
