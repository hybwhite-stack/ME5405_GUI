"""Composable preprocessing operations for the ME5405 OCR project."""

from .basic import image_diagnostics, save_image, to_gray
from .geometry import perspective
from .pipeline import (
    LEGACY_METHODS,
    OPERATIONS,
    PRESETS,
    Candidate,
    OperationSpec,
    PipelineResult,
    PipelineUnavailable,
    make_legacy_candidate,
    normalize_recipe,
    recipe_label,
    run_pipeline,
)

__all__ = [
    "Candidate",
    "LEGACY_METHODS",
    "OPERATIONS",
    "OperationSpec",
    "PRESETS",
    "PipelineResult",
    "PipelineUnavailable",
    "image_diagnostics",
    "make_legacy_candidate",
    "normalize_recipe",
    "perspective",
    "recipe_label",
    "run_pipeline",
    "save_image",
    "to_gray",
]
