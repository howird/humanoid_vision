"""PHALP Pipeline Modules.

This package provides modular implementations of each stage in the PHALP
tracking pipeline, with comprehensive type annotations.
"""

from humanoid_vision.pipeline import detection
from humanoid_vision.pipeline import feature_extraction
from humanoid_vision.pipeline import association

__all__ = [
    "types",
    "detection",
    "feature_extraction",
    "association",
]
