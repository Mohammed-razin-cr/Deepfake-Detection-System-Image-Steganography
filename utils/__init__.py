"""Utilities package for deepfake detection."""

from .preprocessing import ImagePreprocessor, VideoProcessor
from .inference import InferenceEngine

# Lazy import to avoid scipy dependency issues on startup
def __getattr__(name):
    if name == 'MetricsCalculator':
        from .metrics import MetricsCalculator
        return MetricsCalculator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ImagePreprocessor',
    'VideoProcessor',
    'InferenceEngine',
    'MetricsCalculator'
]
