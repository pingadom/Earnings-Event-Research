"""Feature engineering. Every feature carries the timestamp at which it became
knowable, and the assembler refuses to emit a panel that violates it."""

from .assemble import assemble_features, cohort_key, cross_sectional_normalise
from .fundamentals import FUNDAMENTAL_FEATURES, build_fundamental_features
from .surprise import build_surprise_features
from .text import TextFeatureExtractor, build_text_features

__all__ = [
    "FUNDAMENTAL_FEATURES",
    "TextFeatureExtractor",
    "assemble_features",
    "cohort_key",
    "build_fundamental_features",
    "build_surprise_features",
    "build_text_features",
    "cross_sectional_normalise",
]
