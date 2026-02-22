"""
Feature engineering module for NFL game predictions.

Provides tools for computing, validating, and storing predictive features
from raw NFL statistics with zero look-ahead bias.
"""

from src.features.feature_config import (
    FeatureMetadata,
    FEATURE_VERSION,
    ROLLING_WINDOWS,
    get_all_feature_names,
    get_feature_descriptions,
)
from src.features.validators import (
    LookAheadBiasError,
    validate_no_future_data,
    validate_rolling_window_integrity,
    validate_chronological_order,
    validate_feature_completeness,
    run_all_validations,
)

__all__ = [
    # Core classes (lazy to avoid requiring DATABASE_URL at import time)
    "FeatureEngineer",
    "FeatureBuilder",
    "FeatureStore",
    "FeatureMetadata",
    # Configuration
    "FEATURE_VERSION",
    "ROLLING_WINDOWS",
    "get_all_feature_names",
    "get_feature_descriptions",
    # Validation
    "LookAheadBiasError",
    "validate_no_future_data",
    "validate_rolling_window_integrity",
    "validate_chronological_order",
    "validate_feature_completeness",
    "run_all_validations",
]


def __getattr__(name: str):
    if name == "FeatureEngineer":
        from src.features.feature_engineering import FeatureEngineer

        return FeatureEngineer
    if name == "FeatureBuilder":
        from src.features.feature_builder import FeatureBuilder

        return FeatureBuilder
    if name == "FeatureStore":
        from src.features.feature_store import FeatureStore

        return FeatureStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
