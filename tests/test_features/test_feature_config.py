"""Tests for feature configuration and metadata."""

import pytest
from datetime import datetime

from src.features.feature_config import (
    FeatureMetadata,
    FEATURE_VERSION,
    ROLLING_WINDOWS,
    get_all_feature_names,
    get_feature_descriptions,
    MIN_FEATURE_COUNT,
)


def test_feature_version():
    """Test that feature version is defined."""
    assert FEATURE_VERSION is not None
    assert isinstance(FEATURE_VERSION, str)
    assert FEATURE_VERSION.startswith("v")


def test_rolling_windows():
    """Test that rolling windows are defined."""
    assert len(ROLLING_WINDOWS) >= 3
    assert 3 in ROLLING_WINDOWS
    assert 5 in ROLLING_WINDOWS
    assert 10 in ROLLING_WINDOWS


def test_get_all_feature_names():
    """Test that feature name generation produces expected count."""
    feature_names = get_all_feature_names()

    # Should have at least MIN_FEATURE_COUNT features
    assert len(feature_names) >= MIN_FEATURE_COUNT

    # Should have no duplicates
    assert len(feature_names) == len(set(feature_names))

    # Check for expected feature patterns
    assert any("points_scored_avg_" in name for name in feature_names)
    assert any("points_allowed_avg_" in name for name in feature_names)
    assert any("turnover_diff_avg_" in name for name in feature_names)
    assert any("yards_per_play" in name for name in feature_names)

    # Check situational features
    assert "home_win_pct" in feature_names
    assert "rest_days" in feature_names
    assert "current_streak" in feature_names

    # Check derived features
    assert "point_diff_trend_5" in feature_names


def test_get_feature_descriptions():
    """Test that all features have descriptions."""
    descriptions = get_feature_descriptions()

    # All features should have descriptions
    feature_names = get_all_feature_names()
    for name in feature_names:
        assert name in descriptions
        assert len(descriptions[name]) > 0


def test_feature_metadata():
    """Test FeatureMetadata class."""
    metadata = FeatureMetadata(
        version="v1.0",
        creation_date=datetime.now(),
        feature_names=["feature1", "feature2"],
        row_count=100,
        season_range=(2020, 2024),
        description="Test features",
    )

    # Test to_dict
    metadata_dict = metadata.to_dict()
    assert metadata_dict["version"] == "v1.0"
    assert len(metadata_dict["feature_names"]) == 2
    assert metadata_dict["row_count"] == 100
    assert metadata_dict["season_range"] == (2020, 2024)

    # Test from_dict
    restored = FeatureMetadata.from_dict(metadata_dict)
    assert restored.version == metadata.version
    assert restored.feature_names == metadata.feature_names
    assert restored.row_count == metadata.row_count


def test_min_feature_count():
    """Test that we meet minimum feature count requirement."""
    feature_names = get_all_feature_names()
    assert len(feature_names) >= MIN_FEATURE_COUNT

    # Should have well over 20 features
    print(f"Total features: {len(feature_names)}")
    assert len(feature_names) >= 50  # We should have 50+ features
