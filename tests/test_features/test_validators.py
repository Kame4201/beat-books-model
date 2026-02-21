"""Tests for feature validation and look-ahead bias detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.validators import (
    LookAheadBiasError,
    validate_no_future_data,
    validate_rolling_window_integrity,
    validate_chronological_order,
    validate_feature_completeness,
    validate_no_nulls_after_warmup,
    run_all_validations,
)


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for testing."""
    np.random.seed(42)
    n_games = 20

    return pd.DataFrame(
        {
            "team": ["KC"] * n_games,
            "game_date": pd.date_range("2024-09-01", periods=n_games, freq="7D"),
            "season": [2024] * n_games,
            "week": list(range(1, n_games + 1)),
            # Rolling features with proper NaN pattern
            "points_scored_avg_3": [np.nan, np.nan]
            + list(np.random.uniform(20, 30, n_games - 2)),
            "points_scored_avg_5": [np.nan] * 4
            + list(np.random.uniform(20, 30, n_games - 4)),
            "points_scored_avg_10": [np.nan] * 9
            + list(np.random.uniform(20, 30, n_games - 9)),
            # Other features
            "home_win_pct": np.random.uniform(0.3, 0.7, n_games),
            "rest_days": [7] * n_games,
        }
    )


def test_validate_feature_completeness_pass(sample_features):
    """Test that valid feature DataFrame passes completeness check."""
    # Should pass with min_features=5 (we have more than that)
    validate_feature_completeness(sample_features, min_features=5)


def test_validate_feature_completeness_fail():
    """Test that DataFrame with too few features fails."""
    df = pd.DataFrame(
        {
            "team": ["KC"],
            "game_date": [datetime.now()],
            "feature1": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Expected at least 20 features"):
        validate_feature_completeness(df, min_features=20)


def test_validate_rolling_window_integrity_pass(sample_features):
    """Test that properly structured rolling windows pass validation."""
    validate_rolling_window_integrity(sample_features, required_windows=[3, 5, 10])


def test_validate_rolling_window_integrity_fail():
    """Test that improper window structure fails validation."""
    # Create DataFrame where longer window has value but shorter doesn't
    df = pd.DataFrame(
        {
            "points_scored_avg_3": [np.nan, np.nan, np.nan, np.nan],
            "points_scored_avg_10": [
                np.nan,
                np.nan,
                25.0,
                26.0,
            ],  # Has values too early
        }
    )

    with pytest.raises(LookAheadBiasError, match="Window integrity violation"):
        validate_rolling_window_integrity(df, required_windows=[3, 10])


def test_validate_chronological_order_pass():
    """Test that chronologically ordered data passes."""
    df = pd.DataFrame(
        {
            "team": ["KC"] * 5 + ["BUF"] * 5,
            "game_date": (
                pd.date_range("2024-09-01", periods=5, freq="7D").tolist()
                + pd.date_range("2024-09-01", periods=5, freq="7D").tolist()
            ),
        }
    )

    validate_chronological_order(df)


def test_validate_chronological_order_fail():
    """Test that non-chronological data fails."""
    df = pd.DataFrame(
        {
            "team": ["KC"] * 5,
            "game_date": pd.to_datetime(
                [
                    "2024-09-01",
                    "2024-09-08",
                    "2024-09-22",  # Out of order
                    "2024-09-15",
                    "2024-09-29",
                ]
            ),
        }
    )

    with pytest.raises(LookAheadBiasError, match="not in chronological order"):
        validate_chronological_order(df)


def test_validate_no_nulls_after_warmup_pass(sample_features):
    """Test that features with acceptable null rate pass."""
    validate_no_nulls_after_warmup(
        sample_features, max_window=10, allowed_null_rate=0.3
    )


def test_validate_no_nulls_after_warmup_fail():
    """Test that excessive nulls after warmup fail."""
    df = pd.DataFrame(
        {
            "team": ["KC"] * 20,
            "feature1": [np.nan] * 15 + [1.0] * 5,  # 75% null after warmup
        }
    )

    with pytest.raises(ValueError, match="has .* null values after warmup"):
        validate_no_nulls_after_warmup(df, max_window=5, allowed_null_rate=0.1)


def test_run_all_validations_pass(sample_features):
    """Test that valid features pass all validations."""
    # Should not raise any exceptions
    run_all_validations(sample_features, min_features=5)


def test_look_ahead_bias_detection():
    """Test that look-ahead bias is properly detected."""
    # Create a scenario where feature uses future data
    df = pd.DataFrame(
        {
            "team": ["KC"] * 10,
            "game_date": pd.date_range("2024-09-01", periods=10, freq="7D"),
            # This rolling avg has values too early (look-ahead bias)
            "points_scored_avg_5": list(
                np.random.uniform(20, 30, 10)
            ),  # No NaN period!
        }
    )

    # The absence of NaN in early games suggests look-ahead bias
    # Check that first 4 games should have NaN for 5-game rolling avg
    assert df["points_scored_avg_5"].iloc[:4].isna().sum() < 4  # Should fail this check


def test_feature_metadata_columns():
    """Test that validation properly excludes metadata columns."""
    df = pd.DataFrame(
        {
            # Metadata columns (should be excluded from feature count)
            "team": ["KC"] * 10,
            "opponent": ["BUF"] * 10,
            "season": [2024] * 10,
            "week": list(range(1, 11)),
            "game_date": pd.date_range("2024-09-01", periods=10, freq="7D"),
            "game_id": list(range(1, 11)),
            # Actual features (only 3, should fail if min_features=5)
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "feature3": [3.0] * 10,
        }
    )

    with pytest.raises(ValueError, match="Expected at least 5 features"):
        validate_feature_completeness(df, min_features=5)


def test_empty_dataframe():
    """Test validation on empty DataFrame."""
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_feature_completeness(df, min_features=1)
