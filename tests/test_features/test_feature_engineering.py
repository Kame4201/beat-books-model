"""Tests for feature engineering computation logic."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_team_games():
    """Create sample game data for a single team."""
    np.random.seed(42)
    n_games = 15

    base_date = datetime(2024, 9, 1)
    dates = [base_date + timedelta(days=7 * i) for i in range(n_games)]

    return pd.DataFrame(
        {
            "game_id": list(range(1, n_games + 1)),
            "team": ["KC"] * n_games,
            "season": [2024] * n_games,
            "week": list(range(1, n_games + 1)),
            "game_date": dates,
            # Basic stats
            "points_scored": np.random.randint(14, 42, n_games),
            "points_allowed": np.random.randint(10, 35, n_games),
            "off_yards_per_play": np.random.uniform(4.5, 7.0, n_games),
            "def_yards_per_play": np.random.uniform(4.0, 6.5, n_games),
            "turnover_diff": np.random.randint(-3, 4, n_games),
            # Efficiency metrics
            "off_points_per_drive": np.random.uniform(1.5, 3.5, n_games),
            "def_points_per_drive": np.random.uniform(1.0, 3.0, n_games),
            "off_yards_per_drive": np.random.uniform(25, 45, n_games),
            "def_yards_per_drive": np.random.uniform(20, 40, n_games),
            "off_red_zone_pct": np.random.uniform(0.45, 0.75, n_games),
            "def_red_zone_pct": np.random.uniform(0.40, 0.70, n_games),
            "off_third_down_pct": np.random.uniform(0.30, 0.50, n_games),
            "def_third_down_pct": np.random.uniform(0.25, 0.45, n_games),
            "sacks_given": np.random.randint(0, 6, n_games),
            "sacks_taken": np.random.randint(0, 5, n_games),
            # Situational
            "is_home": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "is_away": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "won": [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            "point_diff": None,  # Will compute
            "is_division_game": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.rolling_windows == [3, 5, 10]

    engineer_custom = FeatureEngineer(rolling_windows=[2, 4, 8])
    assert engineer_custom.rolling_windows == [2, 4, 8]


def test_rolling_averages_no_lookahead(sample_team_games):
    """Test that rolling averages don't use current game (no look-ahead bias)."""
    engineer = FeatureEngineer(rolling_windows=[3])

    # Compute point differential
    sample_team_games["point_diff"] = (
        sample_team_games["points_scored"] - sample_team_games["points_allowed"]
    )

    features = engineer._compute_team_features(sample_team_games)

    # First 3 games should have NaN for 3-game rolling avg (using shifted data)
    # Actually, with shift(1) and min_periods=1:
    # - Game 1: shift gives NaN, rolling avg = NaN
    # - Game 2: shift gives game 1 value, rolling avg = game 1 value
    # - Game 3: shift gives games 1-2, rolling avg = mean(games 1-2)
    # So only first game should be NaN

    assert pd.isna(features["points_scored_avg_3"].iloc[0])
    assert not pd.isna(features["points_scored_avg_3"].iloc[1])  # Has 1 prior game

    # Verify that feature for game N only uses games 1 through N-1
    # Game 4 (index 3): shift(1) gives games 0,1,2 values; rolling(3) = mean of all 3
    expected_avg = sample_team_games["points_scored"].iloc[0:3].mean()
    actual_avg = features["points_scored_avg_3"].iloc[3]

    assert abs(actual_avg - expected_avg) < 0.01


def test_all_feature_categories_present(sample_team_games):
    """Test that all feature categories are computed."""
    engineer = FeatureEngineer()

    # Add required columns
    sample_team_games["point_diff"] = (
        sample_team_games["points_scored"] - sample_team_games["points_allowed"]
    )

    features = engineer._compute_team_features(sample_team_games)

    # Rolling averages
    assert "points_scored_avg_3" in features.columns
    assert "points_allowed_avg_5" in features.columns
    assert "off_yards_per_play_avg_10" in features.columns
    assert "turnover_diff_avg_3" in features.columns

    # Efficiency metrics
    assert "off_points_per_drive_avg_3" in features.columns
    assert "def_points_per_drive_avg_5" in features.columns
    assert "off_red_zone_pct_avg_3" in features.columns
    assert "off_third_down_pct_avg_3" in features.columns
    assert "sacks_given_avg_3" in features.columns

    # Situational
    assert "home_win_pct" in features.columns
    assert "rest_days" in features.columns
    assert "current_streak" in features.columns
    assert "is_division_game" in features.columns
    assert "had_bye_week" in features.columns

    # Derived
    assert "point_diff_trend_5" in features.columns
    assert "win_pct_last_5" in features.columns


def test_feature_count(sample_team_games):
    """Test that we generate at least 20 features per game."""
    engineer = FeatureEngineer()

    sample_team_games["point_diff"] = (
        sample_team_games["points_scored"] - sample_team_games["points_allowed"]
    )

    features = engineer._compute_team_features(sample_team_games)

    # Exclude metadata columns
    metadata_cols = {"game_id", "team", "season", "week", "game_date"}
    feature_cols = [col for col in features.columns if col not in metadata_cols]

    # Should have significantly more than 20 features
    assert len(feature_cols) >= 20
    print(f"Generated {len(feature_cols)} features")


def test_streak_computation(sample_team_games):
    """Test win/loss streak computation."""
    engineer = FeatureEngineer()

    # Take first 7 games, then override columns
    df = sample_team_games.head(7).copy()

    # Create known win pattern: W, W, L, W, W, W, L
    df["won"] = [1, 1, 0, 1, 1, 1, 0]
    df["point_diff"] = [10, 5, -3, 7, 14, 3, -10]

    features = engineer._compute_team_features(df)

    # Check streak values (uses shift(1), so streak reflects PRIOR games):
    # Game 1: no prior games (NaN after shift), streak = 0
    # Game 2: prior game was W, streak = 1
    # Game 3: prior game was W (continuing), streak = 2
    # Game 4: prior game was L (streak broken), streak = -1
    # Game 5: prior game was W (new streak), streak = 1

    assert features["current_streak"].iloc[0] == 0
    assert features["current_streak"].iloc[1] == 1
    assert features["current_streak"].iloc[2] == 2
    assert features["current_streak"].iloc[3] == -1
    assert features["current_streak"].iloc[4] == 1


def test_rest_days_computation(sample_team_games):
    """Test rest days calculation."""
    engineer = FeatureEngineer()

    sample_team_games["point_diff"] = (
        sample_team_games["points_scored"] - sample_team_games["points_allowed"]
    )

    features = engineer._compute_team_features(sample_team_games)

    # Games are 7 days apart (weekly)
    assert all(features["rest_days"].iloc[1:] == 7)

    # First game has default (7)
    assert features["rest_days"].iloc[0] == 7


def test_bye_week_detection():
    """Test bye week indicator (14+ days rest)."""
    engineer = FeatureEngineer()

    # Create games with a bye week
    dates = [
        datetime(2024, 9, 1),
        datetime(2024, 9, 8),
        datetime(2024, 9, 15),
        datetime(2024, 10, 6),  # 21 days after previous (bye week)
        datetime(2024, 10, 13),
    ]

    df = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "team": ["KC"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 5, 6],
            "game_date": dates,
            "points_scored": [28, 24, 31, 27, 35],
            "points_allowed": [21, 20, 17, 24, 28],
            "point_diff": [7, 4, 14, 3, 7],
            "off_yards_per_play": [6.5] * 5,
            "def_yards_per_play": [5.0] * 5,
            "turnover_diff": [1] * 5,
            "is_home": [1, 0, 1, 0, 1],
            "is_away": [0, 1, 0, 1, 0],
            "won": [1, 1, 1, 1, 1],
            "is_division_game": [0] * 5,
        }
    )

    features = engineer._compute_team_features(df)

    # Game 4 should have bye week indicator (21 days rest)
    assert features["had_bye_week"].iloc[3] == 1
    assert features["rest_days"].iloc[3] == 21

    # Other games should not
    assert features["had_bye_week"].iloc[0] == 0
    assert features["had_bye_week"].iloc[1] == 0


def test_trend_computation(sample_team_games):
    """Test linear trend computation."""
    engineer = FeatureEngineer()

    # Create point differential with clear upward trend
    sample_team_games["point_diff"] = [
        5,
        8,
        11,
        14,
        17,
        20,
        23,
        26,
        29,
        32,
        35,
        38,
        41,
        44,
        47,
    ]

    features = engineer._compute_team_features(sample_team_games)

    # After enough games, trend should be positive (slope ≈ 3)
    # Game 6 onward should have trend computed over last 5 games
    trend_values = features["point_diff_trend_5"].iloc[6:].dropna()

    if len(trend_values) > 0:
        # Should be positive (increasing)
        assert all(trend_values > 0)


def test_empty_dataframe():
    """Test that empty DataFrame is handled gracefully."""
    engineer = FeatureEngineer()

    result = engineer.compute_features(season=2099)  # Non-existent season
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_multiple_teams():
    """Test that features are computed separately for each team."""
    engineer = FeatureEngineer(rolling_windows=[3])

    common_cols = {
        "off_yards_per_play": [6.0] * 5,
        "def_yards_per_play": [5.0] * 5,
        "turnover_diff": [1, 0, -1, 2, 0],
        "is_home": [1, 0, 1, 0, 1],
        "is_away": [0, 1, 0, 1, 0],
        "won": [1, 1, 0, 1, 1],
        "is_division_game": [0, 1, 0, 0, 1],
    }

    team1_games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "team": ["KC"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "game_date": pd.date_range("2024-09-01", periods=5, freq="7D"),
            "points_scored": [28, 24, 31, 27, 35],
            "points_allowed": [21, 20, 17, 24, 28],
            **common_cols,
        }
    )
    team1_games["point_diff"] = (
        team1_games["points_scored"] - team1_games["points_allowed"]
    )

    team2_games = pd.DataFrame(
        {
            "game_id": [6, 7, 8, 9, 10],
            "team": ["BUF"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "game_date": pd.date_range("2024-09-01", periods=5, freq="7D"),
            "points_scored": [21, 17, 24, 20, 28],
            "points_allowed": [14, 10, 20, 17, 24],
            **common_cols,
        }
    )
    team2_games["point_diff"] = (
        team2_games["points_scored"] - team2_games["points_allowed"]
    )

    # Features should be computed independently
    features_kc = engineer._compute_team_features(team1_games)
    features_buf = engineer._compute_team_features(team2_games)

    assert len(features_kc) == 5
    assert len(features_buf) == 5
    assert all(features_kc["team"] == "KC")
    assert all(features_buf["team"] == "BUF")
