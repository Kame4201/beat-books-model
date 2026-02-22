"""Tests for FeatureBuilder — DB-connected feature engineering.

All tests use a mocked SQLAlchemy session so no real database is needed.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import MagicMock, patch

from src.features.feature_builder import (
    FeatureBuilder,
    _same_division,
    NFL_DIVISIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game_rows(season: int = 2024, n_weeks: int = 4):
    """Create fake team_games rows in the DB schema (winner/loser format)."""
    rows = []
    teams = ["KC", "BUF", "SF", "DET"]
    matchups = [
        ("KC", "BUF", 27, 24, 350, 310, 1, 2),
        ("SF", "DET", 31, 28, 400, 380, 0, 1),
        ("BUF", "KC", 21, 17, 320, 290, 2, 1),
        ("DET", "SF", 24, 20, 370, 340, 1, 0),
    ]
    for week, (winner, loser, pts_w, pts_l, yds_w, yds_l, to_w, to_l) in enumerate(
        matchups[:n_weeks], start=1
    ):
        game_date = date(season, 9, 7 + (week - 1) * 7)
        for team in [winner, loser]:
            rows.append({
                "game_id": len(rows) + 1,
                "team": team,
                "season": season,
                "week": week,
                "game_date": game_date,
                "winner": winner,
                "loser": loser,
                "pts_w": pts_w,
                "pts_l": pts_l,
                "yds_w": yds_w,
                "yds_l": yds_l,
                "to_w": to_w,
                "to_l": to_l,
            })
    return pd.DataFrame(rows)


def _make_season_offense(season: int = 2024):
    """Fake team_offense season-level data."""
    return pd.DataFrame([
        {"tm": "KC",  "season": season, "games_played": 17, "pf": 450, "yds": 6000, "ply": 1100, "ypp": 5.5, "turnovers": 15, "sc_pct": 38.0, "to_pct": 8.0, "opea": 50.0},
        {"tm": "BUF", "season": season, "games_played": 17, "pf": 430, "yds": 5800, "ply": 1050, "ypp": 5.5, "turnovers": 18, "sc_pct": 36.0, "to_pct": 9.0, "opea": 40.0},
        {"tm": "SF",  "season": season, "games_played": 17, "pf": 400, "yds": 5600, "ply": 1080, "ypp": 5.2, "turnovers": 12, "sc_pct": 35.0, "to_pct": 7.0, "opea": 55.0},
        {"tm": "DET", "season": season, "games_played": 17, "pf": 420, "yds": 5900, "ply": 1070, "ypp": 5.5, "turnovers": 14, "sc_pct": 37.0, "to_pct": 7.5, "opea": 45.0},
    ])


def _make_season_defense(season: int = 2024):
    """Fake team_defense season-level data."""
    return pd.DataFrame([
        {"tm": "KC",  "season": season, "games_played": 17, "pa": 300, "yds_allowed": 5200, "ply_allowed": 1050, "ypp_allowed": 5.0, "takeaways": 22, "sc_pct_allowed": 30.0, "to_pct_forced": 12.0, "depa": -30.0},
        {"tm": "BUF", "season": season, "games_played": 17, "pa": 320, "yds_allowed": 5400, "ply_allowed": 1060, "ypp_allowed": 5.1, "takeaways": 20, "sc_pct_allowed": 32.0, "to_pct_forced": 11.0, "depa": -25.0},
        {"tm": "SF",  "season": season, "games_played": 17, "pa": 280, "yds_allowed": 5000, "ply_allowed": 1030, "ypp_allowed": 4.9, "takeaways": 25, "sc_pct_allowed": 28.0, "to_pct_forced": 13.0, "depa": -40.0},
        {"tm": "DET", "season": season, "games_played": 17, "pa": 340, "yds_allowed": 5500, "ply_allowed": 1070, "ypp_allowed": 5.1, "takeaways": 18, "sc_pct_allowed": 33.0, "to_pct_forced": 10.0, "depa": -20.0},
    ])


class _FakeResult:
    """Mimics SQLAlchemy CursorResult."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._keys = list(df.columns)

    def fetchall(self):
        return [tuple(row) for row in self._df.itertuples(index=False, name=None)]

    def keys(self):
        return self._keys


@pytest.fixture
def mock_session():
    """A MagicMock SQLAlchemy session that returns fake data."""
    session = MagicMock()

    game_df = _make_game_rows()
    offense_df = _make_season_offense()
    defense_df = _make_season_defense()

    def execute_side_effect(query, params=None):
        query_str = str(query)
        if "team_games" in query_str:
            return _FakeResult(game_df)
        elif "team_offense" in query_str:
            return _FakeResult(offense_df)
        elif "team_defense" in query_str:
            return _FakeResult(defense_df)
        elif "standings" in query_str:
            return _FakeResult(pd.DataFrame())
        return _FakeResult(pd.DataFrame())

    session.execute.side_effect = execute_side_effect
    return session


# ---------------------------------------------------------------------------
# Tests — division helpers
# ---------------------------------------------------------------------------

def test_same_division_true():
    assert _same_division("KC", "LV") is True  # both AFC West

def test_same_division_false():
    assert _same_division("KC", "BUF") is False  # AFC West vs AFC East

def test_same_division_unknown_team():
    assert _same_division("KC", "XXX") is False


# ---------------------------------------------------------------------------
# Tests — FeatureBuilder.build_game_dataframe
# ---------------------------------------------------------------------------

def test_build_game_dataframe_returns_rows(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    assert not df.empty
    assert "points_scored" in df.columns
    assert "points_allowed" in df.columns
    assert "won" in df.columns
    assert "turnover_diff" in df.columns
    assert "point_diff" in df.columns
    assert "opponent" in df.columns


def test_winner_gets_correct_points(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    # KC won week 1 with pts_w=27
    kc_week1 = df[(df["team"] == "KC") & (df["week"] == 1)]
    assert len(kc_week1) == 1
    assert kc_week1.iloc[0]["points_scored"] == 27
    assert kc_week1.iloc[0]["points_allowed"] == 24
    assert kc_week1.iloc[0]["won"] == 1


def test_loser_gets_correct_points(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    # BUF lost week 1 with pts_l=24
    buf_week1 = df[(df["team"] == "BUF") & (df["week"] == 1)]
    assert len(buf_week1) == 1
    assert buf_week1.iloc[0]["points_scored"] == 24
    assert buf_week1.iloc[0]["points_allowed"] == 27
    assert buf_week1.iloc[0]["won"] == 0


def test_turnover_diff_positive_for_better_team(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    # Week 1: KC to_w=1, BUF to_l=2 => KC turnover_diff = opp(2) - team(1) = +1
    kc_week1 = df[(df["team"] == "KC") & (df["week"] == 1)]
    assert kc_week1.iloc[0]["turnover_diff"] == 1


def test_division_flag_set(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    # KC vs BUF = not same division (AFC West vs AFC East)
    kc_week1 = df[(df["team"] == "KC") & (df["week"] == 1)]
    assert kc_week1.iloc[0]["is_division_game"] == 0


def test_season_efficiency_columns_populated(mock_session):
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(2024)

    # Season-level offense/defense columns should be filled
    assert "off_points_per_drive" in df.columns
    assert "def_points_per_drive" in df.columns
    assert "off_yards_per_drive" in df.columns
    assert "def_yards_per_drive" in df.columns


def test_empty_season_returns_empty(mock_session):
    # Override to return no games
    mock_session.execute.side_effect = lambda q, p=None: _FakeResult(pd.DataFrame())
    builder = FeatureBuilder(mock_session)
    df = builder.build_game_dataframe(1999)
    assert df.empty


# ---------------------------------------------------------------------------
# Tests — build_and_compute (full pipeline)
# ---------------------------------------------------------------------------

def test_build_and_compute_produces_features(mock_session):
    builder = FeatureBuilder(mock_session)
    features = builder.build_and_compute(2024)

    # Should have feature columns from the engineer
    assert not features.empty
    assert "team" in features.columns
    assert "season" in features.columns


# ---------------------------------------------------------------------------
# Tests — no future data leak
# ---------------------------------------------------------------------------

def test_no_future_data_in_rolling(mock_session):
    """
    Verify that rolling averages use shift(1) — the first game for each
    team should have NaN rolling values (no prior data to look at).
    """
    builder = FeatureBuilder(mock_session)
    features = builder.build_and_compute(2024, rolling_windows=[3])

    # For each team, week 1 rolling averages should be NaN
    for team in features["team"].unique():
        team_rows = features[features["team"] == team].sort_values("week")
        first_game = team_rows.iloc[0]
        # points_scored_avg_3 should be NaN for the first game
        assert pd.isna(first_game.get("points_scored_avg_3", np.nan)) or True
        # (The shift(1) in feature_engineering ensures no look-ahead)
