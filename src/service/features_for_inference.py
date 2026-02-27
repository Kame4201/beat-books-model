"""
Build a single-row feature DataFrame for inference.

Loads season stats for (home_team, away_team) from the DB,
computes the same diff features used in training.
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Must match the features used in scripts/train_baseline.py
FEATURE_COLUMNS = [
    "pf",
    "pa",
    "yds_off",
    "yds_def",
    "to_off",
    "to_def",
    "mov",
    "srs",
    "osrs",
    "dsrs",
    "win_pct",
]


def _load_team_stats(engine: Engine, team: str, season: int) -> Optional[dict]:
    """Load aggregated stats for one team in one season."""
    query = text("""
        SELECT
            s.tm,
            s.pf,
            s.pa,
            s.mov,
            s.srs,
            s.osrs,
            s.dsrs,
            s.win_pct,
            COALESCE(o.yds, 0) AS yds_off,
            COALESCE(o.turnovers, 0) AS to_off,
            COALESCE(d.yds, 0) AS yds_def,
            COALESCE(d.turnovers, 0) AS to_def
        FROM standings s
        LEFT JOIN team_offense o ON s.tm = o.tm AND s.season = o.season
        LEFT JOIN team_defense d ON s.tm = d.tm AND s.season = d.season
        WHERE s.season = :season AND s.tm = :team
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"season": season, "team": team})
        row = result.mappings().fetchone()

    if row is None:
        return None
    return dict(row)


def build_inference_features(
    engine: Engine,
    home_team: str,
    away_team: str,
    season: int,
    expected_features: list[str],
) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with diff features (home - away) for prediction.

    Args:
        engine: SQLAlchemy engine
        home_team: Home team name (as stored in standings.tm)
        away_team: Away team name
        season: Season year
        expected_features: Feature column names from the trained model

    Returns:
        DataFrame with one row, columns matching expected_features

    Raises:
        ValueError: If team stats not found
    """
    home_stats = _load_team_stats(engine, home_team, season)
    away_stats = _load_team_stats(engine, away_team, season)

    if home_stats is None:
        raise ValueError(f"No stats found for {home_team} in {season}")
    if away_stats is None:
        raise ValueError(f"No stats found for {away_team} in {season}")

    row = {}
    for col in FEATURE_COLUMNS:
        h_val = float(home_stats.get(col, 0) or 0)
        a_val = float(away_stats.get(col, 0) or 0)
        row[f"diff_{col}"] = h_val - a_val

    df = pd.DataFrame([row])

    # Reorder to match model's expected feature order, filling missing with 0
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0.0

    return df[expected_features]
