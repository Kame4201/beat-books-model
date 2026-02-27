"""
Lightweight feature builder for inference time.

Given (home_team, away_team, season) and a list of required feature columns,
queries the database for each team's season stats and constructs a 1-row
DataFrame with the same columns used during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

TEAM_SEASON_STATS_QUERY = text("""
    SELECT
        tm, season,
        pf, pa, g,
        yds, ply, ypp, turnovers,
        sc_pct
    FROM team_offense
    WHERE season = :season AND LOWER(tm) = LOWER(:team)
""")

TEAM_DEFENSE_STATS_QUERY = text("""
    SELECT
        tm, season,
        pa AS def_pa, yds AS def_yds, ply AS def_ply,
        ypp AS def_ypp, turnovers AS takeaways,
        sc_pct AS def_sc_pct
    FROM team_defense
    WHERE season = :season AND LOWER(tm) = LOWER(:team)
""")

TEAM_STANDINGS_QUERY = text("""
    SELECT
        tm, w, l, win_pct, pf, pa, pd, mov, sos, srs
    FROM standings
    WHERE season = :season AND LOWER(tm) = LOWER(:team)
""")

TEAM_RECENT_GAMES_QUERY = text("""
    SELECT
        team_abbr AS team, season, week,
        winner, loser, pts_w, pts_l,
        yds_w, yds_l, to_w, to_l
    FROM team_games
    WHERE season = :season AND LOWER(team_abbr) = LOWER(:team)
    ORDER BY week DESC
    LIMIT :limit
""")


def build_inference_features(
    session: Session,
    home_team: str,
    away_team: str,
    season: int,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a 1-row feature DataFrame for a matchup prediction.

    Queries team_offense, team_defense, standings, and recent games
    to approximate the rolling features used during training.

    Args:
        session: SQLAlchemy read-only session
        home_team: Home team abbreviation (e.g., "KC")
        away_team: Away team abbreviation (e.g., "SF")
        season: NFL season year
        feature_names: List of feature column names to produce

    Returns:
        1-row DataFrame with the requested feature columns
    """
    home_stats = _get_team_stats(session, home_team, season)
    away_stats = _get_team_stats(session, away_team, season)

    row: dict[str, float] = {}
    for col in feature_names:
        row[col] = _compute_feature(col, home_stats, away_stats)

    return pd.DataFrame([row])


def build_inference_features_synthetic(
    home_team: str,
    away_team: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build inference features without a database (fallback for testing).
    Uses neutral default values.
    """
    row: dict[str, float] = {}
    for col in feature_names:
        if "points_scored" in col:
            row[col] = 22.0
        elif "points_allowed" in col:
            row[col] = 22.0
        elif "yards_per_play" in col:
            row[col] = 5.5
        elif "turnover_diff" in col:
            row[col] = 0.0
        elif "streak" in col:
            row[col] = 0.0
        elif "win_pct" in col:
            row[col] = 0.5
        elif "division" in col:
            row[col] = 0.0
        else:
            row[col] = 0.0
    return pd.DataFrame([row])


def _get_team_stats(session: Session, team: str, season: int) -> dict:
    """Fetch aggregated stats for a single team."""
    stats: dict[str, object] = {"team": team}

    # Offense
    result = session.execute(TEAM_SEASON_STATS_QUERY, {"season": season, "team": team})
    row = result.fetchone()
    if row:
        mapping = dict(row._mapping)
        games = max(mapping.get("g", 1), 1)
        stats["ppg"] = mapping.get("pf", 0) / games
        stats["ypp"] = mapping.get("ypp", 5.5)
        stats["to_per_game"] = mapping.get("turnovers", 0) / games
        stats["sc_pct"] = mapping.get("sc_pct", 0)

    # Defense
    result = session.execute(TEAM_DEFENSE_STATS_QUERY, {"season": season, "team": team})
    row = result.fetchone()
    if row:
        mapping = dict(row._mapping)
        stats["def_ppg"] = mapping.get("def_pa", 0) / max(games, 1)
        stats["def_ypp"] = mapping.get("def_ypp", 5.5)
        stats["takeaways_per_game"] = mapping.get("takeaways", 0) / max(games, 1)

    # Standings
    result = session.execute(TEAM_STANDINGS_QUERY, {"season": season, "team": team})
    row = result.fetchone()
    if row:
        mapping = dict(row._mapping)
        stats["win_pct"] = mapping.get("win_pct", 0.5)
        stats["mov"] = mapping.get("mov", 0)
        stats["srs"] = mapping.get("srs", 0)

    # Recent games for streak
    result = session.execute(
        TEAM_RECENT_GAMES_QUERY, {"season": season, "team": team, "limit": 5}
    )
    recent = result.fetchall()
    if recent:
        streak = 0
        wins_last_5 = 0
        pts_scored = []
        pts_allowed = []
        for g in recent:
            m = dict(g._mapping)
            won = m["winner"].upper() == team.upper() if m.get("winner") else False
            if won:
                wins_last_5 += 1
                pts_scored.append(m.get("pts_w", 0))
                pts_allowed.append(m.get("pts_l", 0))
            else:
                pts_scored.append(m.get("pts_l", 0))
                pts_allowed.append(m.get("pts_w", 0))
        # Streak: count consecutive results from most recent
        for g in recent:
            m = dict(g._mapping)
            won = m["winner"].upper() == team.upper() if m.get("winner") else False
            if streak == 0:
                streak = 1 if won else -1
            elif won and streak > 0:
                streak += 1
            elif not won and streak < 0:
                streak -= 1
            else:
                break
        stats["current_streak"] = float(streak)
        stats["win_pct_last_5"] = wins_last_5 / max(len(recent), 1)
        stats["pts_scored_avg"] = np.mean(pts_scored) if pts_scored else 22.0
        stats["pts_allowed_avg"] = np.mean(pts_allowed) if pts_allowed else 22.0

    # Defaults
    stats.setdefault("ppg", 22.0)
    stats.setdefault("def_ppg", 22.0)
    stats.setdefault("ypp", 5.5)
    stats.setdefault("def_ypp", 5.5)
    stats.setdefault("to_per_game", 1.0)
    stats.setdefault("takeaways_per_game", 1.0)
    stats.setdefault("win_pct", 0.5)
    stats.setdefault("current_streak", 0.0)
    stats.setdefault("win_pct_last_5", 0.5)
    stats.setdefault("pts_scored_avg", 22.0)
    stats.setdefault("pts_allowed_avg", 22.0)
    stats.setdefault("mov", 0.0)
    stats.setdefault("srs", 0.0)

    return stats


def _compute_feature(col: str, home: dict, away: dict) -> float:
    """Map a training feature column name to a value from team stats."""
    # Rolling-average features: use home team's stats
    if col == "points_scored_avg_3" or col == "points_scored_avg_5":
        return home.get("pts_scored_avg", home.get("ppg", 22.0))
    if col == "points_allowed_avg_3" or col == "points_allowed_avg_5":
        return home.get("pts_allowed_avg", home.get("def_ppg", 22.0))
    if col == "off_yards_per_play_avg_3":
        return home.get("ypp", 5.5)
    if col == "def_yards_per_play_avg_3":
        return home.get("def_ypp", 5.5)
    if col == "turnover_diff_avg_3":
        return home.get("takeaways_per_game", 1.0) - home.get("to_per_game", 1.0)
    if col == "current_streak":
        return home.get("current_streak", 0.0)
    if col == "win_pct_last_5":
        return home.get("win_pct_last_5", 0.5)
    if col == "is_division_game":
        return 0.0  # would need division lookup; default to non-division
    # Fallback
    return 0.0
