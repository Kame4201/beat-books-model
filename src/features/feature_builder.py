"""
FeatureBuilder: connects the real NFL database to the FeatureEngineer.

Queries game-level rows from ``team_games`` and season-level aggregates
from ``team_offense`` / ``team_defense`` / ``standings``, then transforms
them into the DataFrame schema that FeatureEngineer expects.

CRITICAL: This module uses READ-ONLY database access. It NEVER creates,
alters, or drops tables.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore

# ---------------------------------------------------------------------------
# SQL Queries — all READ-ONLY
# ---------------------------------------------------------------------------

GAME_LEVEL_QUERY = text("""
    SELECT
        tg.id          AS game_id,
        tg.team_abbr   AS team,
        tg.season,
        tg.week,
        tg.game_date,
        tg.winner,
        tg.loser,
        tg.pts_w,
        tg.pts_l,
        tg.yds_w,
        tg.yds_l,
        tg.to_w,
        tg.to_l
    FROM team_games tg
    WHERE tg.season = :season
    ORDER BY tg.team_abbr, tg.game_date, tg.week
""")

GAME_LEVEL_WEEKS_QUERY = text("""
    SELECT
        tg.id          AS game_id,
        tg.team_abbr   AS team,
        tg.season,
        tg.week,
        tg.game_date,
        tg.winner,
        tg.loser,
        tg.pts_w,
        tg.pts_l,
        tg.yds_w,
        tg.yds_l,
        tg.to_w,
        tg.to_l
    FROM team_games tg
    WHERE tg.season = :season AND tg.week IN :weeks
    ORDER BY tg.team_abbr, tg.game_date, tg.week
""")

SEASON_OFFENSE_QUERY = text("""
    SELECT
        tm, season, g AS games_played,
        pf, yds, ply, ypp, turnovers,
        sc_pct, to_pct, opea
    FROM team_offense
    WHERE season = :season
""")

SEASON_DEFENSE_QUERY = text("""
    SELECT
        tm, season, g AS games_played,
        pa, yds AS yds_allowed, ply AS ply_allowed,
        ypp AS ypp_allowed, turnovers AS takeaways,
        sc_pct AS sc_pct_allowed, to_pct AS to_pct_forced,
        depa
    FROM team_defense
    WHERE season = :season
""")

STANDINGS_QUERY = text("""
    SELECT
        tm, season, w, l, t, win_pct, pf, pa, pd,
        mov, sos, srs, osrs, dsrs
    FROM standings
    WHERE season = :season
""")

# NFL divisions for is_division_game derivation
NFL_DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SEA", "SF"],
}

# Build reverse lookup: team abbreviation → division name
_TEAM_TO_DIVISION: dict[str, str] = {}
for _div, _teams in NFL_DIVISIONS.items():
    for _t in _teams:
        _TEAM_TO_DIVISION[_t] = _div


def _same_division(team_a: str, team_b: str) -> bool:
    """Return True if both teams are in the same division."""
    div_a = _TEAM_TO_DIVISION.get(team_a)
    div_b = _TEAM_TO_DIVISION.get(team_b)
    return div_a is not None and div_a == div_b


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """
    Queries the shared PostgreSQL database and produces a DataFrame suitable
    for ``FeatureEngineer.compute_features()``.

    Uses READ-ONLY access via ``db_reader.get_read_session()``.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_game_dataframe(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Build a game-level DataFrame from the database for one season.

        The output DataFrame has columns expected by ``FeatureEngineer``:
        game_id, team, season, week, game_date,
        points_scored, points_allowed, off_yards_per_play, def_yards_per_play,
        turnover_diff, is_home, is_away, won, point_diff,
        is_division_game, opponent, …

        Columns that cannot be derived from game-level data (drive-based
        efficiency metrics) are filled from season-level averages so the
        feature engineer still has valid numbers to work with.

        Args:
            season: NFL season year
            weeks: Optional list of specific weeks

        Returns:
            DataFrame ready for FeatureEngineer
        """
        raw = self._query_games(season, weeks)
        if raw.empty:
            return pd.DataFrame()

        df = self._derive_game_columns(raw)
        df = self._enrich_with_season_stats(df, season)
        df = self._add_division_flags(df)
        return df

    def build_and_compute(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Convenience: build game data, then run feature engineering.

        Returns:
            Computed features DataFrame (empty if no games found)
        """
        game_df = self.build_game_dataframe(season, weeks)
        if game_df.empty:
            return pd.DataFrame()

        engineer = FeatureEngineer(rolling_windows=rolling_windows)
        # Override the placeholder _load_game_data to return our DB data
        engineer._load_game_data = lambda s, w=None: game_df  # type: ignore[assignment]
        return engineer.compute_features(season, weeks)

    def build_compute_and_store(
        self,
        season: int,
        weeks: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        description: str = "",
    ) -> Optional[pd.DataFrame]:
        """
        Full pipeline: query DB → compute features → save to feature store.

        Returns:
            Computed features DataFrame, or None if no data
        """
        features = self.build_and_compute(season, weeks, rolling_windows)
        if features.empty:
            return None

        store = FeatureStore()
        store.save(features, description=description or f"Season {season}")
        return features

    # ------------------------------------------------------------------
    # Private — DB queries
    # ------------------------------------------------------------------

    def _query_games(
        self, season: int, weeks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Execute the game-level query and return a raw DataFrame."""
        if weeks:
            result = self.session.execute(
                GAME_LEVEL_WEEKS_QUERY,
                {"season": season, "weeks": tuple(weeks)},
            )
        else:
            result = self.session.execute(GAME_LEVEL_QUERY, {"season": season})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=result.keys())

    def _query_season_offense(self, season: int) -> pd.DataFrame:
        result = self.session.execute(SEASON_OFFENSE_QUERY, {"season": season})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=result.keys())

    def _query_season_defense(self, season: int) -> pd.DataFrame:
        result = self.session.execute(SEASON_DEFENSE_QUERY, {"season": season})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=result.keys())

    def _query_standings(self, season: int) -> pd.DataFrame:
        result = self.session.execute(STANDINGS_QUERY, {"season": season})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=result.keys())

    # ------------------------------------------------------------------
    # Private — transformations
    # ------------------------------------------------------------------

    def _derive_game_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Derive won, points_scored, points_allowed, etc. from the
        winner/loser structure.
        """
        df = raw.copy()

        # Determine if this team's row is the winner
        is_winner = df["team"] == df["winner"]

        # Points
        df["points_scored"] = np.where(is_winner, df["pts_w"], df["pts_l"])
        df["points_allowed"] = np.where(is_winner, df["pts_l"], df["pts_w"])

        # Yards
        df["off_total_yards"] = np.where(is_winner, df["yds_w"], df["yds_l"])
        df["def_total_yards"] = np.where(is_winner, df["yds_l"], df["yds_w"])

        # Turnovers (winner turnovers vs loser turnovers)
        team_turnovers = np.where(is_winner, df["to_w"], df["to_l"])
        opp_turnovers = np.where(is_winner, df["to_l"], df["to_w"])
        df["turnover_diff"] = opp_turnovers - team_turnovers

        # Win/loss
        df["won"] = is_winner.astype(int)
        df["point_diff"] = df["points_scored"] - df["points_allowed"]

        # Opponent
        df["opponent"] = np.where(is_winner, df["loser"], df["winner"])

        # Home/away approximation:
        # Pro-Football-Reference uses "@" prefix for away teams in game logs.
        # The team_games table's team_abbr IS the team — we heuristically
        # mark the winner as home when no explicit column exists, but that's
        # wrong ~42% of the time. For now, set all to 0 (unknown) since the
        # DB doesn't store home/away. This is a known limitation (see #22).
        df["is_home"] = 0
        df["is_away"] = 0

        # Use a conservative default for yards per play (~65 plays/game)
        estimated_plays = 65.0
        df["off_yards_per_play"] = df["off_total_yards"] / estimated_plays
        df["def_yards_per_play"] = df["def_total_yards"] / estimated_plays

        return df

    def _enrich_with_season_stats(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        Fill efficiency columns that aren't available at game level
        using season-level averages from team_offense / team_defense.

        These are constants per team per season — they won't vary game
        to game — but they give the feature engineer valid numbers for
        rolling averages on efficiency metrics (which will converge to
        the constant, essentially weighting recent seasons more).
        """
        offense = self._query_season_offense(season)
        defense = self._query_season_defense(season)

        # Map season-level columns to the feature engineer's expected names
        if not offense.empty:
            off_map = offense.set_index("tm")
            games_played = off_map["games_played"].replace(0, 1)  # avoid div/0
            df["off_points_per_drive"] = (
                df["team"].map(off_map["sc_pct"]).fillna(0).astype(float)
            )
            df["off_yards_per_drive"] = (
                df["team"]
                .map((off_map["yds"] / games_played).round(1))
                .fillna(0)
                .astype(float)
            )
            df["off_red_zone_pct"] = (
                df["team"].map(off_map["sc_pct"]).fillna(0).astype(float)
            )
            df["off_third_down_pct"] = 0.0  # Not available in schema
            df["sacks_given"] = 0.0  # Not available at team level

        if not defense.empty:
            def_map = defense.set_index("tm")
            games_played = def_map["games_played"].replace(0, 1)
            df["def_points_per_drive"] = (
                df["team"].map(def_map["sc_pct_allowed"]).fillna(0).astype(float)
            )
            df["def_yards_per_drive"] = (
                df["team"]
                .map((def_map["yds_allowed"] / games_played).round(1))
                .fillna(0)
                .astype(float)
            )
            df["def_red_zone_pct"] = (
                df["team"].map(def_map["sc_pct_allowed"]).fillna(0).astype(float)
            )
            df["def_third_down_pct"] = 0.0
            df["sacks_taken"] = 0.0

        # Defaults for any columns still missing
        for col in [
            "off_points_per_drive",
            "def_points_per_drive",
            "off_yards_per_drive",
            "def_yards_per_drive",
            "off_red_zone_pct",
            "def_red_zone_pct",
            "off_third_down_pct",
            "def_third_down_pct",
            "sacks_given",
            "sacks_taken",
        ]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _add_division_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark games as divisional matchups."""
        if "opponent" in df.columns:
            df["is_division_game"] = df.apply(
                lambda r: int(_same_division(str(r["team"]), str(r["opponent"]))),
                axis=1,
            )
        else:
            df["is_division_game"] = 0
        return df
