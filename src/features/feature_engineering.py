"""
Core feature engineering logic for NFL game predictions.

CRITICAL: All features use ONLY historical data (no look-ahead bias).
Rolling windows compute over games BEFORE the current game.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.features.feature_config import ROLLING_WINDOWS


class FeatureEngineer:
    """
    Computes predictive features from raw NFL statistics.

    All computations maintain strict chronological ordering to prevent look-ahead bias.
    """

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        """
        Initialize feature engineer.

        Args:
            rolling_windows: List of game window sizes for rolling averages
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

    def compute_features(
        self, season: int, weeks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute features for a season.

        Args:
            season: NFL season year
            weeks: Specific weeks to compute (None = all weeks)

        Returns:
            DataFrame with computed features per team per game
        """
        # Load raw game data
        games_df = self._load_game_data(season, weeks)

        if games_df.empty:
            print(f"No games found for season {season}")
            return pd.DataFrame()

        # Sort by date to ensure chronological processing
        games_df = games_df.sort_values(["team", "game_date"]).reset_index(drop=True)

        # Compute features by category
        features_list = []

        for team in games_df["team"].unique():
            team_games = games_df[games_df["team"] == team].copy()
            team_features = self._compute_team_features(team_games)
            features_list.append(team_features)

        # Combine all teams
        all_features = pd.concat(features_list, ignore_index=True)

        return all_features

    def _load_game_data(
        self, season: int, weeks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load game data from database.

        Args:
            season: Season year
            weeks: Specific weeks to load

        Returns:
            DataFrame with game-level data
        """
        # NOTE: This is a placeholder. In production, would query actual DB tables.
        # For now, return empty DataFrame since DB may not be available in CI.

        # Example query structure (commented out):
        # session = next(get_read_session())
        # query = """
        #     SELECT
        #         g.game_id,
        #         g.season,
        #         g.week,
        #         g.game_date,
        #         g.home_team,
        #         g.away_team,
        #         g.home_score,
        #         g.away_score,
        #         ...
        #     FROM games g
        #     WHERE g.season = :season
        # """
        # if weeks:
        #     query += " AND g.week IN :weeks"
        # result = session.execute(query, {"season": season, "weeks": tuple(weeks)})

        # Placeholder return
        return pd.DataFrame()

    def _compute_team_features(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a single team's games.

        Args:
            team_games: DataFrame of games for one team (chronologically sorted)

        Returns:
            DataFrame with computed features
        """
        features = team_games[["game_id", "team", "season", "week", "game_date"]].copy()

        # Compute rolling average features
        features = self._add_rolling_averages(features, team_games)

        # Compute efficiency metrics
        features = self._add_efficiency_metrics(features, team_games)

        # Compute situational features
        features = self._add_situational_features(features, team_games)

        # Compute derived/trend features
        features = self._add_derived_features(features, team_games)

        return features

    def _add_rolling_averages(
        self, features: pd.DataFrame, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add rolling average features.

        CRITICAL: Rolling windows use only PRIOR games (not including current game).
        """
        # Points scored and allowed
        for window in self.rolling_windows:
            # Points scored
            features[f"points_scored_avg_{window}"] = (
                team_games["points_scored"]
                .shift(1)  # Exclude current game
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Points allowed
            features[f"points_allowed_avg_{window}"] = (
                team_games["points_allowed"]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Offensive yards per play
            features[f"off_yards_per_play_avg_{window}"] = (
                team_games["off_yards_per_play"]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Defensive yards per play allowed
            features[f"def_yards_per_play_avg_{window}"] = (
                team_games["def_yards_per_play"]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Turnover differential
            features[f"turnover_diff_avg_{window}"] = (
                team_games["turnover_diff"]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

        return features

    def _add_efficiency_metrics(
        self, features: pd.DataFrame, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add efficiency metric features (points/yards per drive, red zone, third down).
        """
        for window in self.rolling_windows:
            # Offensive points per drive
            features[f"off_points_per_drive_avg_{window}"] = (
                team_games.get("off_points_per_drive", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Defensive points per drive allowed
            features[f"def_points_per_drive_avg_{window}"] = (
                team_games.get("def_points_per_drive", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Offensive yards per drive
            features[f"off_yards_per_drive_avg_{window}"] = (
                team_games.get("off_yards_per_drive", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Defensive yards per drive allowed
            features[f"def_yards_per_drive_avg_{window}"] = (
                team_games.get("def_yards_per_drive", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Red zone efficiency (offense)
            features[f"off_red_zone_pct_avg_{window}"] = (
                team_games.get("off_red_zone_pct", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Red zone efficiency (defense)
            features[f"def_red_zone_pct_avg_{window}"] = (
                team_games.get("def_red_zone_pct", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Third down conversion rate (offense)
            features[f"off_third_down_pct_avg_{window}"] = (
                team_games.get("off_third_down_pct", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Third down conversion rate (defense)
            features[f"def_third_down_pct_avg_{window}"] = (
                team_games.get("def_third_down_pct", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Sacks given up
            features[f"sacks_given_avg_{window}"] = (
                team_games.get("sacks_given", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Sacks recorded
            features[f"sacks_taken_avg_{window}"] = (
                team_games.get("sacks_taken", pd.Series(dtype=float))
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

        return features

    def _add_situational_features(
        self, features: pd.DataFrame, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add situational features (home/away, rest, schedule strength, streaks).
        """
        # Home/away splits (season-to-date, excluding current game)
        features["home_win_pct"] = self._compute_expanding_mean(
            team_games, "is_home", "won", shift=1
        )
        features["away_win_pct"] = self._compute_expanding_mean(
            team_games, "is_away", "won", shift=1
        )

        # Point differential splits
        features["home_point_diff_avg"] = self._compute_expanding_mean(
            team_games, "is_home", "point_diff", shift=1
        )
        features["away_point_diff_avg"] = self._compute_expanding_mean(
            team_games, "is_away", "point_diff", shift=1
        )

        # Rest days (days since last game)
        features["rest_days"] = team_games["game_date"].diff().dt.days.fillna(7)

        # Strength of schedule (opponent win pct, season-to-date)
        features["strength_of_schedule"] = self._compute_strength_of_schedule(
            team_games
        )

        # Current win/loss streak
        features["current_streak"] = self._compute_streak(team_games)

        # Division game indicator
        features["is_division_game"] = team_games.get("is_division_game", 0)

        # Bye week indicator (had bye in last 2 weeks)
        features["had_bye_week"] = (features["rest_days"] >= 14).astype(int)

        return features

    def _add_derived_features(
        self, features: pd.DataFrame, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add derived/trend features.
        """
        # Point differential trend (linear regression slope over last 5 games)
        features["point_diff_trend_5"] = (
            team_games.get("point_diff", pd.Series(dtype=float))
            .shift(1)
            .rolling(window=5, min_periods=3)
            .apply(self._compute_trend, raw=True)
        )

        # Turnover margin trend
        features["turnover_margin_trend_5"] = (
            team_games.get("turnover_diff", pd.Series(dtype=float))
            .shift(1)
            .rolling(window=5, min_periods=3)
            .apply(self._compute_trend, raw=True)
        )

        # Win percentage over last 5 games
        features["win_pct_last_5"] = (
            team_games.get("won", pd.Series(dtype=float))
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )

        return features

    @staticmethod
    def _compute_expanding_mean(
        df: pd.DataFrame, condition_col: str, value_col: str, shift: int = 0
    ) -> pd.Series:
        """
        Compute expanding mean for rows matching a condition.

        Args:
            df: DataFrame
            condition_col: Column to filter on (must be 1/True)
            value_col: Column to compute mean of
            shift: Number of rows to shift (0 = include current, 1 = exclude current)

        Returns:
            Series with expanding mean
        """
        if condition_col not in df.columns or value_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        # Mask for condition
        mask = df[condition_col] == 1

        # Compute expanding mean only for matching rows
        result = pd.Series(np.nan, index=df.index)

        if shift > 0:
            # Shift values to exclude current game
            values = df[value_col].shift(shift)
        else:
            values = df[value_col]

        # Expanding mean where condition is true
        expanding_sum = (values * mask).cumsum()
        expanding_count = mask.cumsum()

        result = expanding_sum / expanding_count.replace(0, np.nan)

        return result

    @staticmethod
    def _compute_strength_of_schedule(df: pd.DataFrame) -> pd.Series:
        """
        Compute strength of schedule (opponent win percentage).

        NOTE: This would require opponent game history from DB.
        Placeholder implementation returns NaN.
        """
        return pd.Series(np.nan, index=df.index)

    @staticmethod
    def _compute_streak(df: pd.DataFrame) -> pd.Series:
        """
        Compute current win/loss streak.

        Positive values = wins, negative values = losses.
        """
        if "won" not in df.columns:
            return pd.Series(0, index=df.index)

        # Shift to exclude current game
        won = df["won"].shift(1)

        streak = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if pd.isna(won.iloc[i]):
                streak.iloc[i] = 0
            elif won.iloc[i] == won.iloc[i - 1]:
                # Continue streak
                if won.iloc[i] == 1:
                    streak.iloc[i] = max(1, streak.iloc[i - 1] + 1)
                else:
                    streak.iloc[i] = min(-1, streak.iloc[i - 1] - 1)
            else:
                # Streak broken
                if won.iloc[i] == 1:
                    streak.iloc[i] = 1
                else:
                    streak.iloc[i] = -1

        return streak

    @staticmethod
    def _compute_trend(window: np.ndarray) -> float:
        """
        Compute linear trend (slope) for a window of values.

        Args:
            window: Array of values

        Returns:
            Slope of linear regression
        """
        if len(window) < 2:
            return np.nan

        # Remove NaN
        valid = ~np.isnan(window)
        if valid.sum() < 2:
            return np.nan

        x = np.arange(len(window))[valid]
        y = window[valid]

        # Simple linear regression: slope = cov(x,y) / var(x)
        if len(x) < 2:
            return np.nan

        slope = np.polyfit(x, y, 1)[0]
        return slope
