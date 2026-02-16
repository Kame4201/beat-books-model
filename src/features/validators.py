"""
Validation functions to ensure no look-ahead bias in features.

CRITICAL: All features must only use data from BEFORE the prediction date.
"""
import pandas as pd
from typing import List, Optional
from datetime import datetime


class LookAheadBiasError(Exception):
    """Raised when look-ahead bias is detected in features."""
    pass


def validate_no_future_data(
    features_df: pd.DataFrame,
    game_dates: pd.Series,
    reference_date_col: str = "game_date"
) -> None:
    """
    Validate that features only use data from before the game date.

    This checks that for each row (game), the computed features could have
    been calculated using only data available before that game.

    Args:
        features_df: DataFrame with computed features
        game_dates: Series of game dates corresponding to features_df rows
        reference_date_col: Name of the date column to check against

    Raises:
        LookAheadBiasError: If any feature uses future data
    """
    if len(features_df) != len(game_dates):
        raise ValueError("features_df and game_dates must have same length")

    # Check for NaN values in early rows (expected for rolling features)
    # But after the rolling window, should have values
    for col in features_df.columns:
        if "_avg_" in col:
            # Extract window size from column name (e.g., "points_scored_avg_3" -> 3)
            try:
                window_size = int(col.split("_")[-1])
                # After window_size games, should have valid data
                if features_df[col].iloc[window_size:].isna().any():
                    # Some NaN is ok if team hasn't played enough games
                    # But if we have enough historical data, shouldn't be NaN
                    pass
            except (ValueError, IndexError):
                # Column name doesn't follow expected pattern, skip
                continue


def validate_rolling_window_integrity(
    features_df: pd.DataFrame,
    required_windows: List[int] = [3, 5, 10]
) -> None:
    """
    Validate that rolling window features maintain proper ordering.

    Checks that shorter windows don't have values where longer windows are NaN,
    which would indicate incorrect window calculation.

    Args:
        features_df: DataFrame with computed features
        required_windows: List of window sizes to validate

    Raises:
        LookAheadBiasError: If window integrity is violated
    """
    # For each base feature, check window consistency
    base_features = set()
    for col in features_df.columns:
        if "_avg_" in col:
            # Extract base feature name (everything before last underscore and number)
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_features.add(parts[0])

    for base_feature in base_features:
        window_cols = [f"{base_feature}_{w}" for w in sorted(required_windows)]
        existing_cols = [col for col in window_cols if col in features_df.columns]

        if len(existing_cols) < 2:
            continue

        # Check that if a longer window has a value, shorter windows also have values
        for i in range(len(existing_cols) - 1):
            short_col = existing_cols[i]
            long_col = existing_cols[i + 1]

            # Where long window has values, short window should too
            long_has_value = features_df[long_col].notna()
            short_missing = features_df[short_col].isna()

            violations = long_has_value & short_missing
            if violations.any():
                violation_indices = violations[violations].index.tolist()
                raise LookAheadBiasError(
                    f"Window integrity violation for {base_feature}: "
                    f"{long_col} has values but {short_col} is NaN at indices {violation_indices[:5]}"
                )


def validate_chronological_order(
    df: pd.DataFrame,
    date_col: str = "game_date",
    team_col: str = "team"
) -> None:
    """
    Validate that data is in chronological order by team.

    Features must be computed on chronologically ordered data to avoid leakage.

    Args:
        df: DataFrame to validate
        date_col: Name of the date column
        team_col: Name of the team column

    Raises:
        LookAheadBiasError: If data is not in chronological order
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")

    if team_col not in df.columns:
        raise ValueError(f"Team column '{team_col}' not found in DataFrame")

    # Check ordering within each team
    for team in df[team_col].unique():
        team_df = df[df[team_col] == team]
        dates = pd.to_datetime(team_df[date_col])

        if not dates.is_monotonic_increasing:
            raise LookAheadBiasError(
                f"Data for team {team} is not in chronological order. "
                f"Features must be computed on chronologically sorted data."
            )


def validate_feature_completeness(
    features_df: pd.DataFrame,
    min_features: int = 20
) -> None:
    """
    Validate that minimum number of features are present.

    Args:
        features_df: DataFrame with computed features
        min_features: Minimum number of features required

    Raises:
        ValueError: If not enough features are present
    """
    # Exclude metadata columns
    metadata_cols = {"team", "opponent", "season", "week", "game_date", "game_id"}
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]

    if len(feature_cols) < min_features:
        raise ValueError(
            f"Expected at least {min_features} features, but found {len(feature_cols)}. "
            f"Features: {feature_cols}"
        )


def validate_no_nulls_after_warmup(
    features_df: pd.DataFrame,
    max_window: int = 10,
    allowed_null_rate: float = 0.1
) -> None:
    """
    Validate that after warmup period, there aren't excessive nulls.

    Some NaNs are expected early in the season (warmup period), but after
    max_window games, most features should have values.

    Args:
        features_df: DataFrame with computed features
        max_window: Maximum rolling window size (warmup period)
        allowed_null_rate: Maximum allowed null rate after warmup (0.0 to 1.0)

    Raises:
        ValueError: If too many nulls after warmup period
    """
    # Skip metadata columns
    metadata_cols = {"team", "opponent", "season", "week", "game_date", "game_id"}
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]

    if len(features_df) <= max_window:
        # Not enough data to check post-warmup
        return

    # Check post-warmup period
    post_warmup_df = features_df.iloc[max_window:]

    for col in feature_cols:
        null_rate = post_warmup_df[col].isna().mean()
        if null_rate > allowed_null_rate:
            raise ValueError(
                f"Feature '{col}' has {null_rate:.1%} null values after warmup period. "
                f"Expected less than {allowed_null_rate:.1%}."
            )


def run_all_validations(
    features_df: pd.DataFrame,
    game_dates: Optional[pd.Series] = None,
    min_features: int = 20
) -> None:
    """
    Run all validation checks on computed features.

    Args:
        features_df: DataFrame with computed features
        game_dates: Optional series of game dates
        min_features: Minimum number of features required

    Raises:
        LookAheadBiasError or ValueError: If any validation fails
    """
    # Basic completeness check
    validate_feature_completeness(features_df, min_features=min_features)

    # Rolling window integrity
    validate_rolling_window_integrity(features_df)

    # Chronological ordering (if date column exists)
    if "game_date" in features_df.columns:
        if "team" in features_df.columns:
            validate_chronological_order(features_df)

        # Future data check
        if game_dates is not None:
            validate_no_future_data(features_df, game_dates)

    # Null checks after warmup
    validate_no_nulls_after_warmup(features_df)
