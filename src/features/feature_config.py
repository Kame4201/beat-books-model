"""
Feature configuration and metadata for the NFL prediction system.

Defines all feature names, versions, rolling windows, and metadata.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


# Feature version for tracking changes over time
FEATURE_VERSION = "v1.0"

# Rolling window sizes (in games)
ROLLING_WINDOWS = [3, 5, 10]


@dataclass
class FeatureMetadata:
    """Metadata for a feature set."""

    version: str
    creation_date: datetime
    feature_names: List[str]
    row_count: int
    season_range: tuple  # (start_season, end_season)
    description: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "version": self.version,
            "creation_date": self.creation_date.isoformat(),
            "feature_names": self.feature_names,
            "row_count": self.row_count,
            "season_range": self.season_range,
            "description": self.description,
            "additional_info": self.additional_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """Create metadata from dictionary."""
        data["creation_date"] = datetime.fromisoformat(data["creation_date"])
        return cls(**data)


# Feature name templates for rolling averages
ROLLING_FEATURE_TEMPLATES = {
    # Points
    "points_scored_avg_{window}": "Average points scored over last {window} games",
    "points_allowed_avg_{window}": "Average points allowed over last {window} games",

    # Yards per play
    "off_yards_per_play_avg_{window}": "Offensive yards per play over last {window} games",
    "def_yards_per_play_avg_{window}": "Defensive yards per play allowed over last {window} games",

    # Turnover differential
    "turnover_diff_avg_{window}": "Turnover differential over last {window} games",

    # Efficiency metrics
    "off_points_per_drive_avg_{window}": "Offensive points per drive over last {window} games",
    "def_points_per_drive_avg_{window}": "Defensive points per drive allowed over last {window} games",
    "off_yards_per_drive_avg_{window}": "Offensive yards per drive over last {window} games",
    "def_yards_per_drive_avg_{window}": "Defensive yards per drive allowed over last {window} games",

    # Red zone efficiency
    "off_red_zone_pct_avg_{window}": "Offensive red zone TD percentage over last {window} games",
    "def_red_zone_pct_avg_{window}": "Defensive red zone TD percentage allowed over last {window} games",

    # Third down efficiency
    "off_third_down_pct_avg_{window}": "Offensive third down conversion rate over last {window} games",
    "def_third_down_pct_avg_{window}": "Defensive third down conversion rate allowed over last {window} games",

    # Sack rate
    "sacks_given_avg_{window}": "Sacks given up per game over last {window} games",
    "sacks_taken_avg_{window}": "Sacks recorded per game over last {window} games",
}


# Situational features (non-rolling)
SITUATIONAL_FEATURES = {
    "home_win_pct": "Win percentage at home (season-to-date)",
    "away_win_pct": "Win percentage on the road (season-to-date)",
    "home_point_diff_avg": "Average point differential at home (season-to-date)",
    "away_point_diff_avg": "Average point differential on the road (season-to-date)",
    "rest_days": "Days of rest since last game",
    "strength_of_schedule": "Opponent cumulative win percentage (season-to-date)",
    "current_streak": "Current win/loss streak (positive=wins, negative=losses)",
    "is_division_game": "Binary indicator if game is within division",
    "had_bye_week": "Binary indicator if team had a bye in last 2 weeks",
}


# Derived/trend features
DERIVED_FEATURES = {
    "point_diff_trend_5": "Linear trend of point differential over last 5 games (slope)",
    "turnover_margin_trend_5": "Linear trend of turnover margin over last 5 games (slope)",
    "win_pct_last_5": "Win percentage over last 5 games",
}


def get_all_feature_names() -> List[str]:
    """
    Generate complete list of all feature names.

    Returns:
        List of all feature names that will be computed.
    """
    feature_names = []

    # Add rolling features for each window
    for template in ROLLING_FEATURE_TEMPLATES.keys():
        for window in ROLLING_WINDOWS:
            feature_name = template.replace("{window}", str(window))
            feature_names.append(feature_name)

    # Add situational features
    feature_names.extend(SITUATIONAL_FEATURES.keys())

    # Add derived features
    feature_names.extend(DERIVED_FEATURES.keys())

    return sorted(feature_names)


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions for all features.

    Returns:
        Dictionary mapping feature names to descriptions.
    """
    descriptions = {}

    # Add rolling feature descriptions
    for template, desc_template in ROLLING_FEATURE_TEMPLATES.items():
        for window in ROLLING_WINDOWS:
            feature_name = template.replace("{window}", str(window))
            description = desc_template.replace("{window}", str(window))
            descriptions[feature_name] = description

    # Add situational feature descriptions
    descriptions.update(SITUATIONAL_FEATURES)

    # Add derived feature descriptions
    descriptions.update(DERIVED_FEATURES)

    return descriptions


# Expected minimum feature count (for validation)
MIN_FEATURE_COUNT = 20
