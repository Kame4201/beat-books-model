"""
Advanced stats provider (AWS Next Gen Stats, PFF grades).

Phase 2 feature — provides the interface and null/stub implementations.
Live providers will be added when API access / licensing is resolved.

CRITICAL: This module only READS external data. It never creates or
modifies database tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TeamAdvancedStats:
    """Advanced stats for a team in a given season/week."""

    team: str
    season: int
    week: int

    # Next Gen Stats (NGS)
    cpoe: Optional[float] = None  # Completion % Over Expected
    ryoe: Optional[float] = None  # Rush Yards Over Expected
    avg_separation: Optional[float] = None  # WR avg separation (yards)
    avg_time_to_throw: Optional[float] = None  # seconds

    # PFF grades (0-100 scale)
    pff_offense_grade: Optional[float] = None
    pff_defense_grade: Optional[float] = None
    pff_pass_block_grade: Optional[float] = None
    pff_run_block_grade: Optional[float] = None

    def to_feature_dict(self) -> Dict[str, float]:
        """
        Return flat dict of features.

        Missing values are replaced with 0.0 (neutral) so downstream
        models can handle them without special-casing.
        """
        return {
            "adv_cpoe": self.cpoe if self.cpoe is not None else 0.0,
            "adv_ryoe": self.ryoe if self.ryoe is not None else 0.0,
            "adv_avg_separation": (
                self.avg_separation if self.avg_separation is not None else 0.0
            ),
            "adv_avg_time_to_throw": (
                self.avg_time_to_throw if self.avg_time_to_throw is not None else 0.0
            ),
            "adv_pff_offense_grade": (
                self.pff_offense_grade if self.pff_offense_grade is not None else 0.0
            ),
            "adv_pff_defense_grade": (
                self.pff_defense_grade if self.pff_defense_grade is not None else 0.0
            ),
            "adv_pff_pass_block_grade": (
                self.pff_pass_block_grade
                if self.pff_pass_block_grade is not None
                else 0.0
            ),
            "adv_pff_run_block_grade": (
                self.pff_run_block_grade
                if self.pff_run_block_grade is not None
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class AdvancedStatsProvider(ABC):
    """Abstract base for advanced-stats backends."""

    @abstractmethod
    def get_team_stats(self, team: str, season: int, week: int) -> TeamAdvancedStats:
        """Return advanced stats for a team in a given week."""

    def get_game_stats(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> Dict[str, TeamAdvancedStats]:
        return {
            "home": self.get_team_stats(home_team, season, week),
            "away": self.get_team_stats(away_team, season, week),
        }


class NullAdvancedStatsProvider(AdvancedStatsProvider):
    """
    Stub provider returning empty stats (all None → 0.0 in features).

    Use until NGS / PFF API access is configured.
    """

    def get_team_stats(self, team: str, season: int, week: int) -> TeamAdvancedStats:
        return TeamAdvancedStats(team=team, season=season, week=week)


class FixtureAdvancedStatsProvider(AdvancedStatsProvider):
    """
    Returns pre-loaded data from a dict. Useful for testing and
    backtesting with historical snapshots.
    """

    def __init__(self, data: Dict[str, TeamAdvancedStats]):
        """
        Args:
            data: mapping of ``"team-season-week"`` → TeamAdvancedStats
        """
        self._data = data

    @staticmethod
    def _key(team: str, season: int, week: int) -> str:
        return f"{team}-{season}-{week}"

    def get_team_stats(self, team: str, season: int, week: int) -> TeamAdvancedStats:
        key = self._key(team, season, week)
        if key in self._data:
            return self._data[key]
        return TeamAdvancedStats(team=team, season=season, week=week)


def get_advanced_stats_provider(
    fixture_data: Optional[Dict[str, TeamAdvancedStats]] = None,
) -> AdvancedStatsProvider:
    """
    Factory.

    - Pass fixture_data for testing / backtesting with snapshots
    - Default returns NullAdvancedStatsProvider
    """
    if fixture_data is not None:
        return FixtureAdvancedStatsProvider(fixture_data)
    return NullAdvancedStatsProvider()
