"""
Injury and roster data provider for NFL game predictions.

Provides features reflecting player availability and its impact on team
strength.  Uses positional weighting so that a QB absence counts far more
than a backup punter being out.

CRITICAL: This module only READS external/cached data.  It never creates
or modifies database tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class InjuryStatus(str, Enum):
    """NFL injury designations."""

    ACTIVE = "active"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    IR = "injured_reserve"
    PUP = "pup"


class Position(str, Enum):
    """Positional categories for weighting."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    OL = "OL"
    DL = "DL"
    LB = "LB"
    DB = "DB"
    K = "K"
    P = "P"
    OTHER = "OTHER"


# Positional impact weights — QB absence is catastrophic; K/P is minimal
POSITION_WEIGHTS: Dict[Position, float] = {
    Position.QB: 1.00,
    Position.RB: 0.40,
    Position.WR: 0.35,
    Position.TE: 0.25,
    Position.OL: 0.30,
    Position.DL: 0.30,
    Position.LB: 0.30,
    Position.DB: 0.30,
    Position.K: 0.05,
    Position.P: 0.05,
    Position.OTHER: 0.10,
}

# Probability a player actually misses the game based on designation
STATUS_MISS_PROBABILITY: Dict[InjuryStatus, float] = {
    InjuryStatus.ACTIVE: 0.0,
    InjuryStatus.QUESTIONABLE: 0.50,
    InjuryStatus.DOUBTFUL: 0.85,
    InjuryStatus.OUT: 1.0,
    InjuryStatus.IR: 1.0,
    InjuryStatus.PUP: 1.0,
}


@dataclass
class PlayerInjury:
    """Injury report entry for a single player."""

    player_name: str
    position: Position
    status: InjuryStatus
    is_starter: bool = True


@dataclass
class TeamInjuryReport:
    """Aggregate injury report for one team for one game."""

    team: str
    game_id: str
    injuries: List[PlayerInjury] = field(default_factory=list)

    # ------- Feature computation -------

    @property
    def qb1_active(self) -> bool:
        """Is the starting QB expected to play?"""
        for inj in self.injuries:
            if inj.position == Position.QB and inj.is_starter:
                return STATUS_MISS_PROBABILITY[inj.status] < 0.5
        return True  # no injury listed → active

    @property
    def key_players_out_count(self) -> int:
        """Number of starters expected to miss the game."""
        return sum(
            1
            for inj in self.injuries
            if inj.is_starter and STATUS_MISS_PROBABILITY[inj.status] >= 0.5
        )

    @property
    def injury_adjusted_strength(self) -> float:
        """
        1.0 = fully healthy roster; lower = more impacted.

        Each absent starter reduces strength proportional to their
        positional weight.
        """
        penalty = 0.0
        for inj in self.injuries:
            if inj.is_starter:
                miss_prob = STATUS_MISS_PROBABILITY[inj.status]
                weight = POSITION_WEIGHTS.get(inj.position, 0.1)
                penalty += miss_prob * weight
        return max(0.0, 1.0 - penalty)

    def to_feature_dict(self) -> Dict[str, float]:
        """Return flat dict suitable for merging into a feature DataFrame."""
        return {
            "injury_qb1_active": float(self.qb1_active),
            "injury_key_players_out": float(self.key_players_out_count),
            "injury_adjusted_strength": self.injury_adjusted_strength,
        }


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class InjuryProvider(ABC):
    """Abstract base for injury/roster data backends."""

    @abstractmethod
    def get_team_injuries(
        self, team: str, game_id: str, season: int, week: int
    ) -> TeamInjuryReport:
        """Return injury report for a team in a specific game week."""

    def get_game_injuries(
        self,
        home_team: str,
        away_team: str,
        game_id: str,
        season: int,
        week: int,
    ) -> Dict[str, TeamInjuryReport]:
        """Return injury reports for both teams."""
        return {
            "home": self.get_team_injuries(home_team, game_id, season, week),
            "away": self.get_team_injuries(away_team, game_id, season, week),
        }


class NullInjuryProvider(InjuryProvider):
    """
    Fallback provider returning healthy rosters.

    Use this when injury data is unavailable or for baseline comparisons.
    """

    def get_team_injuries(
        self, team: str, game_id: str, season: int, week: int
    ) -> TeamInjuryReport:
        return TeamInjuryReport(team=team, game_id=game_id, injuries=[])


def get_injury_provider() -> InjuryProvider:
    """Factory — returns NullInjuryProvider until a real data source is connected."""
    return NullInjuryProvider()
