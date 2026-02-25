"""
News sentiment and public betting data provider.

Phase 2 feature — provides the interface and null/fixture implementations.
Live providers will be added when data source licensing is resolved.

Features:
- Public betting percentages (% of bets on each side)
- Line movement (opening → closing)
- Reverse line movement (sharp money indicator)
- News sentiment scores (NLP-based)
- Narrative factors (coaching changes, revenge games, etc.)

CRITICAL: This module only READS external data. It never creates or
modifies database tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GameSentiment:
    """Sentiment and public betting data for a single game."""

    game_id: str
    home_team: str
    away_team: str

    # Public betting percentages
    public_bet_pct_home: Optional[float] = None  # 0-100
    public_money_pct_home: Optional[float] = None  # 0-100

    # Line movement
    opening_spread: Optional[float] = None  # negative = home favored
    closing_spread: Optional[float] = None
    line_movement: Optional[float] = None  # closing - opening

    # News sentiment (-1.0 to 1.0 scale)
    news_sentiment_home: Optional[float] = None
    news_sentiment_away: Optional[float] = None

    # Narrative flags
    is_revenge_game: bool = False
    is_prime_time: bool = False
    is_divisional: bool = False
    home_coaching_change: bool = False
    away_coaching_change: bool = False

    @property
    def money_vs_bet_divergence(self) -> Optional[float]:
        """
        Divergence between money % and bet %.

        Positive = sharp money on home team (money % > bet %).
        """
        if self.public_money_pct_home is not None and self.public_bet_pct_home is not None:
            return self.public_money_pct_home - self.public_bet_pct_home
        return None

    @property
    def reverse_line_movement(self) -> Optional[bool]:
        """
        True if line moves opposite to public betting direction.

        This is a classic sharp money signal.
        """
        if (
            self.line_movement is not None
            and self.public_bet_pct_home is not None
        ):
            public_on_home = self.public_bet_pct_home > 50
            line_moved_toward_home = self.line_movement < 0
            return public_on_home != line_moved_toward_home
        return None

    def to_feature_dict(self) -> Dict[str, float]:
        """Return flat dict suitable for merging into a feature DataFrame."""
        rlm = self.reverse_line_movement
        div = self.money_vs_bet_divergence
        return {
            "sent_public_bet_pct_home": self.public_bet_pct_home if self.public_bet_pct_home is not None else 50.0,
            "sent_money_vs_bet_divergence": div if div is not None else 0.0,
            "sent_line_movement": self.line_movement if self.line_movement is not None else 0.0,
            "sent_reverse_line_movement": float(rlm) if rlm is not None else 0.0,
            "sent_news_sentiment_home": self.news_sentiment_home if self.news_sentiment_home is not None else 0.0,
            "sent_news_sentiment_away": self.news_sentiment_away if self.news_sentiment_away is not None else 0.0,
            "sent_is_revenge_game": float(self.is_revenge_game),
            "sent_is_prime_time": float(self.is_prime_time),
            "sent_is_divisional": float(self.is_divisional),
            "sent_home_coaching_change": float(self.home_coaching_change),
            "sent_away_coaching_change": float(self.away_coaching_change),
        }


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class SentimentProvider(ABC):
    """Abstract base for sentiment / public betting data backends."""

    @abstractmethod
    def get_game_sentiment(
        self, game_id: str, home_team: str, away_team: str,
        season: int, week: int,
    ) -> GameSentiment:
        """Return sentiment data for a single game."""

    def get_week_sentiment(
        self, games: List[Dict[str, str]], season: int, week: int,
    ) -> List[GameSentiment]:
        """Fetch sentiment for all games in a week. Default: iterate."""
        return [
            self.get_game_sentiment(
                g["game_id"], g["home_team"], g["away_team"], season, week
            )
            for g in games
        ]


class NullSentimentProvider(SentimentProvider):
    """
    Fallback provider returning neutral sentiment.

    Use when no sentiment data source is available.
    """

    def get_game_sentiment(
        self, game_id: str, home_team: str, away_team: str,
        season: int, week: int,
    ) -> GameSentiment:
        return GameSentiment(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
        )


class FixtureSentimentProvider(SentimentProvider):
    """
    Returns pre-loaded sentiment data from a dict.

    Useful for testing and backtesting with historical snapshots.
    """

    def __init__(self, data: Dict[str, GameSentiment]):
        """
        Args:
            data: mapping of game_id → GameSentiment
        """
        self._data = data

    def get_game_sentiment(
        self, game_id: str, home_team: str, away_team: str,
        season: int, week: int,
    ) -> GameSentiment:
        if game_id in self._data:
            return self._data[game_id]
        return GameSentiment(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
        )


def get_sentiment_provider(
    fixture_data: Optional[Dict[str, GameSentiment]] = None,
) -> SentimentProvider:
    """
    Factory.

    - Pass fixture_data for testing / backtesting with snapshots
    - Default returns NullSentimentProvider
    """
    if fixture_data is not None:
        return FixtureSentimentProvider(fixture_data)
    return NullSentimentProvider()
