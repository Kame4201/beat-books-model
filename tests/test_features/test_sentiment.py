"""Tests for sentiment and public betting data provider."""

import pytest
from src.features.sentiment import (
    FixtureSentimentProvider,
    GameSentiment,
    NullSentimentProvider,
    get_sentiment_provider,
)


class TestGameSentiment:
    def test_feature_dict_keys(self):
        gs = GameSentiment(game_id="g1", home_team="KC", away_team="BUF")
        d = gs.to_feature_dict()
        expected = {
            "sent_public_bet_pct_home", "sent_money_vs_bet_divergence",
            "sent_line_movement", "sent_reverse_line_movement",
            "sent_news_sentiment_home", "sent_news_sentiment_away",
            "sent_is_revenge_game", "sent_is_prime_time",
            "sent_is_divisional", "sent_home_coaching_change",
            "sent_away_coaching_change",
        }
        assert set(d.keys()) == expected

    def test_neutral_defaults(self):
        gs = GameSentiment(game_id="g1", home_team="KC", away_team="BUF")
        d = gs.to_feature_dict()
        assert d["sent_public_bet_pct_home"] == 50.0
        assert d["sent_line_movement"] == 0.0
        assert d["sent_is_revenge_game"] == 0.0

    def test_money_vs_bet_divergence(self):
        gs = GameSentiment(
            game_id="g1", home_team="KC", away_team="BUF",
            public_bet_pct_home=65.0, public_money_pct_home=45.0,
        )
        assert gs.money_vs_bet_divergence == -20.0

    def test_money_vs_bet_divergence_none(self):
        gs = GameSentiment(game_id="g1", home_team="KC", away_team="BUF")
        assert gs.money_vs_bet_divergence is None

    def test_reverse_line_movement_true(self):
        """Public on home but line moves away from home → RLM."""
        gs = GameSentiment(
            game_id="g1", home_team="KC", away_team="BUF",
            public_bet_pct_home=70.0, line_movement=1.5,  # line moved against home
        )
        assert gs.reverse_line_movement is True

    def test_reverse_line_movement_false(self):
        """Public on home and line moves toward home → no RLM."""
        gs = GameSentiment(
            game_id="g1", home_team="KC", away_team="BUF",
            public_bet_pct_home=70.0, line_movement=-1.0,
        )
        assert gs.reverse_line_movement is False

    def test_reverse_line_movement_none(self):
        gs = GameSentiment(game_id="g1", home_team="KC", away_team="BUF")
        assert gs.reverse_line_movement is None

    def test_narrative_flags(self):
        gs = GameSentiment(
            game_id="g1", home_team="KC", away_team="BUF",
            is_revenge_game=True, is_prime_time=True, is_divisional=True,
            home_coaching_change=True,
        )
        d = gs.to_feature_dict()
        assert d["sent_is_revenge_game"] == 1.0
        assert d["sent_is_prime_time"] == 1.0
        assert d["sent_is_divisional"] == 1.0
        assert d["sent_home_coaching_change"] == 1.0
        assert d["sent_away_coaching_change"] == 0.0


class TestNullProvider:
    def test_returns_neutral(self):
        p = NullSentimentProvider()
        gs = p.get_game_sentiment("g1", "KC", "BUF", 2024, 1)
        assert gs.public_bet_pct_home is None
        assert gs.news_sentiment_home is None

    def test_week_sentiment(self):
        p = NullSentimentProvider()
        games = [
            {"game_id": "g1", "home_team": "KC", "away_team": "BUF"},
            {"game_id": "g2", "home_team": "DAL", "away_team": "PHI"},
        ]
        results = p.get_week_sentiment(games, 2024, 1)
        assert len(results) == 2


class TestFixtureProvider:
    def test_returns_fixture_data(self):
        data = {
            "g1": GameSentiment(
                game_id="g1", home_team="KC", away_team="BUF",
                public_bet_pct_home=72.0, news_sentiment_home=0.6,
            ),
        }
        p = FixtureSentimentProvider(data)
        gs = p.get_game_sentiment("g1", "KC", "BUF", 2024, 1)
        assert gs.public_bet_pct_home == 72.0

    def test_missing_returns_neutral(self):
        p = FixtureSentimentProvider({})
        gs = p.get_game_sentiment("g1", "KC", "BUF", 2024, 1)
        assert gs.public_bet_pct_home is None


class TestFactory:
    def test_default_is_null(self):
        p = get_sentiment_provider()
        assert isinstance(p, NullSentimentProvider)

    def test_with_fixture(self):
        p = get_sentiment_provider(fixture_data={})
        assert isinstance(p, FixtureSentimentProvider)
