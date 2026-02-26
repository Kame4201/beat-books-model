"""Tests for advanced stats provider (NGS / PFF)."""

import pytest
from src.features.advanced_stats import (
    AdvancedStatsProvider,
    FixtureAdvancedStatsProvider,
    NullAdvancedStatsProvider,
    TeamAdvancedStats,
    get_advanced_stats_provider,
)


class TestTeamAdvancedStats:
    def test_feature_dict_keys(self):
        stats = TeamAdvancedStats(team="KC", season=2024, week=1)
        d = stats.to_feature_dict()
        expected = {
            "adv_cpoe", "adv_ryoe", "adv_avg_separation",
            "adv_avg_time_to_throw", "adv_pff_offense_grade",
            "adv_pff_defense_grade", "adv_pff_pass_block_grade",
            "adv_pff_run_block_grade",
        }
        assert set(d.keys()) == expected

    def test_none_replaced_with_zero(self):
        stats = TeamAdvancedStats(team="KC", season=2024, week=1)
        d = stats.to_feature_dict()
        assert all(v == 0.0 for v in d.values())

    def test_populated_values(self):
        stats = TeamAdvancedStats(
            team="KC", season=2024, week=1,
            cpoe=3.5, ryoe=0.8, pff_offense_grade=82.1,
        )
        d = stats.to_feature_dict()
        assert d["adv_cpoe"] == 3.5
        assert d["adv_ryoe"] == 0.8
        assert d["adv_pff_offense_grade"] == 82.1
        assert d["adv_pff_defense_grade"] == 0.0  # still None


class TestNullProvider:
    def test_returns_empty_stats(self):
        p = NullAdvancedStatsProvider()
        stats = p.get_team_stats("KC", 2024, 1)
        assert stats.cpoe is None
        assert stats.team == "KC"

    def test_game_stats(self):
        p = NullAdvancedStatsProvider()
        result = p.get_game_stats("KC", "BUF", 2024, 1)
        assert "home" in result and "away" in result
        assert result["home"].team == "KC"
        assert result["away"].team == "BUF"


class TestFixtureProvider:
    def test_returns_fixture_data(self):
        data = {
            "KC-2024-1": TeamAdvancedStats(
                team="KC", season=2024, week=1,
                cpoe=4.0, pff_offense_grade=85.0,
            ),
        }
        p = FixtureAdvancedStatsProvider(data)
        stats = p.get_team_stats("KC", 2024, 1)
        assert stats.cpoe == 4.0
        assert stats.pff_offense_grade == 85.0

    def test_missing_key_returns_empty(self):
        p = FixtureAdvancedStatsProvider({})
        stats = p.get_team_stats("KC", 2024, 1)
        assert stats.cpoe is None


class TestFactory:
    def test_default_is_null(self):
        p = get_advanced_stats_provider()
        assert isinstance(p, NullAdvancedStatsProvider)

    def test_with_fixture(self):
        p = get_advanced_stats_provider(fixture_data={})
        assert isinstance(p, FixtureAdvancedStatsProvider)
