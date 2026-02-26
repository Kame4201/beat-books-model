"""Tests for injury/roster feature provider."""

import pytest
from src.features.injuries import (
    InjuryStatus,
    NullInjuryProvider,
    PlayerInjury,
    Position,
    POSITION_WEIGHTS,
    STATUS_MISS_PROBABILITY,
    TeamInjuryReport,
    get_injury_provider,
)


# ---------------------------------------------------------------------------
# TeamInjuryReport feature computation
# ---------------------------------------------------------------------------

class TestTeamInjuryReport:
    def test_healthy_roster(self):
        report = TeamInjuryReport(team="KC", game_id="g1", injuries=[])
        assert report.qb1_active is True
        assert report.key_players_out_count == 0
        assert report.injury_adjusted_strength == 1.0

    def test_qb_out(self):
        report = TeamInjuryReport(
            team="KC", game_id="g1",
            injuries=[
                PlayerInjury("P. Mahomes", Position.QB, InjuryStatus.OUT, is_starter=True),
            ],
        )
        assert report.qb1_active is False
        assert report.key_players_out_count == 1
        assert report.injury_adjusted_strength < 1.0

    def test_qb_questionable(self):
        report = TeamInjuryReport(
            team="KC", game_id="g1",
            injuries=[
                PlayerInjury("P. Mahomes", Position.QB, InjuryStatus.QUESTIONABLE, is_starter=True),
            ],
        )
        # 50% miss probability — coin flip, but feature should reflect uncertainty
        assert report.qb1_active is False  # 0.50 >= 0.5 threshold
        assert report.injury_adjusted_strength < 1.0

    def test_multiple_starters_out(self):
        report = TeamInjuryReport(
            team="KC", game_id="g1",
            injuries=[
                PlayerInjury("RB1", Position.RB, InjuryStatus.OUT, is_starter=True),
                PlayerInjury("WR1", Position.WR, InjuryStatus.OUT, is_starter=True),
                PlayerInjury("WR3", Position.WR, InjuryStatus.OUT, is_starter=False),
            ],
        )
        assert report.key_players_out_count == 2  # non-starter excluded
        assert report.injury_adjusted_strength < 1.0

    def test_non_starter_doesnt_affect_count(self):
        report = TeamInjuryReport(
            team="KC", game_id="g1",
            injuries=[
                PlayerInjury("Backup", Position.WR, InjuryStatus.OUT, is_starter=False),
            ],
        )
        assert report.key_players_out_count == 0

    def test_ir_player(self):
        report = TeamInjuryReport(
            team="KC", game_id="g1",
            injuries=[
                PlayerInjury("LB1", Position.LB, InjuryStatus.IR, is_starter=True),
            ],
        )
        assert report.key_players_out_count == 1
        assert report.injury_adjusted_strength == pytest.approx(1.0 - 0.30, abs=0.01)

    def test_feature_dict_keys(self):
        report = TeamInjuryReport(team="KC", game_id="g1", injuries=[])
        d = report.to_feature_dict()
        assert set(d.keys()) == {
            "injury_qb1_active",
            "injury_key_players_out",
            "injury_adjusted_strength",
        }

    def test_strength_floor(self):
        """Strength should never go below 0."""
        injuries = [
            PlayerInjury(f"P{i}", Position.QB, InjuryStatus.OUT, is_starter=True)
            for i in range(10)
        ]
        report = TeamInjuryReport(team="KC", game_id="g1", injuries=injuries)
        assert report.injury_adjusted_strength >= 0.0


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------

class TestNullProvider:
    def test_returns_healthy(self):
        provider = NullInjuryProvider()
        report = provider.get_team_injuries("KC", "g1", 2024, 1)
        assert report.injuries == []
        assert report.injury_adjusted_strength == 1.0

    def test_game_injuries(self):
        provider = NullInjuryProvider()
        reports = provider.get_game_injuries("KC", "BUF", "g1", 2024, 1)
        assert "home" in reports and "away" in reports


class TestFactory:
    def test_default_is_null(self):
        p = get_injury_provider()
        assert isinstance(p, NullInjuryProvider)


# ---------------------------------------------------------------------------
# Enum / weight coverage
# ---------------------------------------------------------------------------

class TestEnums:
    def test_all_positions_have_weights(self):
        for pos in Position:
            assert pos in POSITION_WEIGHTS

    def test_all_statuses_have_miss_prob(self):
        for status in InjuryStatus:
            assert status in STATUS_MISS_PROBABILITY
            assert 0.0 <= STATUS_MISS_PROBABILITY[status] <= 1.0
