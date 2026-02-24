"""
Unit tests for bankroll management.

Tests include validation of all bankroll management rules.
"""

import pytest
from datetime import date, timedelta
from src.strategy.bankroll import BankrollManager


class TestBankrollManagerInitialization:
    """Test BankrollManager initialization."""

    def test_valid_initialization(self):
        """Test valid initialization."""
        bm = BankrollManager(
            starting_bankroll=10000,
            max_bet_pct=0.05,
            min_bet_pct=0.01,
            stop_loss_pct=0.50,
            daily_exposure_pct=0.15,
        )
        assert bm.starting_bankroll == 10000
        assert bm.current_bankroll == 10000
        assert bm.max_bet_pct == 0.05
        assert bm.min_bet_pct == 0.01
        assert bm.stop_loss_pct == 0.50
        assert bm.daily_exposure_pct == 0.15

    def test_invalid_starting_bankroll(self):
        """Test that negative/zero bankroll raises error."""
        with pytest.raises(ValueError, match="starting_bankroll must be positive"):
            BankrollManager(starting_bankroll=0)

        with pytest.raises(ValueError, match="starting_bankroll must be positive"):
            BankrollManager(starting_bankroll=-1000)

    def test_invalid_max_bet_pct(self):
        """Test that invalid max_bet_pct raises error."""
        with pytest.raises(ValueError, match="max_bet_pct must be between"):
            BankrollManager(starting_bankroll=10000, max_bet_pct=0.0)

        with pytest.raises(ValueError, match="max_bet_pct must be between"):
            BankrollManager(starting_bankroll=10000, max_bet_pct=1.5)

    def test_invalid_min_bet_pct(self):
        """Test that invalid min_bet_pct raises error."""
        with pytest.raises(ValueError, match="min_bet_pct must be between"):
            BankrollManager(starting_bankroll=10000, min_bet_pct=0.0)

        # min > max
        with pytest.raises(ValueError, match="min_bet_pct must be between"):
            BankrollManager(starting_bankroll=10000, min_bet_pct=0.1, max_bet_pct=0.05)

    def test_invalid_stop_loss_pct(self):
        """Test that invalid stop_loss_pct raises error."""
        with pytest.raises(ValueError, match="stop_loss_pct must be between"):
            BankrollManager(starting_bankroll=10000, stop_loss_pct=0.0)

        with pytest.raises(ValueError, match="stop_loss_pct must be between"):
            BankrollManager(starting_bankroll=10000, stop_loss_pct=1.5)

    def test_invalid_daily_exposure_pct(self):
        """Test that invalid daily_exposure_pct raises error."""
        with pytest.raises(ValueError, match="daily_exposure_pct must be between"):
            BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.0)

        with pytest.raises(ValueError, match="daily_exposure_pct must be between"):
            BankrollManager(starting_bankroll=10000, daily_exposure_pct=1.5)


class TestStopLoss:
    """Test stop-loss functionality."""

    def test_stop_loss_not_triggered(self):
        """Test that stop-loss is not triggered when above threshold."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 6000  # 60% of starting, above 50% threshold
        assert not bm.check_stop_loss()
        assert not bm.is_stopped

    def test_stop_loss_triggered(self):
        """Test that stop-loss is triggered when below threshold."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 4500  # 45% of starting, below 50% threshold
        assert bm.check_stop_loss()
        assert bm.is_stopped

    def test_stop_loss_exact_threshold(self):
        """Test stop-loss at exact threshold."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 5000  # Exactly 50%
        assert not bm.check_stop_loss()  # Not triggered (>= threshold)

        bm.current_bankroll = 4999.99  # Just below
        assert bm.check_stop_loss()  # Triggered

    def test_stop_loss_threshold_property(self):
        """Test stop_loss_threshold property."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        assert bm.stop_loss_threshold == 5000

        bm2 = BankrollManager(starting_bankroll=5000, stop_loss_pct=0.40)
        assert bm2.stop_loss_threshold == 2000


class TestMaxBetSize:
    """Test maximum bet sizing with constraints."""

    def test_kelly_below_max(self):
        """Test when Kelly size is below max cap."""
        bm = BankrollManager(starting_bankroll=10000, max_bet_pct=0.05)
        # Kelly suggests 3%
        bet_size = bm.get_max_bet_size(kelly_size=0.03)
        assert bet_size == 300.0  # 3% of 10000

    def test_kelly_above_max_cap(self):
        """Test that max cap is applied when Kelly suggests more."""
        bm = BankrollManager(starting_bankroll=10000, max_bet_pct=0.05)
        # Kelly suggests 10% but max is 5%
        bet_size = bm.get_max_bet_size(kelly_size=0.10)
        assert bet_size == 500.0  # Capped at 5%

    def test_kelly_below_min_threshold(self):
        """Test that bets below minimum are rounded to zero."""
        bm = BankrollManager(
            starting_bankroll=10000, max_bet_pct=0.05, min_bet_pct=0.01
        )
        # Kelly suggests 0.5%, below 1% minimum
        bet_size = bm.get_max_bet_size(kelly_size=0.005)
        assert bet_size == 0.0

    def test_kelly_at_min_threshold(self):
        """Test that bet at minimum threshold is accepted."""
        bm = BankrollManager(
            starting_bankroll=10000, max_bet_pct=0.05, min_bet_pct=0.01
        )
        # Kelly suggests exactly 1%
        bet_size = bm.get_max_bet_size(kelly_size=0.01)
        assert bet_size == 100.0

    def test_stop_loss_prevents_betting(self):
        """Test that stop-loss prevents betting."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 4000  # Below stop-loss
        bet_size = bm.get_max_bet_size(kelly_size=0.03)
        assert bet_size == 0.0  # Can't bet when stopped

    def test_bet_size_scales_with_bankroll(self):
        """Test that bet size scales with current bankroll."""
        bm = BankrollManager(starting_bankroll=10000, max_bet_pct=0.05)

        # Initial bankroll
        bet1 = bm.get_max_bet_size(kelly_size=0.03)
        assert bet1 == 300.0  # 3% of 10000

        # Bankroll grows
        bm.current_bankroll = 15000
        bet2 = bm.get_max_bet_size(kelly_size=0.03)
        assert bet2 == 450.0  # 3% of 15000

        # Bankroll shrinks
        bm.current_bankroll = 8000
        bet3 = bm.get_max_bet_size(kelly_size=0.03)
        assert bet3 == 240.0  # 3% of 8000


class TestDailyExposure:
    """Test daily exposure limits."""

    def test_initial_exposure_zero(self):
        """Test that initial daily exposure is zero."""
        bm = BankrollManager(starting_bankroll=10000)
        today = date.today()
        assert bm.get_daily_exposure(today) == 0.0

    def test_add_single_bet(self):
        """Test adding a single bet to daily exposure."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        today = date.today()

        assert bm.add_bet(500, today)
        assert bm.get_daily_exposure(today) == 500.0

    def test_add_multiple_bets_same_day(self):
        """Test adding multiple bets on the same day."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        today = date.today()

        assert bm.add_bet(500, today)
        assert bm.add_bet(300, today)
        assert bm.add_bet(200, today)
        assert bm.get_daily_exposure(today) == 1000.0

    def test_daily_exposure_limit_exceeded(self):
        """Test that daily exposure limit prevents over-betting."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        # Max daily exposure: 15% of 10000 = $1500
        today = date.today()

        # Add $1000
        assert bm.add_bet(1000, today)

        # Try to add another $600 (would exceed $1500)
        can_bet, reason = bm.can_add_bet(600, today)
        assert not can_bet
        assert "Daily exposure limit exceeded" in reason

    def test_daily_exposure_different_days(self):
        """Test that exposure is tracked separately per day."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        today = date.today()
        tomorrow = today + timedelta(days=1)

        # Max $1500 per day
        assert bm.add_bet(1500, today)
        assert bm.get_daily_exposure(today) == 1500.0

        # Should be able to bet full amount again tomorrow
        assert bm.add_bet(1500, tomorrow)
        assert bm.get_daily_exposure(tomorrow) == 1500.0

        # Check both days
        assert bm.get_daily_exposure(today) == 1500.0
        assert bm.get_daily_exposure(tomorrow) == 1500.0

    def test_can_add_bet_checks(self):
        """Test can_add_bet validation."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        today = date.today()

        # Should be OK initially
        can_bet, reason = bm.can_add_bet(500, today)
        assert can_bet
        assert reason == "OK"

        # Add bet
        bm.add_bet(1000, today)

        # Should still be OK (1000 + 400 < 1500)
        can_bet, reason = bm.can_add_bet(400, today)
        assert can_bet

        # Should fail (1000 + 600 > 1500)
        can_bet, reason = bm.can_add_bet(600, today)
        assert not can_bet

    def test_stop_loss_prevents_adding_bet(self):
        """Test that stop-loss prevents adding bets."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 4000  # Trigger stop-loss
        today = date.today()

        can_bet, reason = bm.can_add_bet(100, today)
        assert not can_bet
        assert "Stop-loss triggered" in reason


class TestSettleBet:
    """Test bet settlement and bankroll updates."""

    def test_settle_winning_bet(self):
        """Test settling a winning bet."""
        bm = BankrollManager(starting_bankroll=10000)
        bm.settle_bet(profit=100)  # Won $100
        assert bm.current_bankroll == 10100

    def test_settle_losing_bet(self):
        """Test settling a losing bet."""
        bm = BankrollManager(starting_bankroll=10000)
        bm.settle_bet(profit=-200)  # Lost $200
        assert bm.current_bankroll == 9800

    def test_multiple_settlements(self):
        """Test multiple bet settlements."""
        bm = BankrollManager(starting_bankroll=10000)
        bm.settle_bet(100)  # Win
        bm.settle_bet(-50)  # Loss
        bm.settle_bet(200)  # Win
        bm.settle_bet(-100)  # Loss
        assert bm.current_bankroll == 10150  # Net +150

    def test_bankroll_cannot_go_negative(self):
        """Test that bankroll cannot go below zero."""
        bm = BankrollManager(starting_bankroll=1000)
        bm.settle_bet(-1500)  # Lose more than bankroll
        assert bm.current_bankroll == 0.0  # Capped at zero

    def test_settle_triggers_stop_loss(self):
        """Test that settlement can trigger stop-loss."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        assert not bm.is_stopped

        # Lose enough to trigger stop-loss
        bm.settle_bet(-5500)  # Down to $4500
        assert bm.is_stopped
        assert bm.check_stop_loss()


class TestBankrollStats:
    """Test bankroll statistics."""

    def test_initial_stats(self):
        """Test stats at initialization."""
        bm = BankrollManager(
            starting_bankroll=10000, max_bet_pct=0.05, min_bet_pct=0.01
        )
        stats = bm.get_stats()

        assert stats["starting_bankroll"] == 10000
        assert stats["current_bankroll"] == 10000
        assert stats["total_return"] == 0
        assert stats["total_return_pct"] == 0.0
        assert stats["stop_loss_threshold"] == 5000
        assert not stats["is_stopped"]
        assert stats["max_bet_dollars"] == 500  # 5% of 10000
        assert stats["min_bet_dollars"] == 100  # 1% of 10000

    def test_stats_after_profit(self):
        """Test stats after profitable trades."""
        bm = BankrollManager(starting_bankroll=10000)
        bm.current_bankroll = 12000  # +$2000

        stats = bm.get_stats()
        assert stats["current_bankroll"] == 12000
        assert stats["total_return"] == 2000
        assert stats["total_return_pct"] == 0.20  # +20%

    def test_stats_after_loss(self):
        """Test stats after losses."""
        bm = BankrollManager(starting_bankroll=10000)
        bm.current_bankroll = 8500  # -$1500

        stats = bm.get_stats()
        assert stats["current_bankroll"] == 8500
        assert stats["total_return"] == -1500
        assert stats["total_return_pct"] == -0.15  # -15%


class TestReset:
    """Test bankroll reset functionality."""

    def test_reset_bankroll(self):
        """Test resetting bankroll to starting amount."""
        bm = BankrollManager(starting_bankroll=10000)
        today = date.today()

        # Make changes
        bm.current_bankroll = 8000
        bm.add_bet(500, today)
        bm.settle_bet(-1000)

        # Reset
        bm.reset()

        assert bm.current_bankroll == 10000
        assert bm.get_daily_exposure(today) == 0.0
        assert not bm.is_stopped

    def test_reset_clears_stop_loss(self):
        """Test that reset clears stop-loss state."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 4000
        bm.check_stop_loss()  # Explicitly trigger stop-loss check
        assert bm.is_stopped

        bm.reset()
        assert not bm.is_stopped
        assert bm.current_bankroll == 10000


class TestAcceptanceCriteria:
    """Test acceptance criteria from requirements."""

    def test_max_bet_5_percent_cap(self):
        """Verify max bet is 5% of current bankroll."""
        bm = BankrollManager(starting_bankroll=10000, max_bet_pct=0.05)
        # Even if Kelly suggests 10%, should be capped at 5%
        bet = bm.get_max_bet_size(kelly_size=0.10)
        assert bet == 500.0  # 5% of 10000

    def test_min_bet_1_percent(self):
        """Verify min bet is 1% of bankroll."""
        bm = BankrollManager(
            starting_bankroll=10000, max_bet_pct=0.05, min_bet_pct=0.01
        )
        # Bet below 1% should be rejected
        bet = bm.get_max_bet_size(kelly_size=0.005)
        assert bet == 0.0

        # Bet at or above 1% should be accepted
        bet = bm.get_max_bet_size(kelly_size=0.01)
        assert bet == 100.0

    def test_stop_loss_50_percent(self):
        """Verify stop-loss halts trading at 50% of starting bankroll."""
        bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
        bm.current_bankroll = 4900  # Below 50%

        assert bm.check_stop_loss()
        assert bm.get_max_bet_size(kelly_size=0.03) == 0.0  # Can't bet

    def test_daily_exposure_15_percent(self):
        """Verify daily exposure limit is 15% of bankroll."""
        bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
        today = date.today()

        # Should be able to risk up to $1500
        assert bm.add_bet(1500, today)

        # But not $1501
        can_bet, _ = bm.can_add_bet(1, today)
        assert not can_bet
