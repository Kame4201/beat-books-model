"""
Tests for bankroll simulation and bet sizing.
"""
import pytest
from src.backtesting.bankroll_sim import (
    BankrollTracker,
    calculate_bet_size,
    calculate_kelly_criterion,
    should_place_bet,
    calculate_bet_payout,
    american_odds_to_probability,
    probability_to_american_odds,
)
from src.backtesting.types import BacktestConfig, BetSizingMethod


@pytest.fixture
def flat_config():
    """Backtest config with flat bet sizing."""
    return BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2024-W1",
        end_date="2024-W17",
        bet_sizing_method=BetSizingMethod.FLAT,
        flat_bet_size=100.0,
    )


@pytest.fixture
def percentage_config():
    """Backtest config with percentage bet sizing."""
    return BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2024-W1",
        end_date="2024-W17",
        bet_sizing_method=BetSizingMethod.PERCENTAGE,
        percentage_bet_size=0.02,
    )


@pytest.fixture
def kelly_config():
    """Backtest config with Kelly Criterion bet sizing."""
    return BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2024-W1",
        end_date="2024-W17",
        bet_sizing_method=BetSizingMethod.KELLY,
        kelly_fraction=0.25,
    )


def test_calculate_bet_size_flat(flat_config):
    """Test flat bet sizing."""
    bet_size = calculate_bet_size(
        flat_config,
        bankroll=1000.0,
        model_prob=0.60,
        market_prob=0.55,
        market_odds=-110,
    )
    assert bet_size == 100.0


def test_calculate_bet_size_percentage(percentage_config):
    """Test percentage bet sizing."""
    bet_size = calculate_bet_size(
        percentage_config,
        bankroll=1000.0,
        model_prob=0.60,
        market_prob=0.55,
        market_odds=-110,
    )
    assert bet_size == pytest.approx(20.0, rel=1e-6)  # 2% of 1000


def test_calculate_bet_size_kelly(kelly_config):
    """Test Kelly Criterion bet sizing."""
    bet_size = calculate_bet_size(
        kelly_config,
        bankroll=1000.0,
        model_prob=0.60,
        market_prob=0.5238,  # -110 odds
        market_odds=-110,
    )
    # Should be > 0 since we have edge
    assert bet_size > 0
    # Should be < max bet fraction (5% = 50)
    assert bet_size <= 50.0


def test_calculate_bet_size_respects_min(flat_config):
    """Test that bet size respects minimum."""
    flat_config.flat_bet_size = 5.0
    bet_size = calculate_bet_size(
        flat_config,
        bankroll=1000.0,
        model_prob=0.60,
    )
    assert bet_size == flat_config.min_bet_size  # 10.0 by default


def test_calculate_bet_size_respects_max(flat_config):
    """Test that bet size respects maximum."""
    flat_config.flat_bet_size = 1000.0
    bet_size = calculate_bet_size(
        flat_config,
        bankroll=1000.0,
        model_prob=0.60,
    )
    assert bet_size == flat_config.max_bet_size  # 500.0 by default


def test_calculate_bet_size_respects_max_fraction(percentage_config):
    """Test that bet size respects max bankroll fraction."""
    percentage_config.percentage_bet_size = 0.10  # 10%
    bet_size = calculate_bet_size(
        percentage_config,
        bankroll=1000.0,
        model_prob=0.60,
    )
    # Should cap at 5% (max_bet_fraction)
    assert bet_size == pytest.approx(50.0, rel=1e-6)


def test_calculate_kelly_criterion():
    """Test Kelly Criterion calculation."""
    # Scenario: Model says 60%, market says 52.38% (odds -110)
    kelly = calculate_kelly_criterion(
        model_prob=0.60,
        market_prob=0.5238,
        market_odds=-110,
    )
    # Should have positive edge
    assert kelly > 0
    # Should be reasonable (not > 50% of bankroll)
    assert kelly < 0.5


def test_calculate_kelly_criterion_no_edge():
    """Test Kelly when there's no edge."""
    kelly = calculate_kelly_criterion(
        model_prob=0.50,
        market_prob=0.50,
        market_odds=-110,
    )
    # Should be 0 or negative (no bet)
    assert kelly <= 0.01


def test_calculate_kelly_criterion_positive_odds():
    """Test Kelly with positive odds (underdog)."""
    kelly = calculate_kelly_criterion(
        model_prob=0.60,
        market_prob=0.40,
        market_odds=+150,
    )
    # Should have positive edge
    assert kelly > 0


def test_should_place_bet_with_edge(kelly_config):
    """Test bet placement when edge exceeds threshold."""
    should_bet = should_place_bet(
        kelly_config,
        model_prob=0.60,
        market_prob=0.55,
    )
    # Edge = 0.05 > 0.01 threshold
    assert should_bet is True


def test_should_place_bet_no_edge(kelly_config):
    """Test bet placement when edge is below threshold."""
    should_bet = should_place_bet(
        kelly_config,
        model_prob=0.51,
        market_prob=0.50,
    )
    # Edge = 0.01 = threshold (borderline)
    assert should_bet is True


def test_should_place_bet_insufficient_edge(kelly_config):
    """Test bet placement when edge is insufficient."""
    should_bet = should_place_bet(
        kelly_config,
        model_prob=0.505,
        market_prob=0.50,
    )
    # Edge = 0.005 < 0.01 threshold
    assert should_bet is False


def test_should_place_bet_no_market_data(kelly_config):
    """Test bet placement with no market data."""
    should_bet = should_place_bet(
        kelly_config,
        model_prob=0.60,
        market_prob=None,
    )
    assert should_bet is False


def test_calculate_bet_payout_win():
    """Test payout calculation for winning bet."""
    payout, profit = calculate_bet_payout(
        bet_amount=100.0,
        odds=-110,
        won=True,
    )
    # -110 odds: win $100 to win $90.91
    assert profit == pytest.approx(90.909, rel=0.01)
    assert payout == pytest.approx(190.909, rel=0.01)


def test_calculate_bet_payout_loss():
    """Test payout calculation for losing bet."""
    payout, profit = calculate_bet_payout(
        bet_amount=100.0,
        odds=-110,
        won=False,
    )
    assert payout == 0.0
    assert profit == -100.0


def test_calculate_bet_payout_positive_odds():
    """Test payout with positive odds (underdog)."""
    payout, profit = calculate_bet_payout(
        bet_amount=100.0,
        odds=+150,
        won=True,
    )
    # +150 odds: $100 wins $150
    assert profit == pytest.approx(150.0, rel=1e-6)
    assert payout == pytest.approx(250.0, rel=1e-6)


def test_american_odds_to_probability():
    """Test converting American odds to probability."""
    # -110 odds (favorite)
    prob = american_odds_to_probability(-110)
    assert prob == pytest.approx(0.5238, rel=0.01)

    # +150 odds (underdog)
    prob = american_odds_to_probability(+150)
    assert prob == pytest.approx(0.40, rel=0.01)

    # -200 odds (heavy favorite)
    prob = american_odds_to_probability(-200)
    assert prob == pytest.approx(0.6667, rel=0.01)


def test_probability_to_american_odds():
    """Test converting probability to American odds."""
    # 52.38% favorite
    odds = probability_to_american_odds(0.5238)
    assert odds == pytest.approx(-110, rel=0.1)

    # 40% underdog
    odds = probability_to_american_odds(0.40)
    assert odds == pytest.approx(+150, rel=0.1)

    # 66.67% favorite
    odds = probability_to_american_odds(0.6667)
    assert odds == pytest.approx(-200, rel=0.1)


def test_bankroll_tracker_initialization():
    """Test BankrollTracker initialization."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    assert tracker.current_bankroll == 1000.0
    assert tracker.starting_bankroll == 1000.0
    assert tracker.total_bets == 0
    assert tracker.total_wagered == 0.0
    assert tracker.total_profit == 0.0


def test_bankroll_tracker_place_winning_bet():
    """Test placing a winning bet."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)

    assert tracker.current_bankroll == 1091.0
    assert tracker.total_bets == 1
    assert tracker.total_wagered == 100.0
    assert tracker.total_profit == 91.0
    assert tracker.wins == 1
    assert tracker.losses == 0


def test_bankroll_tracker_place_losing_bet():
    """Test placing a losing bet."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    assert tracker.current_bankroll == 900.0
    assert tracker.total_bets == 1
    assert tracker.total_wagered == 100.0
    assert tracker.total_profit == -100.0
    assert tracker.wins == 0
    assert tracker.losses == 1


def test_bankroll_tracker_multiple_bets():
    """Test multiple bets."""
    tracker = BankrollTracker(starting_bankroll=1000.0)

    # Win
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    # Win
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    # Loss
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    assert tracker.current_bankroll == pytest.approx(1082.0, rel=1e-6)
    assert tracker.total_bets == 3
    assert tracker.total_wagered == 300.0
    assert tracker.total_profit == pytest.approx(82.0, rel=1e-6)
    assert tracker.wins == 2
    assert tracker.losses == 1


def test_bankroll_tracker_get_roi():
    """Test ROI calculation."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    roi = tracker.get_roi()
    # (91 - 100) / 200 * 100 = -4.5%
    assert roi == pytest.approx(-4.5, rel=1e-6)


def test_bankroll_tracker_get_win_rate():
    """Test win rate calculation."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    win_rate = tracker.get_win_rate()
    assert win_rate == pytest.approx(66.667, rel=0.01)


def test_bankroll_tracker_get_max_drawdown():
    """Test max drawdown calculation."""
    tracker = BankrollTracker(starting_bankroll=1000.0)

    # Win: 1000 -> 1091 (new peak)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    # Win: 1091 -> 1182 (new peak)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    # Loss: 1182 -> 1082
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)
    # Loss: 1082 -> 982
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    max_dd, max_dd_pct = tracker.get_max_drawdown()
    # Peak 1182, trough 982, drawdown = 200
    assert max_dd == pytest.approx(200.0, rel=1e-6)
    assert max_dd_pct == pytest.approx((200 / 1182) * 100, rel=1e-6)


def test_bankroll_tracker_get_bankroll_curve():
    """Test bankroll curve retrieval."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    curve = tracker.get_bankroll_curve()
    assert curve["0"] == 1000.0
    assert curve["1"] == 1091.0
    assert curve["2"] == pytest.approx(991.0, rel=1e-6)


def test_bankroll_tracker_get_summary():
    """Test summary statistics."""
    tracker = BankrollTracker(starting_bankroll=1000.0)
    tracker.place_bet(bet_amount=100.0, profit=91.0, won=True)
    tracker.place_bet(bet_amount=100.0, profit=-100.0, won=False)

    summary = tracker.get_summary()
    assert summary["starting_bankroll"] == 1000.0
    assert summary["ending_bankroll"] == pytest.approx(991.0, rel=1e-6)
    assert summary["total_bets"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
