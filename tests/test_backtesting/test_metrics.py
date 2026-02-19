"""
Tests for backtesting metrics calculations.
"""

import math

import pytest
from src.backtesting.types import PredictionRecord
from src.backtesting import metrics


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.65,
            predicted_spread=-3.5,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=24,
            actual_spread=-3.0,
            market_home_win_prob=0.60,
            market_spread=-3.0,
            closing_line=-3.5,
            bet_placed=True,
            bet_type="moneyline",
            bet_amount=100.0,
            bet_on_home=True,
            bet_won=True,
            bet_payout=191.0,
            bet_profit=91.0,
        ),
        PredictionRecord(
            game_id="2",
            week=1,
            season=2024,
            home_team="SF",
            away_team="DAL",
            predicted_home_win_prob=0.45,
            predicted_spread=2.5,
            actual_home_win=False,
            actual_home_score=20,
            actual_away_score=24,
            actual_spread=4.0,
            market_home_win_prob=0.50,
            market_spread=0.0,
            closing_line=0.5,
            bet_placed=True,
            bet_type="moneyline",
            bet_amount=100.0,
            bet_on_home=False,
            bet_won=True,
            bet_payout=200.0,
            bet_profit=100.0,
        ),
        PredictionRecord(
            game_id="3",
            week=2,
            season=2024,
            home_team="BUF",
            away_team="MIA",
            predicted_home_win_prob=0.70,
            predicted_spread=-5.0,
            actual_home_win=False,
            actual_home_score=17,
            actual_away_score=21,
            actual_spread=4.0,
            market_home_win_prob=0.55,
            market_spread=-2.5,
            closing_line=-3.0,
            bet_placed=True,
            bet_type="moneyline",
            bet_amount=150.0,
            bet_on_home=True,
            bet_won=False,
            bet_payout=0.0,
            bet_profit=-150.0,
        ),
    ]


def test_calculate_accuracy(sample_predictions):
    """Test overall accuracy calculation."""
    accuracy = metrics.calculate_accuracy(sample_predictions)
    # 2 correct out of 3
    assert accuracy == pytest.approx(2 / 3, rel=1e-6)


def test_calculate_accuracy_empty():
    """Test accuracy with no predictions."""
    assert metrics.calculate_accuracy([]) == 0.0


def test_calculate_home_away_accuracy(sample_predictions):
    """Test home and away accuracy calculation."""
    home_acc, away_acc = metrics.calculate_home_away_accuracy(sample_predictions)
    # Home picks: game 1 (correct), game 3 (wrong) = 1/2 = 0.5
    # Away picks: game 2 (correct) = 1/1 = 1.0
    assert home_acc == pytest.approx(0.5, rel=1e-6)
    assert away_acc == pytest.approx(1.0, rel=1e-6)


def test_calculate_log_loss(sample_predictions):
    """Test log loss calculation."""
    log_loss = metrics.calculate_log_loss(sample_predictions)
    # Should be positive and finite
    assert log_loss > 0
    assert log_loss < 10  # Reasonable upper bound


def test_calculate_log_loss_empty():
    """Test log loss with no predictions."""
    assert metrics.calculate_log_loss([]) == float("inf")


def test_calculate_brier_score(sample_predictions):
    """Test Brier score calculation."""
    brier = metrics.calculate_brier_score(sample_predictions)
    # Should be between 0 and 1
    assert 0 <= brier <= 1


def test_calculate_brier_score_perfect():
    """Test Brier score with perfect predictions."""
    perfect = [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=1.0,
            predicted_spread=-7.0,
            actual_home_win=True,
            actual_home_score=30,
            actual_away_score=20,
            actual_spread=-10.0,
        ),
        PredictionRecord(
            game_id="2",
            week=1,
            season=2024,
            home_team="SF",
            away_team="DAL",
            predicted_home_win_prob=0.0,
            predicted_spread=7.0,
            actual_home_win=False,
            actual_home_score=14,
            actual_away_score=28,
            actual_spread=14.0,
        ),
    ]
    brier = metrics.calculate_brier_score(perfect)
    assert brier == pytest.approx(0.0, abs=1e-6)


def test_calculate_roi(sample_predictions):
    """Test ROI calculation."""
    total_wagered, total_profit, roi = metrics.calculate_roi(sample_predictions)

    assert total_wagered == 350.0  # 100 + 100 + 150
    assert total_profit == 41.0  # 91 + 100 - 150
    assert roi == pytest.approx((41 / 350) * 100, rel=1e-6)


def test_calculate_roi_no_bets():
    """Test ROI with no bets placed."""
    no_bets = [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.55,
            predicted_spread=-2.0,
            actual_home_win=True,
            actual_home_score=24,
            actual_away_score=21,
            actual_spread=-3.0,
            bet_placed=False,
        )
    ]
    total_wagered, total_profit, roi = metrics.calculate_roi(no_bets)
    assert total_wagered == 0.0
    assert total_profit == 0.0
    assert roi == 0.0


def test_calculate_max_drawdown(sample_predictions):
    """Test max drawdown calculation."""
    starting_bankroll = 1000.0
    max_dd, max_dd_pct = metrics.calculate_max_drawdown(
        sample_predictions, starting_bankroll
    )

    # After game 1: 1000 + 91 = 1091 (new peak)
    # After game 2: 1091 + 100 = 1191 (new peak)
    # After game 3: 1191 - 150 = 1041
    # Drawdown: 1191 - 1041 = 150
    assert max_dd == pytest.approx(150.0, rel=1e-6)
    assert max_dd_pct == pytest.approx((150 / 1191) * 100, rel=1e-6)


def test_calculate_sharpe_ratio(sample_predictions):
    """Test Sharpe ratio calculation."""
    sharpe = metrics.calculate_sharpe_ratio(sample_predictions)
    # Should be finite
    assert math.isfinite(sharpe)


def test_calculate_sharpe_ratio_insufficient_data():
    """Test Sharpe ratio with insufficient data."""
    one_bet = [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.65,
            predicted_spread=-3.5,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=24,
            actual_spread=-3.0,
            bet_placed=True,
            bet_amount=100.0,
            bet_profit=91.0,
        )
    ]
    sharpe = metrics.calculate_sharpe_ratio(one_bet)
    assert sharpe == 0.0


def test_calculate_clv(sample_predictions):
    """Test CLV calculation."""
    avg_clv, clv_wins, clv_total = metrics.calculate_clv(sample_predictions)

    # All 3 games have closing line data
    assert clv_total == 3
    assert avg_clv is not None


def test_calculate_clv_no_data():
    """Test CLV with no closing line data."""
    no_clv = [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.65,
            predicted_spread=-3.5,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=24,
            actual_spread=-3.0,
            bet_placed=True,
            closing_line=None,
        )
    ]
    avg_clv, clv_wins, clv_total = metrics.calculate_clv(no_clv)
    assert avg_clv is None
    assert clv_total == 0


def test_calculate_edge_bucket_accuracy():
    """Test edge bucket accuracy calculation."""
    predictions = [
        # Edge 0.02 (1-3%), correct
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.62,
            predicted_spread=-3.0,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=24,
            actual_spread=-3.0,
            market_home_win_prob=0.60,
        ),
        # Edge 0.04 (3-5%), wrong
        PredictionRecord(
            game_id="2",
            week=1,
            season=2024,
            home_team="SF",
            away_team="DAL",
            predicted_home_win_prob=0.54,
            predicted_spread=-1.0,
            actual_home_win=False,
            actual_home_score=20,
            actual_away_score=24,
            actual_spread=4.0,
            market_home_win_prob=0.50,
        ),
        # Edge 0.10 (5%+), correct
        PredictionRecord(
            game_id="3",
            week=2,
            season=2024,
            home_team="BUF",
            away_team="MIA",
            predicted_home_win_prob=0.70,
            predicted_spread=-5.0,
            actual_home_win=True,
            actual_home_score=28,
            actual_away_score=24,
            actual_spread=-4.0,
            market_home_win_prob=0.60,
        ),
    ]

    acc_1_3, acc_3_5, acc_5_plus = metrics.calculate_edge_bucket_accuracy(predictions)

    assert acc_1_3 == pytest.approx(1.0, rel=1e-6)  # 1/1
    assert acc_3_5 == pytest.approx(0.0, rel=1e-6)  # 0/1
    assert acc_5_plus == pytest.approx(1.0, rel=1e-6)  # 1/1


def test_calculate_ats_when_disagree():
    """Test ATS calculation when model disagrees with market."""
    predictions = [
        # Model picks home, market picks away, home covers
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.65,
            predicted_spread=-3.5,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=20,
            actual_spread=-7.0,
            market_home_win_prob=0.40,
            market_spread=3.0,
        ),
        # Model picks away, market picks home, away covers
        PredictionRecord(
            game_id="2",
            week=1,
            season=2024,
            home_team="SF",
            away_team="DAL",
            predicted_home_win_prob=0.30,
            predicted_spread=5.0,
            actual_home_win=False,
            actual_home_score=17,
            actual_away_score=28,
            actual_spread=11.0,
            market_home_win_prob=0.55,
            market_spread=-3.0,
        ),
    ]

    record, win_rate = metrics.calculate_ats_when_disagree(predictions)

    assert record == "2-0-0"
    assert win_rate == pytest.approx(1.0, rel=1e-6)


def test_calculate_ats_when_disagree_no_disagreements():
    """Test ATS when model always agrees with market."""
    predictions = [
        PredictionRecord(
            game_id="1",
            week=1,
            season=2024,
            home_team="KC",
            away_team="DET",
            predicted_home_win_prob=0.65,
            predicted_spread=-3.5,
            actual_home_win=True,
            actual_home_score=27,
            actual_away_score=24,
            actual_spread=-3.0,
            market_home_win_prob=0.60,  # Both pick home
        ),
    ]

    record, win_rate = metrics.calculate_ats_when_disagree(predictions)

    assert record is None
    assert win_rate is None
