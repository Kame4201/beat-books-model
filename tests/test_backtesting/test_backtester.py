"""
Tests for the main Backtester class.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.backtesting import Backtester, BacktestConfig, BetSizingMethod


@pytest.fixture
def sample_game_data():
    """Sample game data for testing."""
    np.random.seed(42)

    data = []
    for season in [2023, 2024]:
        for week in range(1, 18):
            for game in range(4):  # 4 games per week
                game_id = f"{season}_{week}_{game}"
                home_score = np.random.randint(14, 35)
                away_score = np.random.randint(14, 35)

                data.append({
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "home_team": f"TEAM_{game % 4}",
                    "away_team": f"TEAM_{(game + 1) % 4}",
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": home_score > away_score,
                    "spread": float(home_score - away_score),
                    # Features
                    "home_offense_rating": np.random.uniform(0.4, 0.6),
                    "away_offense_rating": np.random.uniform(0.4, 0.6),
                    "home_defense_rating": np.random.uniform(0.4, 0.6),
                    "away_defense_rating": np.random.uniform(0.4, 0.6),
                    # Market data
                    "market_spread": np.random.uniform(-7, 7),
                    "market_odds": -110,
                })

    return pd.DataFrame(data)


@pytest.fixture
def simple_config():
    """Simple backtest configuration."""
    return BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2023-W1",
        end_date="2024-W17",
        initial_training_weeks=17,
        step_size=1,
        starting_bankroll=10000.0,
        bet_sizing_method=BetSizingMethod.FLAT,
        flat_bet_size=100.0,
        min_edge=0.05,  # Only bet with 5%+ edge
    )


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()

    # Simple model: predict home win probability based on offense rating
    def predict_proba(X):
        # Return shape (n_samples, 2) for binary classification
        home_prob = X["home_offense_rating"].values
        away_prob = 1 - home_prob
        return np.column_stack([away_prob, home_prob])

    model.predict_proba = predict_proba
    model.fit = Mock()

    return model


def test_backtester_initialization(simple_config):
    """Test Backtester initialization."""
    backtester = Backtester(simple_config)
    assert backtester.config == simple_config
    assert backtester.predictions == []
    assert backtester.bankroll_tracker.current_bankroll == 10000.0


def test_backtester_validate_data(simple_config, sample_game_data):
    """Test data validation."""
    backtester = Backtester(simple_config)

    # Should not raise with valid data
    feature_cols = ["home_offense_rating", "away_offense_rating"]
    backtester._validate_data(sample_game_data, feature_cols, "home_win")


def test_backtester_validate_data_missing_required(simple_config):
    """Test data validation with missing required columns."""
    backtester = Backtester(simple_config)

    # Missing required columns
    bad_data = pd.DataFrame({
        "season": [2024],
        "week": [1],
        # Missing other required columns
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        backtester._validate_data(bad_data, ["feature1"], "home_win")


def test_backtester_validate_data_missing_features(simple_config, sample_game_data):
    """Test data validation with missing feature columns."""
    backtester = Backtester(simple_config)

    with pytest.raises(ValueError, match="Missing feature columns"):
        backtester._validate_data(sample_game_data, ["nonexistent_feature"], "home_win")


def test_backtester_get_data_for_periods(simple_config, sample_game_data):
    """Test getting data for specific time periods."""
    backtester = Backtester(simple_config)

    # Get data for week 1-3 of 2023
    periods = pd.DataFrame({
        "season": [2023, 2023, 2023],
        "week": [1, 2, 3],
    })

    result = backtester._get_data_for_periods(sample_game_data, periods)

    assert len(result) == 12  # 4 games per week * 3 weeks
    assert result["season"].unique() == [2023]
    assert set(result["week"].unique()) == {1, 2, 3}


def test_backtester_run_basic(simple_config, sample_game_data, mock_model):
    """Test basic backtest run."""
    backtester = Backtester(simple_config)

    feature_cols = [
        "home_offense_rating",
        "away_offense_rating",
        "home_defense_rating",
        "away_defense_rating",
    ]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
        target_column="home_win",
    )

    # Check result structure
    assert result.run_id.startswith("backtest_")
    assert result.config == simple_config
    assert result.total_games == len(sample_game_data)

    # Check metrics exist
    assert 0 <= result.metrics.overall_accuracy <= 1
    assert result.metrics.total_bets >= 0
    assert result.metrics.log_loss > 0

    # Check predictions were made
    # With initial_training_weeks=17 and 34 total weeks, we predict ~17 weeks
    # 17 weeks * 4 games = ~68 games
    assert len(result.predictions) > 50

    # Check bankroll curve
    assert len(result.bankroll_curve) > 0


def test_backtester_run_no_data_leakage(simple_config, sample_game_data, mock_model):
    """Test that walk-forward prevents data leakage."""
    backtester = Backtester(simple_config)

    feature_cols = ["home_offense_rating", "away_offense_rating"]

    # Run backtest
    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    # Verify predictions are only made after initial training window
    first_pred = result.predictions[0]
    assert first_pred.week > simple_config.initial_training_weeks or first_pred.season > 2023

    # Verify model was fit multiple times (once per step)
    assert mock_model.fit.call_count > 1


def test_backtester_run_with_small_step(sample_game_data, mock_model):
    """Test backtest with small step size."""
    config = BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2023-W1",
        end_date="2024-W17",
        initial_training_weeks=10,
        step_size=2,  # Step 2 weeks at a time
        bet_sizing_method=BetSizingMethod.FLAT,
        flat_bet_size=100.0,
    )

    backtester = Backtester(config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    # Should have fewer predictions than step_size=1
    assert len(result.predictions) > 0


def test_backtester_run_with_kelly_sizing(sample_game_data, mock_model):
    """Test backtest with Kelly Criterion bet sizing."""
    config = BacktestConfig(
        model_id="test_model",
        feature_version="v1",
        start_date="2023-W1",
        end_date="2024-W17",
        initial_training_weeks=17,
        bet_sizing_method=BetSizingMethod.KELLY,
        kelly_fraction=0.25,
        min_edge=0.01,  # Lower threshold to get more bets
    )

    backtester = Backtester(config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    # Check that some bets were placed
    assert result.metrics.total_bets > 0

    # Check that bet sizes vary (Kelly should adjust based on edge)
    bet_amounts = [p.bet_amount for p in result.predictions if p.bet_placed]
    if len(bet_amounts) > 1:
        # Not all bets should be the same size
        assert len(set(bet_amounts)) > 1


def test_backtester_metrics_calculation(simple_config, sample_game_data, mock_model):
    """Test that all metrics are calculated correctly."""
    backtester = Backtester(simple_config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    metrics = result.metrics

    # All metrics should be present
    assert metrics.overall_accuracy is not None
    assert metrics.home_accuracy is not None
    assert metrics.away_accuracy is not None
    assert metrics.log_loss is not None
    assert metrics.brier_score is not None
    assert metrics.total_bets is not None
    assert metrics.roi is not None
    assert metrics.max_drawdown is not None
    assert metrics.sharpe_ratio is not None

    # Metrics should be reasonable
    assert 0 <= metrics.overall_accuracy <= 1
    assert metrics.log_loss > 0
    assert 0 <= metrics.brier_score <= 1


def test_backtester_bankroll_tracking(simple_config, sample_game_data, mock_model):
    """Test that bankroll is tracked correctly."""
    backtester = Backtester(simple_config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    # Bankroll curve should start at starting_bankroll
    assert result.bankroll_curve["0"] == simple_config.starting_bankroll

    # Final bankroll should equal starting + total_profit
    final_bankroll = float(result.bankroll_curve[str(result.metrics.total_bets)])
    expected = simple_config.starting_bankroll + result.metrics.total_profit
    assert final_bankroll == pytest.approx(expected, rel=1e-6)


def test_backtester_prediction_records(simple_config, sample_game_data, mock_model):
    """Test that prediction records are complete."""
    backtester = Backtester(simple_config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    # Check first prediction has all required fields
    pred = result.predictions[0]
    assert pred.game_id is not None
    assert pred.season is not None
    assert pred.week is not None
    assert pred.home_team is not None
    assert pred.away_team is not None
    assert 0 <= pred.predicted_home_win_prob <= 1
    assert pred.actual_home_win in [True, False]
    assert pred.bankroll_before >= 0
    assert pred.bankroll_after >= 0


def test_backtester_to_dict(simple_config, sample_game_data, mock_model):
    """Test BacktestResult to_dict conversion."""
    backtester = Backtester(simple_config)
    feature_cols = ["home_offense_rating", "away_offense_rating"]

    result = backtester.run(
        data=sample_game_data,
        model=mock_model,
        feature_columns=feature_cols,
    )

    result_dict = result.to_dict()

    # Check structure
    assert "run_id" in result_dict
    assert "config" in result_dict
    assert "metrics" in result_dict
    assert "bankroll_curve" in result_dict

    # Check metrics structure
    assert "overall_accuracy" in result_dict["metrics"]
    assert "roi" in result_dict["metrics"]
    assert "sharpe_ratio" in result_dict["metrics"]
