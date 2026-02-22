"""Tests for spread prediction integration in the backtester."""

import pytest
import numpy as np
import pandas as pd

from src.backtesting.backtester import Backtester
from src.backtesting.types import BacktestConfig, PredictionRecord
from src.backtesting.metrics import calculate_spread_mae, calculate_cover_rate


# ---------------------------------------------------------------------------
# Mock Models
# ---------------------------------------------------------------------------

class MockWinLossModel:
    """Mock win/loss classifier."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


class MockSpreadModel:
    """Mock spread regressor."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._mean = y.mean()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_backtest_data():
    """Create sample data with 4 weeks of games."""
    np.random.seed(42)
    rows = []
    for season in [2023]:
        for week in range(1, 5):
            for game_idx in range(4):
                home_score = np.random.randint(10, 40)
                away_score = np.random.randint(10, 40)
                rows.append({
                    "game_id": f"{season}_{week}_{game_idx}",
                    "season": season,
                    "week": week,
                    "home_team": f"TEAM_H{game_idx}",
                    "away_team": f"TEAM_A{game_idx}",
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": int(home_score > away_score),
                    "spread": float(away_score - home_score),
                    "feature_1": np.random.randn(),
                    "feature_2": np.random.randn(),
                    "feature_3": np.random.randn(),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def backtest_config():
    return BacktestConfig(
        model_id="test",
        feature_version="v1.0",
        start_date="2023-W1",
        end_date="2023-W4",
        initial_training_weeks=2,
        step_size=1,
        starting_bankroll=10000.0,
        min_edge=0.0,
        save_predictions=False,
    )


# ---------------------------------------------------------------------------
# Tests — Spread model integration
# ---------------------------------------------------------------------------

def test_backtester_without_spread_model(backtest_config, sample_backtest_data):
    """Without spread model, predicted_spread should be 0.0."""
    bt = Backtester(backtest_config)
    result = bt.run(
        sample_backtest_data,
        MockWinLossModel(),
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    for pred in result.predictions:
        assert pred.predicted_spread == 0.0


def test_backtester_with_spread_model(backtest_config, sample_backtest_data):
    """With spread model, predicted_spread should be non-zero."""
    bt = Backtester(backtest_config)
    result = bt.run(
        sample_backtest_data,
        MockWinLossModel(),
        feature_columns=["feature_1", "feature_2", "feature_3"],
        spread_model=MockSpreadModel(),
        spread_target_column="spread",
    )
    has_nonzero = any(p.predicted_spread != 0.0 for p in result.predictions)
    assert has_nonzero


def test_spread_model_protocol():
    """SpreadModel should satisfy the SpreadCapableModel protocol."""
    model = MockSpreadModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0])
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 3


# ---------------------------------------------------------------------------
# Tests — Spread metrics
# ---------------------------------------------------------------------------

def test_spread_mae_with_predictions():
    preds = [
        PredictionRecord(
            game_id="1", week=1, season=2023,
            home_team="A", away_team="B",
            predicted_home_win_prob=0.6,
            predicted_spread=-3.0,
            actual_home_win=True,
            actual_home_score=27, actual_away_score=24,
            actual_spread=-3.0,
        ),
        PredictionRecord(
            game_id="2", week=1, season=2023,
            home_team="C", away_team="D",
            predicted_home_win_prob=0.4,
            predicted_spread=7.0,
            actual_home_win=False,
            actual_home_score=17, actual_away_score=28,
            actual_spread=11.0,
        ),
    ]
    mae = calculate_spread_mae(preds)
    # |(-3) - (-3)| + |(7) - (11)| / 2 = (0 + 4) / 2 = 2.0
    assert mae == pytest.approx(2.0)


def test_spread_mae_no_predictions():
    preds = [
        PredictionRecord(
            game_id="1", week=1, season=2023,
            home_team="A", away_team="B",
            predicted_home_win_prob=0.6,
            predicted_spread=0.0,  # no spread predicted
            actual_home_win=True,
            actual_home_score=27, actual_away_score=24,
            actual_spread=-3.0,
        ),
    ]
    assert calculate_spread_mae(preds) is None


def test_cover_rate_no_market():
    """Without market spread, cover rate should be None."""
    preds = [
        PredictionRecord(
            game_id="1", week=1, season=2023,
            home_team="A", away_team="B",
            predicted_home_win_prob=0.6,
            predicted_spread=-3.0,
            actual_home_win=True,
            actual_home_score=27, actual_away_score=24,
            actual_spread=-3.0,
            market_spread=None,
        ),
    ]
    assert calculate_cover_rate(preds) is None
