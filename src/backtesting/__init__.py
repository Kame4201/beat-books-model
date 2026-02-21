"""
Backtesting framework for NFL game prediction models.

Walk-forward validation with expanding window:
- Train on all data up to week N
- Predict week N+1
- Evaluate and simulate betting
- Step forward and repeat

Key components:
- Backtester: Main walk-forward engine
- BankrollTracker: Track bankroll and betting performance
- Metrics: Evaluation functions (accuracy, ROI, Sharpe, etc.)
- Types: Data structures (BacktestConfig, BacktestResult, etc.)
"""

from src.backtesting.backtester import Backtester, load_backtest_result
from src.backtesting.bankroll_sim import (
    BankrollTracker,
    calculate_bet_size,
    should_place_bet,
    calculate_kelly_criterion,
)
from src.backtesting.types import (
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    PredictionRecord,
    BetSizingMethod,
)

__all__ = [
    "Backtester",
    "BankrollTracker",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetrics",
    "PredictionRecord",
    "BetSizingMethod",
    "calculate_bet_size",
    "should_place_bet",
    "calculate_kelly_criterion",
    "load_backtest_result",
]
