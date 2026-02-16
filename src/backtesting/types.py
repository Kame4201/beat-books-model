"""
Data structures for backtesting framework.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class BetSizingMethod(str, Enum):
    """Bet sizing methods."""
    FLAT = "flat"
    PERCENTAGE = "percentage"
    KELLY = "kelly"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    # Model configuration
    model_id: str
    feature_version: str

    # Time window configuration
    start_date: str  # YYYY-MM-DD or season-week format
    end_date: str
    initial_training_weeks: int = 17  # ~1 full NFL season
    step_size: int = 1  # weeks to step forward

    # Bankroll configuration
    starting_bankroll: float = 10000.0
    bet_sizing_method: BetSizingMethod = BetSizingMethod.KELLY
    kelly_fraction: float = 0.25  # fractional Kelly (conservative)
    flat_bet_size: Optional[float] = None  # for FLAT method
    percentage_bet_size: Optional[float] = None  # for PERCENTAGE method (e.g., 0.02 = 2%)

    # Betting constraints
    min_edge: float = 0.01  # only bet if edge > 1%
    max_bet_fraction: float = 0.05  # never bet more than 5% of bankroll
    min_bet_size: float = 10.0
    max_bet_size: float = 500.0

    # Output configuration
    save_predictions: bool = True
    results_dir: str = "backtest_results"


@dataclass
class PredictionRecord:
    """Single prediction made during backtesting."""

    game_id: str
    week: int
    season: int
    home_team: str
    away_team: str

    # Predictions
    predicted_home_win_prob: float
    predicted_spread: float  # negative = home favored

    # Actual outcomes
    actual_home_win: bool
    actual_home_score: int
    actual_away_score: int
    actual_spread: float

    # Market odds (if available)
    market_home_win_prob: Optional[float] = None
    market_spread: Optional[float] = None
    closing_line: Optional[float] = None

    # Betting decision
    bet_placed: bool = False
    bet_type: Optional[str] = None  # "moneyline" or "spread"
    bet_amount: float = 0.0
    bet_on_home: Optional[bool] = None

    # Betting result
    bet_won: Optional[bool] = None
    bet_payout: float = 0.0
    bet_profit: float = 0.0

    # Bankroll at time of bet
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0


@dataclass
class BacktestMetrics:
    """Evaluation metrics for a backtest run."""

    # Prediction accuracy
    overall_accuracy: float
    home_accuracy: float
    away_accuracy: float
    favorite_accuracy: float
    underdog_accuracy: float

    # Probability calibration
    log_loss: float
    brier_score: float

    # Betting performance
    total_bets: int
    total_wagered: float
    total_profit: float
    roi: float  # (total_profit / total_wagered) * 100
    win_rate: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Closing Line Value (if available)
    clv: Optional[float] = None
    clv_wins: Optional[int] = None
    clv_total: Optional[int] = None

    # Win rate by edge bucket
    edge_1_3_pct_accuracy: Optional[float] = None
    edge_3_5_pct_accuracy: Optional[float] = None
    edge_5_plus_pct_accuracy: Optional[float] = None

    # Against The Spread when model disagrees with market
    ats_record_when_disagree: Optional[str] = None  # "W-L-P" format
    ats_win_rate_when_disagree: Optional[float] = None


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    # Metadata
    run_id: str
    timestamp: str
    config: BacktestConfig

    # Date range actually tested
    actual_start_date: str
    actual_end_date: str
    total_games: int

    # Metrics
    metrics: BacktestMetrics

    # Predictions (if saved)
    predictions: List[PredictionRecord] = field(default_factory=list)

    # Bankroll curve (date -> bankroll value)
    bankroll_curve: Dict[str, float] = field(default_factory=dict)

    # Additional metadata
    model_path: Optional[str] = None
    feature_config: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": {
                "model_id": self.config.model_id,
                "feature_version": self.config.feature_version,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "initial_training_weeks": self.config.initial_training_weeks,
                "step_size": self.config.step_size,
                "starting_bankroll": self.config.starting_bankroll,
                "bet_sizing_method": self.config.bet_sizing_method.value,
                "kelly_fraction": self.config.kelly_fraction,
                "min_edge": self.config.min_edge,
                "max_bet_fraction": self.config.max_bet_fraction,
            },
            "actual_start_date": self.actual_start_date,
            "actual_end_date": self.actual_end_date,
            "total_games": self.total_games,
            "metrics": {
                "overall_accuracy": self.metrics.overall_accuracy,
                "home_accuracy": self.metrics.home_accuracy,
                "away_accuracy": self.metrics.away_accuracy,
                "favorite_accuracy": self.metrics.favorite_accuracy,
                "underdog_accuracy": self.metrics.underdog_accuracy,
                "log_loss": self.metrics.log_loss,
                "brier_score": self.metrics.brier_score,
                "total_bets": self.metrics.total_bets,
                "total_wagered": self.metrics.total_wagered,
                "total_profit": self.metrics.total_profit,
                "roi": self.metrics.roi,
                "win_rate": self.metrics.win_rate,
                "max_drawdown": self.metrics.max_drawdown,
                "max_drawdown_pct": self.metrics.max_drawdown_pct,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "clv": self.metrics.clv,
                "edge_1_3_pct_accuracy": self.metrics.edge_1_3_pct_accuracy,
                "edge_3_5_pct_accuracy": self.metrics.edge_3_5_pct_accuracy,
                "edge_5_plus_pct_accuracy": self.metrics.edge_5_plus_pct_accuracy,
                "ats_record_when_disagree": self.metrics.ats_record_when_disagree,
                "ats_win_rate_when_disagree": self.metrics.ats_win_rate_when_disagree,
            },
            "bankroll_curve": self.bankroll_curve,
            "total_predictions": len(self.predictions),
            "model_path": self.model_path,
            "notes": self.notes,
        }
