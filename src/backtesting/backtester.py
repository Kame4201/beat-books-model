"""
Walk-forward backtesting engine for NFL game predictions.

This module implements expanding window walk-forward validation:
1. Train on all data up to week N
2. Predict week N+1
3. Compare predictions to actual outcomes
4. Step forward and repeat

CRITICAL: No data leakage - features and training data only include past games.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Protocol
import pandas as pd
import numpy as np

from src.backtesting.types import (
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    PredictionRecord,
)
from src.backtesting.metrics import (
    calculate_accuracy,
    calculate_home_away_accuracy,
    calculate_favorite_underdog_accuracy,
    calculate_log_loss,
    calculate_brier_score,
    calculate_roi,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_clv,
    calculate_edge_bucket_accuracy,
    calculate_ats_when_disagree,
)
from src.backtesting.bankroll_sim import (
    BankrollTracker,
    calculate_bet_size,
    should_place_bet,
    calculate_bet_payout,
    american_odds_to_probability,
)


class Model(Protocol):
    """Protocol for models used in backtesting."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        ...


class Backtester:
    """
    Walk-forward backtesting engine.

    This class orchestrates the entire backtesting process:
    1. Load historical game data
    2. Walk forward through time
    3. Train model on expanding window
    4. Make predictions on next period
    5. Evaluate predictions and simulate betting
    6. Track metrics and bankroll
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.predictions: List[PredictionRecord] = []
        self.bankroll_tracker = BankrollTracker(config.starting_bankroll)

    def run(
        self,
        data: pd.DataFrame,
        model: Model,
        feature_columns: List[str],
        target_column: str = "home_win",
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            data: Historical game data with features and outcomes
                  Must have columns: season, week, game_id, home_team, away_team,
                                    home_score, away_score, home_win, spread
                  Plus all feature_columns
            model: Model instance with fit() and predict_proba() methods
            feature_columns: List of feature column names
            target_column: Target column name (default: "home_win")

        Returns:
            BacktestResult with all metrics and predictions
        """
        # Validate data
        self._validate_data(data, feature_columns, target_column)

        # Sort by season and week (critical for no data leakage)
        data = data.sort_values(["season", "week"]).reset_index(drop=True)

        # Get unique (season, week) combinations
        time_periods = data[["season", "week"]].drop_duplicates().sort_values(["season", "week"])

        # Initialize tracking
        run_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total_periods = len(time_periods)

        print(f"Starting backtest: {run_id}")
        print(f"Total time periods: {total_periods}")
        print(f"Initial training weeks: {self.config.initial_training_weeks}")

        # Walk forward through time
        for i in range(self.config.initial_training_weeks, total_periods, self.config.step_size):
            train_periods = time_periods.iloc[:i]
            test_period = time_periods.iloc[i:i + self.config.step_size]

            if test_period.empty:
                break

            # Get training and test data
            train_data = self._get_data_for_periods(data, train_periods)
            test_data = self._get_data_for_periods(data, test_period)

            if train_data.empty or test_data.empty:
                continue

            # Train model
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            print(f"Training on {len(train_data)} games, predicting {len(test_data)} games "
                  f"(Season {test_data['season'].iloc[0]}, Week {test_data['week'].iloc[0]})")

            model.fit(X_train, y_train)

            # Make predictions
            X_test = test_data[feature_columns]
            probas = model.predict_proba(X_test)

            # Process predictions and simulate betting
            self._process_predictions(test_data, probas, feature_columns)

        # Calculate final metrics
        metrics = self._calculate_metrics()

        # Create result object
        result = BacktestResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            actual_start_date=f"{data['season'].iloc[0]}-W{data['week'].iloc[0]}",
            actual_end_date=f"{data['season'].iloc[-1]}-W{data['week'].iloc[-1]}",
            total_games=len(data),
            metrics=metrics,
            predictions=self.predictions if self.config.save_predictions else [],
            bankroll_curve=self.bankroll_tracker.get_bankroll_curve(),
        )

        # Save results if configured
        if self.config.save_predictions:
            self._save_results(result)

        return result

    def _validate_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
    ):
        """Validate that data has required columns."""
        required = [
            "season", "week", "game_id", "home_team", "away_team",
            "home_score", "away_score", "home_win", "spread"
        ]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if target_column not in data.columns:
            raise ValueError(f"Missing target column: {target_column}")

    def _get_data_for_periods(
        self,
        data: pd.DataFrame,
        periods: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get data for specific (season, week) combinations."""
        mask = pd.Series(False, index=data.index)
        for _, row in periods.iterrows():
            mask |= (data["season"] == row["season"]) & (data["week"] == row["week"])
        return data[mask]

    def _process_predictions(
        self,
        test_data: pd.DataFrame,
        probas: np.ndarray,
        feature_columns: List[str],
    ):
        """
        Process predictions and simulate betting decisions.

        Args:
            test_data: Test data with actual outcomes
            probas: Predicted probabilities from model
            feature_columns: Feature columns used
        """
        for idx, (i, row) in enumerate(test_data.iterrows()):
            # Get prediction (assuming binary classification: [prob_away_win, prob_home_win])
            predicted_home_prob = float(probas[idx][1] if probas.ndim > 1 else probas[idx])

            # Get market data if available
            market_spread = row.get("market_spread", None)
            market_odds = row.get("market_odds", None)
            closing_line = row.get("closing_line", None)

            # Convert market odds to probability
            market_prob = None
            if market_odds is not None:
                market_prob = american_odds_to_probability(market_odds)

            # Determine if we should bet
            bet_placed = should_place_bet(
                self.config,
                predicted_home_prob,
                market_prob,
            )

            # Calculate bet size and simulate betting
            bet_amount = 0.0
            bet_profit = 0.0
            bet_won = None
            bet_on_home = None

            if bet_placed and market_odds is not None:
                # Determine which side to bet on (where we have edge)
                bet_on_home = predicted_home_prob > 0.5

                # Calculate bet size
                bet_amount = calculate_bet_size(
                    self.config,
                    self.bankroll_tracker.current_bankroll,
                    predicted_home_prob,
                    market_prob,
                    market_odds,
                )

                # Determine outcome
                actual_home_win = bool(row["home_win"])
                bet_won = bet_on_home == actual_home_win

                # Calculate payout
                _, bet_profit = calculate_bet_payout(
                    bet_amount,
                    market_odds,
                    bet_won,
                )

                # Update bankroll
                bankroll_before = self.bankroll_tracker.current_bankroll
                self.bankroll_tracker.place_bet(bet_amount, bet_profit, bet_won)
                bankroll_after = self.bankroll_tracker.current_bankroll
            else:
                bankroll_before = self.bankroll_tracker.current_bankroll
                bankroll_after = self.bankroll_tracker.current_bankroll

            # Create prediction record
            record = PredictionRecord(
                game_id=str(row["game_id"]),
                week=int(row["week"]),
                season=int(row["season"]),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                predicted_home_win_prob=predicted_home_prob,
                predicted_spread=0.0,  # TODO: implement spread prediction
                actual_home_win=bool(row["home_win"]),
                actual_home_score=int(row["home_score"]),
                actual_away_score=int(row["away_score"]),
                actual_spread=float(row["spread"]),
                market_home_win_prob=market_prob,
                market_spread=market_spread,
                closing_line=closing_line,
                bet_placed=bet_placed,
                bet_type="moneyline" if bet_placed else None,
                bet_amount=bet_amount,
                bet_on_home=bet_on_home,
                bet_won=bet_won,
                bet_payout=bet_amount + bet_profit if bet_won else 0.0,
                bet_profit=bet_profit,
                bankroll_before=bankroll_before,
                bankroll_after=bankroll_after,
            )

            self.predictions.append(record)

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate all evaluation metrics from predictions."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        # Accuracy metrics
        overall_acc = calculate_accuracy(self.predictions)
        home_acc, away_acc = calculate_home_away_accuracy(self.predictions)
        fav_acc, dog_acc = calculate_favorite_underdog_accuracy(self.predictions)

        # Probability calibration
        log_loss = calculate_log_loss(self.predictions)
        brier_score = calculate_brier_score(self.predictions)

        # Betting performance
        total_wagered, total_profit, roi = calculate_roi(self.predictions)
        bets = [p for p in self.predictions if p.bet_placed]
        win_rate = sum(1 for p in bets if p.bet_won) / len(bets) if bets else 0.0

        # Risk metrics
        max_dd, max_dd_pct = calculate_max_drawdown(
            self.predictions,
            self.config.starting_bankroll,
        )
        sharpe = calculate_sharpe_ratio(self.predictions)

        # CLV
        avg_clv, clv_wins, clv_total = calculate_clv(self.predictions)

        # Edge buckets
        edge_1_3, edge_3_5, edge_5_plus = calculate_edge_bucket_accuracy(self.predictions)

        # ATS when disagree
        ats_record, ats_win_rate = calculate_ats_when_disagree(self.predictions)

        return BacktestMetrics(
            overall_accuracy=overall_acc,
            home_accuracy=home_acc,
            away_accuracy=away_acc,
            favorite_accuracy=fav_acc,
            underdog_accuracy=dog_acc,
            log_loss=log_loss,
            brier_score=brier_score,
            total_bets=len(bets),
            total_wagered=total_wagered,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            clv=avg_clv,
            clv_wins=clv_wins,
            clv_total=clv_total,
            edge_1_3_pct_accuracy=edge_1_3,
            edge_3_5_pct_accuracy=edge_3_5,
            edge_5_plus_pct_accuracy=edge_5_plus,
            ats_record_when_disagree=ats_record,
            ats_win_rate_when_disagree=ats_win_rate,
        )

    def _save_results(self, result: BacktestResult):
        """Save backtest results to JSON."""
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)

        # Save full results
        filepath = results_dir / f"{result.run_id}.json"
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Results saved to: {filepath}")


def load_backtest_result(run_id: str, results_dir: str = "backtest_results") -> BacktestResult:
    """
    Load a previously saved backtest result.

    Args:
        run_id: Run ID of the backtest
        results_dir: Directory where results are stored

    Returns:
        BacktestResult object
    """
    filepath = Path(results_dir) / f"{run_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"No backtest result found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruct BacktestResult from dict
    # This is a simplified version - full reconstruction would need more work
    return data
