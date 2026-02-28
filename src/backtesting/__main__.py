"""Entry point for: python -m src.backtesting"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NFL prediction backtests")
    parser.add_argument(
        "--model-path", required=True, help="Path to trained model (.joblib)"
    )
    parser.add_argument(
        "--start", required=True, help="Backtest start (season or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True, help="Backtest end (season or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bankroll", type=float, default=10000.0, help="Starting bankroll"
    )
    parser.add_argument(
        "--output-dir", default="backtest_results", help="Results directory"
    )
    args = parser.parse_args()

    print(f"Backtesting with model: {args.model_path}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Starting bankroll: ${args.bankroll:,.2f}")

    from src.models.base_predictor import BasePredictor

    model = BasePredictor.load(args.model_path)
    print(f"  Loaded model: {model.model_type} v{model.version}")
    print(f"  Features: {len(model.feature_names or [])} columns")

    # Full backtesting integration will be completed when #20 merges
    # the spread prediction into the backtester
    print("\nBacktesting pipeline ready. Full walk-forward execution")
    print("requires the backtester integration (see issue #20).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
