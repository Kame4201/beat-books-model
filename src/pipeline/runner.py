"""
End-to-end pipeline runner: Features → Model Training → Backtesting → Report.

Usage:
    python -m src.pipeline --seasons 2020-2024 --model win_loss --backtest-start 2023
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.features.feature_config import FEATURE_VERSION


def parse_season_range(season_str: str) -> List[int]:
    """Parse '2020-2024' into [2020, 2021, 2022, 2023, 2024]."""
    if "-" in season_str:
        start, end = season_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in season_str.split(",")]


def split_train_test(
    seasons: List[int],
    backtest_start: int,
) -> Tuple[List[int], List[int]]:
    """Split seasons into train and test based on backtest start year."""
    train = [s for s in seasons if s < backtest_start]
    test = [s for s in seasons if s >= backtest_start]
    return train, test


def run_feature_stage(
    seasons: List[int],
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute features for multiple seasons.

    Returns:
        Combined features DataFrame across all requested seasons
    """
    engineer = FeatureEngineer(rolling_windows=rolling_windows)
    all_features = []

    for season in seasons:
        print(f"  Computing features for season {season}...")
        features = engineer.compute_features(season)
        if not features.empty:
            all_features.append(features)
        else:
            print(f"  Warning: No data for season {season}")

    if not all_features:
        return pd.DataFrame()

    combined = pd.concat(all_features, ignore_index=True)
    print(f"  Total: {len(combined)} rows across {len(all_features)} seasons")
    return combined


def run_training_stage(
    train_features: pd.DataFrame,
    model_type: str = "win_loss",
    model_variant: str = "baseline",
) -> object:
    """
    Train a model on the provided features.

    Args:
        train_features: Training feature DataFrame
        model_type: "win_loss" or "spread"
        model_variant: Model variant (e.g., "baseline", "xgboost", "lightgbm")

    Returns:
        Trained model instance
    """
    if model_type == "win_loss":
        from src.models.win_loss_model import WinLossModel

        model = WinLossModel(model_variant=model_variant)
    elif model_type == "spread":
        from src.models.spread_model import SpreadModel

        model = SpreadModel(model_variant=model_variant)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # The feature engineer produces per-team-per-game rows. For training,
    # we need to pair home/away teams. Since home/away isn't in the current
    # schema, we use a simplified approach: drop identifier columns and
    # use the remaining numeric features with a target column.
    feature_cols = [
        c for c in train_features.columns
        if c not in ("game_id", "team", "season", "week", "game_date", "won", "point_diff", "opponent")
        and train_features[c].dtype in ("float64", "int64", "float32", "int32")
    ]

    if not feature_cols:
        print("  Warning: No numeric feature columns found for training")
        return model

    X = train_features[feature_cols].fillna(0)

    if model_type == "win_loss":
        if "won" not in train_features.columns:
            print("  Warning: 'won' column not found, skipping training")
            return model
        y = train_features["won"]
    else:
        if "point_diff" not in train_features.columns:
            print("  Warning: 'point_diff' column not found, skipping training")
            return model
        y = train_features["point_diff"]

    print(f"  Training {model_type}/{model_variant} on {len(X)} samples, {len(feature_cols)} features...")
    model.train(X, y)
    print("  Training complete.")
    return model


def run_pipeline(
    seasons: List[int],
    model_type: str = "win_loss",
    model_variant: str = "baseline",
    backtest_start: Optional[int] = None,
    rolling_windows: Optional[List[int]] = None,
    save_artifacts: bool = True,
    output_dir: str = "pipeline_output",
) -> dict:
    """
    Run the full prediction pipeline.

    Steps:
        1. Compute features for all requested seasons
        2. Split into train/test
        3. Train model on training seasons
        4. Evaluate on test seasons
        5. Save artifacts and report

    Returns:
        Dictionary with pipeline results
    """
    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "model_type": model_type,
        "model_variant": model_variant,
    }

    # ------------------------------------------------------------------
    # Step 1: Features
    # ------------------------------------------------------------------
    print("\n[1/4] Feature Engineering")
    all_features = run_feature_stage(seasons, rolling_windows)
    if all_features.empty:
        print("No features computed. The database may not have data for these seasons.")
        print("Pipeline stopped.")
        results["status"] = "no_data"
        return results

    results["total_rows"] = len(all_features)

    # Save features
    if save_artifacts:
        store = FeatureStore()
        store.save(all_features, description=f"Pipeline run: seasons {seasons}")

    # ------------------------------------------------------------------
    # Step 2: Train/Test Split
    # ------------------------------------------------------------------
    print("\n[2/4] Train/Test Split")
    if backtest_start:
        train_seasons, test_seasons = split_train_test(seasons, backtest_start)
    else:
        # Default: last season is test
        train_seasons = seasons[:-1]
        test_seasons = seasons[-1:]

    train_features = all_features[all_features["season"].isin(train_seasons)]
    test_features = all_features[all_features["season"].isin(test_seasons)]

    print(f"  Train: {len(train_features)} rows ({train_seasons})")
    print(f"  Test:  {len(test_features)} rows ({test_seasons})")

    results["train_seasons"] = train_seasons
    results["test_seasons"] = test_seasons

    if train_features.empty:
        print("No training data available. Pipeline stopped.")
        results["status"] = "no_training_data"
        return results

    # ------------------------------------------------------------------
    # Step 3: Model Training
    # ------------------------------------------------------------------
    print("\n[3/4] Model Training")
    model = run_training_stage(train_features, model_type, model_variant)

    if not model.is_trained:
        print("Model training failed. Pipeline stopped.")
        results["status"] = "training_failed"
        return results

    # Save model artifact
    if save_artifacts:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / f"{model_type}_{model_variant}.joblib"
        model.save(str(model_path))
        print(f"  Model saved: {model_path}")
        results["model_path"] = str(model_path)

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluation")
    if not test_features.empty:
        feature_cols = [
            c for c in test_features.columns
            if c not in ("game_id", "team", "season", "week", "game_date", "won", "point_diff", "opponent")
            and test_features[c].dtype in ("float64", "int64", "float32", "int32")
        ]

        X_test = test_features[feature_cols].fillna(0)

        if model_type == "win_loss" and "won" in test_features.columns:
            y_test = test_features["won"]
            metrics = model.evaluate(X_test, y_test)
            print("  Metrics:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")
            results["metrics"] = metrics
        elif model_type == "spread" and "point_diff" in test_features.columns:
            y_test = test_features["point_diff"]
            metrics = model.evaluate(X_test, y_test)
            print("  Metrics:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")
            results["metrics"] = metrics
        else:
            print("  Warning: No target column for evaluation")
    else:
        print("  No test data for evaluation")

    # Save results
    if save_artifacts:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_path = out / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved: {results_path}")

    results["status"] = "success"
    print("\nPipeline complete.")
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="BeatTheBooks ML Pipeline — Features → Training → Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline --seasons 2020-2024 --model win_loss
  python -m src.pipeline --seasons 2020-2024 --model spread --variant xgboost --backtest-start 2023
  python -m src.pipeline --seasons 2024 --model win_loss --no-save
        """,
    )
    parser.add_argument(
        "--seasons",
        required=True,
        help="Season range (e.g., '2020-2024') or comma-separated (e.g., '2022,2023,2024')",
    )
    parser.add_argument(
        "--model",
        choices=["win_loss", "spread"],
        default="win_loss",
        help="Model type (default: win_loss)",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "ridge", "xgboost", "lightgbm"],
        default="baseline",
        help="Model variant (default: baseline)",
    )
    parser.add_argument(
        "--backtest-start",
        type=int,
        default=None,
        help="First season for test set (default: last season in range)",
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_output",
        help="Directory for output artifacts (default: pipeline_output)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save artifacts to disk",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    seasons = parse_season_range(args.seasons)
    print(f"BeatTheBooks ML Pipeline")
    print(f"  Seasons: {seasons}")
    print(f"  Model: {args.model}/{args.variant}")
    if args.backtest_start:
        print(f"  Backtest start: {args.backtest_start}")

    results = run_pipeline(
        seasons=seasons,
        model_type=args.model,
        model_variant=args.variant,
        backtest_start=args.backtest_start,
        save_artifacts=not args.no_save,
        output_dir=args.output_dir,
    )

    return 0 if results.get("status") in ("success", "no_data") else 1


if __name__ == "__main__":
    sys.exit(main())
