#!/usr/bin/env python3
"""
Train a baseline WinLoss model and save the artifact.

Usage:
    # With database (real data):
    DATABASE_URL=postgresql://... python scripts/train_baseline.py

    # Without database (synthetic data for testing):
    python scripts/train_baseline.py --synthetic

    # Specify seasons:
    python scripts/train_baseline.py --seasons 2020-2024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.win_loss_model import WinLossModel
from src.models.model_registry import ModelRegistry
from src.features.feature_config import FEATURE_VERSION

# The canonical feature columns the model is trained on.
TRAINING_FEATURES = [
    "points_scored_avg_3",
    "points_allowed_avg_3",
    "off_yards_per_play_avg_3",
    "def_yards_per_play_avg_3",
    "turnover_diff_avg_3",
    "points_scored_avg_5",
    "points_allowed_avg_5",
    "current_streak",
    "win_pct_last_5",
    "is_division_game",
]


def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic NFL-like game data for training."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame(
        {
            "points_scored_avg_3": rng.uniform(14, 35, n),
            "points_allowed_avg_3": rng.uniform(14, 30, n),
            "off_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
            "def_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
            "turnover_diff_avg_3": rng.uniform(-2.0, 2.0, n),
            "points_scored_avg_5": rng.uniform(14, 35, n),
            "points_allowed_avg_5": rng.uniform(14, 30, n),
            "current_streak": rng.randint(-5, 6, n).astype(float),
            "win_pct_last_5": rng.uniform(0.0, 1.0, n),
            "is_division_game": rng.choice([0, 1], n).astype(float),
        }
    )
    # Target: teams that score more and allow less tend to win
    score_diff = data["points_scored_avg_3"] - data["points_allowed_avg_3"]
    prob = 1 / (1 + np.exp(-0.15 * score_diff))
    data["won"] = rng.binomial(1, prob)
    return data


def build_features_from_db(seasons: list[int]) -> pd.DataFrame:
    """Build features from the real database."""
    from src.core.db_reader import get_read_session
    from src.features.feature_builder import FeatureBuilder

    all_features = []
    with get_read_session() as session:
        builder = FeatureBuilder(session)
        for season in seasons:
            features = builder.build_and_compute(season)
            if not features.empty:
                all_features.append(features)
                print(f"  Season {season}: {len(features)} rows")

    if not all_features:
        raise RuntimeError("No feature data found in database")

    combined = pd.concat(all_features, ignore_index=True)
    return combined


def train_and_save(
    use_synthetic: bool = False,
    seasons: list[int] | None = None,
    variant: str = "baseline",
    output_dir: str = "model_artifacts",
) -> str:
    """Train a model and save the artifact. Returns model_id."""
    if seasons is None:
        seasons = list(range(2020, 2025))

    print(f"Training {variant} model...")

    if use_synthetic:
        print("  Using synthetic data")
        data = generate_synthetic_data(500)
        available_cols = [c for c in TRAINING_FEATURES if c in data.columns]
        X = data[available_cols]
        y = data["won"]
    else:
        print(f"  Building features from DB for seasons {seasons}")
        features = build_features_from_db(seasons)
        available_cols = [c for c in TRAINING_FEATURES if c in features.columns]
        if len(available_cols) < 3:
            raise RuntimeError(
                f"Only {len(available_cols)} training features found. "
                f"Available: {list(features.columns)}"
            )
        X = features[available_cols].fillna(0)
        y = features["won"] if "won" in features.columns else features.iloc[:, -1]

    # Time-based split: last 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  Features: {list(available_cols)}")

    # Train
    model = WinLossModel(model_variant=variant)
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"  Metrics: {metrics}")

    # Register and save
    registry = ModelRegistry(registry_dir=output_dir)
    model_id = registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression" if variant == "baseline" else variant,
        version="1.0.0",
        hyperparameters=(
            model.model.get_params() if hasattr(model.model, "get_params") else {}
        ),
        feature_version=FEATURE_VERSION,
        train_seasons=seasons,
        test_seasons=[seasons[-1]] if seasons else [],
        metrics=metrics,
        feature_names=available_cols,
        notes="synthetic" if use_synthetic else "db",
    )

    artifact_path = Path(output_dir) / f"{model_id}.joblib"
    model.save(str(artifact_path))
    print(f"  Saved artifact: {artifact_path}")
    print(f"  Model ID: {model_id}")

    return model_id


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--seasons", default="2020-2024", help="Season range (e.g. 2020-2024)"
    )
    parser.add_argument("--variant", default="baseline", help="Model variant")
    parser.add_argument(
        "--output-dir", default="model_artifacts", help="Output directory"
    )
    args = parser.parse_args()

    start, end = args.seasons.split("-")
    seasons = list(range(int(start), int(end) + 1))

    model_id = train_and_save(
        use_synthetic=args.synthetic,
        seasons=seasons,
        variant=args.variant,
        output_dir=args.output_dir,
    )
    print(f"\nDone. Model ID: {model_id}")


if __name__ == "__main__":
    main()
