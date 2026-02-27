#!/usr/bin/env python3
"""
Train a baseline Win/Loss classifier using season stats from the database.

Usage:
    python scripts/train_baseline.py --train-seasons 2018-2022 --test-season 2023
    python scripts/train_baseline.py  # defaults: train 2018-2022, test 2023

Requires:
    DATABASE_URL set in .env or environment pointing to beat-books-data DB.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.db_reader import engine
from src.models.model_registry import ModelRegistry
from src.models.win_loss_model import WinLossModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Features we diff between teams (team1 - team2)
FEATURE_COLUMNS = [
    "pf",       # points for
    "pa",       # points against (from standings)
    "yds_off",  # offensive yards
    "yds_def",  # defensive yards allowed
    "to_off",   # turnovers committed (offense)
    "to_def",   # turnovers forced (defense)
    "mov",      # margin of victory
    "srs",      # simple rating system
    "osrs",     # offensive SRS
    "dsrs",     # defensive SRS
    "win_pct",  # win percentage
]


def load_season_stats(season: int) -> pd.DataFrame:
    """Load team stats for a season by joining standings + offense + defense."""
    query = text("""
        SELECT
            s.tm,
            s.pf,
            s.pa,
            s.mov,
            s.srs,
            s.osrs,
            s.dsrs,
            s.win_pct,
            COALESCE(o.yds, 0) AS yds_off,
            COALESCE(o.turnovers, 0) AS to_off,
            COALESCE(d.yds, 0) AS yds_def,
            COALESCE(d.turnovers, 0) AS to_def
        FROM standings s
        LEFT JOIN team_offense o ON s.tm = o.tm AND s.season = o.season
        LEFT JOIN team_defense d ON s.tm = d.tm AND s.season = d.season
        WHERE s.season = :season
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"season": season})
    return df


def load_games(season: int) -> pd.DataFrame:
    """Load game results for a season."""
    query = text("""
        SELECT winner, loser, pts_w, pts_l
        FROM games
        WHERE season = :season AND winner IS NOT NULL AND loser IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"season": season})
    return df


def build_dataset(seasons: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset: for each game, create two rows (winner perspective + loser perspective)
    with feature diffs (team1_stats - team2_stats).

    Returns (X, y) where y=1 means team1 won.
    """
    all_X = []
    all_y = []

    for season in seasons:
        stats = load_season_stats(season)
        games = load_games(season)

        if stats.empty or games.empty:
            logger.warning(f"Season {season}: no data (stats={len(stats)}, games={len(games)})")
            continue

        stats_dict = {row["tm"]: row for _, row in stats.iterrows()}

        for _, game in games.iterrows():
            winner = game["winner"]
            loser = game["loser"]

            if winner not in stats_dict or loser not in stats_dict:
                continue

            w_stats = stats_dict[winner]
            l_stats = stats_dict[loser]

            # Row 1: winner as team1 -> label=1
            row1 = {}
            for col in FEATURE_COLUMNS:
                w_val = w_stats.get(col, 0) or 0
                l_val = l_stats.get(col, 0) or 0
                row1[f"diff_{col}"] = float(w_val) - float(l_val)
            all_X.append(row1)
            all_y.append(1)

            # Row 2: loser as team1 -> label=0
            row2 = {}
            for col in FEATURE_COLUMNS:
                w_val = w_stats.get(col, 0) or 0
                l_val = l_stats.get(col, 0) or 0
                row2[f"diff_{col}"] = float(l_val) - float(w_val)
            all_X.append(row2)
            all_y.append(0)

        logger.info(f"Season {season}: {len(games)} games -> {len(games)*2} rows")

    if not all_X:
        raise ValueError("No training data found. Check DATABASE_URL and that data is scraped.")

    return pd.DataFrame(all_X), pd.Series(all_y)


def parse_seasons(s: str) -> list[int]:
    """Parse '2018-2022' or '2018,2019,2020' into list of ints."""
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Train baseline Win/Loss model")
    parser.add_argument("--train-seasons", default="2018-2022", help="Training seasons (e.g. 2018-2022)")
    parser.add_argument("--test-season", default="2023", help="Test season (e.g. 2023)")
    args = parser.parse_args()

    train_seasons = parse_seasons(args.train_seasons)
    test_seasons = parse_seasons(args.test_season)

    if not settings.DATABASE_URL:
        logger.error("DATABASE_URL not set. Copy .env.example to .env and configure it.")
        sys.exit(1)

    logger.info(f"Training on seasons: {train_seasons}")
    logger.info(f"Testing on seasons: {test_seasons}")

    # Build datasets
    logger.info("Building training dataset...")
    X_train, y_train = build_dataset(train_seasons)
    logger.info(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")

    logger.info("Building test dataset...")
    X_test, y_test = build_dataset(test_seasons)
    logger.info(f"Test set: {len(X_test)} samples")

    # Train
    model = WinLossModel(model_variant="baseline", version="1.0.0")
    model.train(X_train, y_train)
    logger.info("Model trained successfully")

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {metrics}")

    # Register and save
    registry = ModelRegistry(settings.MODEL_ARTIFACTS_PATH)
    model_id = registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters=model.hyperparameters,
        feature_version="1.0.0",
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        metrics=metrics,
        feature_names=list(X_train.columns),
        notes=f"Baseline logistic regression. Train: {train_seasons}, Test: {test_seasons}",
    )

    artifact_path = Path(settings.MODEL_ARTIFACTS_PATH) / f"{model_id}.joblib"
    model.save(str(artifact_path))
    logger.info(f"Model saved: {artifact_path}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Registry: {settings.MODEL_ARTIFACTS_PATH}/registry.json")

    print(f"\nDone! Model ID: {model_id}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Log Loss: {metrics['log_loss']:.3f}")


if __name__ == "__main__":
    main()
