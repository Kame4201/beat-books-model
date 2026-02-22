"""Entry point for: python -m src.features"""

import argparse
import sys

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_store import FeatureStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute NFL prediction features")
    parser.add_argument("--season", type=int, required=True, help="NFL season year")
    parser.add_argument("--weeks", type=str, default=None, help="Comma-separated weeks (default: all)")
    parser.add_argument("--save", action="store_true", help="Save features to feature store")
    args = parser.parse_args()

    weeks = [int(w) for w in args.weeks.split(",")] if args.weeks else None

    engineer = FeatureEngineer()
    features = engineer.compute_features(args.season, weeks)

    if features.empty:
        print(f"No features computed for season {args.season}")
        return 0

    print(f"Computed {len(features)} rows, {len(features.columns)} columns")

    if args.save:
        store = FeatureStore()
        store.save(features, description=f"Season {args.season}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
