"""Entry point for: python -m src.models"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NFL prediction models")
    parser.add_argument("--model", choices=["win_loss", "spread"], required=True)
    parser.add_argument("--variant", default="baseline", choices=["baseline", "ridge", "xgboost", "lightgbm"])
    parser.add_argument("--seasons", required=True, help="Season range (e.g., '2020-2024')")
    parser.add_argument("--output-dir", default="model_artifacts", help="Where to save trained model")
    args = parser.parse_args()

    from src.pipeline.runner import parse_season_range, run_feature_stage, run_training_stage

    seasons = parse_season_range(args.seasons)
    print(f"Training {args.model}/{args.variant} on seasons {seasons}")

    features = run_feature_stage(seasons)
    if features.empty:
        print("No features available. Exiting.")
        return 1

    model = run_training_stage(features, args.model, args.variant)
    if not model.is_trained:
        print("Training failed.")
        return 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{args.model}_{args.variant}.joblib"
    model.save(str(model_path))
    print(f"Model saved: {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
