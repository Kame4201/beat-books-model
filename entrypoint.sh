#!/bin/bash
set -e

if [ ! -f model_artifacts/registry.json ]; then
  echo "No trained model found. Training baseline with synthetic data..."
  python scripts/train_baseline.py --synthetic
fi

echo "Starting application..."
exec "$@"
