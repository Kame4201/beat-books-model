# CLAUDE.md — beat-books-model

## Project Overview
NFL game prediction engine for the BeatTheBooks platform. Computes features from historical data, trains ML models, runs backtests, and sizes bets using Kelly Criterion.

## Architecture
```
Features → Models → Backtesting → Strategy
```
- **Features** (`src/features/`): Transform raw NFL stats into predictive features
- **Models** (`src/models/`): Train and predict game outcomes (Win/Loss + Spread)
- **Backtesting** (`src/backtesting/`): Walk-forward validation to evaluate models historically
- **Strategy** (`src/strategy/`): Kelly Criterion bet sizing and bankroll management

## CRITICAL RULE: READ-ONLY DATABASE ACCESS
This repo NEVER creates, alters, or drops database tables.
All schema changes go through `beat-books-data` via Alembic migrations.
This repo only READS from the shared PostgreSQL database.

## Directory Structure
```
src/
├── core/          # Config, read-only DB connection
├── features/      # Feature engineering pipeline
├── models/        # ML prediction models
├── backtesting/   # Walk-forward validation framework
└── strategy/      # Kelly Criterion, bankroll management
notebooks/         # Jupyter notebooks for exploration
model_artifacts/   # Serialized trained models (gitignored)
tests/             # Unit and integration tests
```

## Rules — ALWAYS Follow
- ALWAYS use db_reader.py for database access (read-only)
- ALWAYS validate for data leakage — no future data in features or training
- ALWAYS version models and features (track in model registry)
- ALWAYS use walk-forward validation (expanding window, not random split)
- ALWAYS log experiment metrics (accuracy, ROI, log-loss, etc.)

## Rules — NEVER Do
- NEVER create, alter, or drop database tables
- NEVER import directly from beat-books-data code (use DB queries only)
- NEVER use random train/test splits (use time-based walk-forward)
- NEVER deploy a model without backtesting it first
- NEVER hardcode database URLs — use config.py

## Common Commands
```bash
# Run tests
pytest
pytest --cov=src --cov-report=html

# Feature engineering (future)
python -m src.features.feature_engineering --season 2024

# Model training (future)
python -m src.models.train --model win_loss --seasons 2020-2024

# Backtesting (future)
python -m src.backtesting.run --model win_loss --start 2022 --end 2024
```

## Database Tables Read From
- team_offense, team_defense
- passing_stats, rushing_stats, receiving_stats, defense_stats
- kicking_stats, punting_stats, return_stats, scoring_stats
- games, standings
- odds (once beat-books-data adds this)

## Related Repos
- beat-books-data: Data ingestion & DB schema (this repo reads from its tables)
- beat-books-api: API gateway (will expose prediction endpoints)
- beat-books-infra: CI/CD, docs, Docker
