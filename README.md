# beat-books-model

NFL game prediction engine for the BeatTheBooks platform.

## What This Will Do

- **Feature Engineering**: Transform raw NFL stats into predictive features (rolling averages, efficiency metrics, situational factors)
- **ML Models**: Predict game outcomes (Win/Loss classification + Point Spread regression)
- **Backtesting**: Walk-forward validation to evaluate model performance historically
- **Bet Sizing**: Kelly Criterion for optimal bankroll management

## Status

Under Construction — This repo is being built. See the issues for the development roadmap.

## Tech Stack

- Python 3.11+
- scikit-learn, XGBoost, LightGBM
- pandas, NumPy
- SQLAlchemy (read-only DB access)

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your DATABASE_URL (same DB as beat-books-data)
pytest
```

## Related Repos

- [beat-books-data](https://github.com/Kame4201/beat-books-data) — Data ingestion & storage
- [beat-books-api](https://github.com/Kame4201/beat-books-api) — API gateway
- [beat-books-infra](https://github.com/Kame4201/beat-books-infra) — CI/CD, docs, Docker
