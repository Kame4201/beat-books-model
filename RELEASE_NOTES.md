## Release: Dev → main

### Summary

This release brings the model service from scaffold to production-ready inference, plus comprehensive feature engineering, backtesting, and testing infrastructure.

**42 files changed, +4479 / -308 lines vs main**

### Highlights

#### Training & Inference (PR #51)
- `scripts/train_baseline.py --synthetic` produces `.joblib` artifact + model registry entry
- `scripts/train_baseline.py --seasons 2020-2024` trains from real DB data
- `ModelManager` loads best registered model on startup; returns HTTP 503 if no model found
- `POST /predictions/predict` performs actual model inference (no longer hardcoded 0.50)
- `GET /predict` gateway endpoint matching beat-books-api contract

#### Feature Engineering (PRs #37, #38, #39, #41–#48)
- Database-connected feature pipeline with zero look-ahead bias (`src/features/feature_builder.py`)
- Weather data features (`src/features/weather.py`)
- Injury/roster data features (`src/features/injuries.py`)
- News/public sentiment features (`src/features/sentiment.py`)
- Advanced stats provider stubs — NGS/PFF (`src/features/advanced_stats.py`)

#### Backtesting & Strategy
- Spread prediction integration in backtesting pipeline (PR #39)
- Backtest visualization and reporting dashboard (PR #46)
- Model comparison and hyperparameter tuning framework (PR #43)
- End-to-end pipeline orchestration CLI (PR #38)

#### Testing & CI
- Test coverage raised to 70%+ threshold (323 tests passing, PR #42)
- Integration test suite added
- `httpx` added to dev deps (FastAPI TestClient requirement)
- All mypy, ruff, black, and security checks green

### How to Test (Local)

```bash
# 1. Train a synthetic baseline model
python3 scripts/train_baseline.py --synthetic

# 2. Start the model service
DATABASE_URL="postgresql://test:test@localhost/test" uvicorn src.main:app --port 8002

# 3. Test POST prediction endpoint (real inference — NOT hardcoded 0.50)
curl -X POST http://localhost:8002/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"KC","away_team":"SF","season":2024,"week":1}'

# 4. Test GET gateway endpoint
curl "http://localhost:8002/predict?team1=KC&team2=SF&season=2024&week=1"

# 5. Check model info
curl http://localhost:8002/model/info

# 6. Run full test suite
pytest -m "not integration" --cov=src --cov-report=term
```

### Infra Dependency

For container-based E2E testing, **beat-books-infra PR #66** is required:
- Mounts `model_artifacts/` volume into the model-service container
- Wires `MODEL_ARTIFACTS_PATH` and `DATABASE_URL` environment variables
- Updates smoke tests for the new prediction contract

### PRs Included
| PR | Title |
|----|-------|
| #51 | feat: training script + real prediction endpoint |
| #48 | feat: #27 sentiment data (Phase 2 stubs) |
| #47 | feat: #23 advanced stats (Phase 2 stubs) |
| #46 | feat: #25 backtest visualization |
| #45 | feat: #22 injury/roster data features |
| #44 | feat: #21 weather data features |
| #43 | feat: #24 model comparison & tuning |
| #42 | feat: #26 test coverage to 70%+ |
| #41 | feat: baseline models |
| #39 | feat: #20 spread backtesting |
| #38 | feat: #19 pipeline CLI |
| #37 | feat: #18 DB features |

### Known Exclusions
None.
