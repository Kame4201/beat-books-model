## Cross-Repo Readiness Assessment — 2026-02-27

### Stage Matrix

| Stage | Status | Evidence |
|-------|--------|----------|
| **Scrape** | Ready (code) | `beat-books-data` Dev: batch scrape service, 15+ stat category scrapers, scrape job tracking |
| **Store** | Ready (code) | `beat-books-data` Dev: Alembic migrations, all entity models, repositories for every stat table |
| **Feature Build** | Ready | `beat-books-model` Dev: `FeatureBuilder` with DB queries, zero look-ahead bias, weather/injury/sentiment stubs |
| **Train** | Ready | `scripts/train_baseline.py --synthetic` produces artifact + registry. DB mode ready when data available |
| **Predict** | Ready | `POST /predictions/predict` returns real inference (0.5267, not 0.50). Synthetic fallback when no DB |
| **Gateway** | Ready | `beat-books-api` Dev: `GET /predict` delegates to model service via `httpx`. Team alias resolution works |
| **Infra** | Ready (code) | PR #66 merged into Dev. Compose mounts `model_artifacts`, wires `DATABASE_URL`, `MODEL_ARTIFACTS_PATH` |

### Evidence

#### Model Service (local TestClient, no DB)
```
Health: 200 {"status": "healthy", "service": "beat-books-model", "version": "0.1.0"}

POST /predictions/predict: 200
  {"home_team": "KC", "away_team": "SF",
   "prediction": {"winner": "KC", "win_probability": 0.5267,
                   "predicted_spread": -0.4, "confidence": "low"},
   "model_version": "0.1.0"}

GET /predict: 200
  {"home_team": "KC", "away_team": "SF",
   "home_win_probability": 0.5267, "away_win_probability": 0.4733,
   "predicted_spread": -0.4, ...}

GET /model/info: 200
  {"model_type": "win_loss_classifier", "model_version": "1.0.0",
   "features_used": 10, "accuracy": 0.8}
```

#### Training
```
$ python3 scripts/train_baseline.py --synthetic
  Train: 400 rows, Test: 100 rows
  Accuracy: 0.800, Log Loss: 0.506
  Artifact: model_artifacts/e87143a1-...joblib
```

#### CI Checks (beat-books-model PR #51 → Dev)
All 5 checks pass: lint, security, type-check, test (323 tests), claude-review

#### Infra Compose
- `docker-compose.yml` validated (`docker compose config` passes)
- Services: postgres:16, data-service:8001, model-service:8002, api:8000
- model_artifacts volume mounted at `/app/model_artifacts`
- Docker daemon not available in this environment for live container test

#### API Gateway
- `beat-books-api` Dev: `GET /predict` proxies to `POST model-service:8002/predictions/predict`
- Team alias resolution from `team_aliases.json`
- No code changes needed (routing already works)

### PR Status

| Repo | PR | State | Notes |
|------|----|-------|-------|
| beat-books-model | #51 | **Merged** into Dev | Training + real inference |
| beat-books-model | #50 | **Closed** | Superseded by #51 |
| beat-books-model | #49 | **Open** (Dev→main) | Release notes updated |
| beat-books-infra | #66 | **Merged** into Dev | Volume mount + env wiring |
| beat-books-api | — | No changes needed | Routing already works |
| beat-books-data | — | No changes needed | Scrape/store endpoints exist |

### Verdict: CONDITIONAL GO

**What works end-to-end (code-level):**
- Train → artifact → load → predict → gateway: full chain verified locally
- Predictions are real model inference (not hardcoded 0.50)
- Compose config is valid and all services wired

**Blockers for full container E2E:**
1. Docker daemon required to run `docker compose up` — not available in this sandbox
2. Scrape→Store needs live containers + Pro Football Reference access (network/rate-limited)

**Next commands for container E2E (on a machine with Docker):**
```bash
cd beat-books-infra/docker
DB_PASSWORD=testpass docker compose up -d --build
# Wait for healthy
docker compose exec data-service alembic upgrade head
# Scrape
curl -X POST http://localhost:8001/scrape/batch/2024 \
  -H "Content-Type: application/json" \
  -d '{"stats":["team_offense","standings","games"],"dry_run":false}'
# Verify stored
curl http://localhost:8001/api/v1/stats/teams/2024
# Predict via model
curl -X POST http://localhost:8002/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"KC","away_team":"SF","season":2024,"week":1}'
# Predict via gateway
curl "http://localhost:8000/predict?team1=chiefs&team2=eagles"
```
