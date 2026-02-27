import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.config import settings
from src.service.model_manager import ModelManager, ModelNotFoundError

logger = logging.getLogger(__name__)

app = FastAPI(title="beat-books-model", version=settings.VERSION)

# Lazy-loaded model manager (loads on first prediction request)
_manager = ModelManager(settings.MODEL_ARTIFACTS_PATH)


# --- Response schemas ---


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    season: int = 2023
    week: int = 0


class PredictionDetail(BaseModel):
    winner: str
    win_probability: float
    predicted_spread: float
    confidence: str


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    prediction: PredictionDetail
    model_version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    features_used: int
    training_date: str | None
    accuracy: float | None


# --- Helpers ---


def _confidence_bucket(prob: float) -> str:
    """Map win probability to a confidence label."""
    diff = abs(prob - 0.5)
    if diff >= 0.20:
        return "high"
    if diff >= 0.10:
        return "medium"
    return "low"


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="beat-books-model",
        version=settings.VERSION,
    )


@app.post(
    "/predictions/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = _manager.model
        meta = _manager.model_meta
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Build features
    try:
        from src.core.db_reader import engine
        from src.service.features_for_inference import build_inference_features

        X = build_inference_features(
            engine=engine,
            home_team=request.home_team,
            away_team=request.away_team,
            season=request.season,
            expected_features=model.feature_names,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.warning("Feature build failed, using zero features: %s", e)
        # Fallback: zero-filled features (still returns a prediction, just less useful)
        import pandas as pd

        X = pd.DataFrame([{f: 0.0 for f in model.feature_names}])

    # Predict
    proba = model.predict_proba(X)
    home_win_prob = float(proba[0, 1])
    winner = request.home_team if home_win_prob >= 0.5 else request.away_team

    return PredictionResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        prediction=PredictionDetail(
            winner=winner,
            win_probability=round(home_win_prob, 4),
            predicted_spread=0.0,  # spread model not yet trained
            confidence=_confidence_bucket(home_win_prob),
        ),
        model_version=meta.get("version", settings.VERSION),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        meta = _manager.model_meta
    except ModelNotFoundError:
        return ModelInfoResponse(
            model_type="none",
            model_version=settings.VERSION,
            features_used=0,
            training_date=None,
            accuracy=None,
        )

    return ModelInfoResponse(
        model_type=meta.get("model_type", "unknown"),
        model_version=meta.get("version", settings.VERSION),
        features_used=len(meta.get("feature_names", [])),
        training_date=meta.get("train_date"),
        accuracy=meta.get("metrics", {}).get("accuracy"),
    )
