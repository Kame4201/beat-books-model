import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.core.config import settings
from src.models.model_manager import ModelManager, ModelNotFoundError

logger = logging.getLogger(__name__)

app = FastAPI(title="beat-books-model", version=settings.VERSION)

# Global model manager — lazy-loaded on first prediction
_manager: Optional[ModelManager] = None


def _get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager(
            artifacts_path=settings.MODEL_ARTIFACTS_PATH,
            model_id=os.environ.get("MODEL_ID"),
        )
    return _manager


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
    season: int
    week: int


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


class GatewayPredictionResponse(BaseModel):
    """Response shape expected by beat-books-api gateway."""

    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_spread: float
    model_version: str
    feature_version: str
    edge_vs_market: float
    recommended_bet_size: float
    bet_recommendation: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    features_used: int
    training_date: str | None
    accuracy: float | None


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
    """Predict game outcome using trained model artifact."""
    return _run_prediction(
        home_team=request.home_team,
        away_team=request.away_team,
        season=request.season,
    )


@app.get("/predict", response_model=GatewayPredictionResponse)
def predict_gateway(
    team1: str = Query(..., description="Home team"),
    team2: str = Query(..., description="Away team"),
    season: int = Query(default=2024, description="NFL season year"),
) -> GatewayPredictionResponse:
    """
    GET endpoint for the API gateway (beat-books-api).

    The gateway calls GET /predict?team1=...&team2=...
    """
    result = _run_prediction(home_team=team1, away_team=team2, season=season)
    prob = result.prediction.win_probability

    manager = _get_manager()
    try:
        info = manager.get_model_info()
        feature_version = info.get("feature_version", "v1.0")
    except ModelNotFoundError:
        feature_version = "v1.0"

    return GatewayPredictionResponse(
        home_team=result.home_team,
        away_team=result.away_team,
        home_win_probability=prob,
        away_win_probability=round(1.0 - prob, 4),
        predicted_spread=result.prediction.predicted_spread,
        model_version=result.prediction.confidence,
        feature_version=feature_version,
        edge_vs_market=0.0,
        recommended_bet_size=0.0,
        bet_recommendation="NO_BET",
    )


def _run_prediction(
    home_team: str,
    away_team: str,
    season: int,
) -> PredictionResponse:
    """Core prediction logic shared by POST and GET endpoints."""
    manager = _get_manager()

    try:
        model = manager.get_model()
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    feature_names = manager.get_feature_names()

    # Build inference features
    try:
        features_df = _build_features(home_team, away_team, season, feature_names)
    except Exception as e:
        logger.warning("DB feature build failed, using synthetic: %s", e)
        from src.features.inference_features import build_inference_features_synthetic

        features_df = build_inference_features_synthetic(
            home_team, away_team, feature_names
        )

    # Predict
    win_prob = float(model.get_win_probabilities(features_df)[0])
    predicted_spread = round((0.5 - win_prob) * 14.0, 1)  # rough spread estimate

    # Confidence bucket
    if abs(win_prob - 0.5) > 0.15:
        confidence = "high"
    elif abs(win_prob - 0.5) > 0.07:
        confidence = "medium"
    else:
        confidence = "low"

    winner = home_team if win_prob >= 0.5 else away_team

    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        prediction=PredictionDetail(
            winner=winner,
            win_probability=round(win_prob, 4),
            predicted_spread=predicted_spread,
            confidence=confidence,
        ),
        model_version=settings.VERSION,
    )


def _build_features(
    home_team: str,
    away_team: str,
    season: int,
    feature_names: list[str],
):
    """Try to build features from the database."""
    from src.core.db_reader import SessionLocal
    from src.features.inference_features import build_inference_features

    session = SessionLocal()
    try:
        return build_inference_features(
            session, home_team, away_team, season, feature_names
        )
    finally:
        session.close()


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    manager = _get_manager()
    try:
        info = manager.get_model_info()
        return ModelInfoResponse(
            model_type=info.get("model_type", "unknown"),
            model_version=info.get("version", settings.VERSION),
            features_used=len(info.get("feature_names", [])),
            training_date=info.get("train_date"),
            accuracy=info.get("metrics", {}).get("accuracy"),
        )
    except ModelNotFoundError:
        return ModelInfoResponse(
            model_type="none",
            model_version=settings.VERSION,
            features_used=0,
            training_date=None,
            accuracy=None,
        )
