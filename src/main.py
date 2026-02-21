from fastapi import FastAPI
from pydantic import BaseModel

from src.core.config import settings

app = FastAPI(title="beat-books-model", version=settings.VERSION)


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
    # TODO: Replace stub with actual model inference
    return PredictionResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        prediction=PredictionDetail(
            winner=request.home_team,
            win_probability=0.50,
            predicted_spread=0.0,
            confidence="low",
        ),
        model_version=settings.VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    # TODO: Replace stub with actual model metadata
    return ModelInfoResponse(
        model_type="stub",
        model_version=settings.VERSION,
        features_used=0,
        training_date=None,
        accuracy=None,
    )
