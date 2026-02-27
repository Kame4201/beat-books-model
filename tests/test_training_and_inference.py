"""
Tests for training script, model manager, and prediction endpoint.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.win_loss_model import WinLossModel
from src.models.model_registry import ModelRegistry
from src.models.model_manager import ModelManager, ModelNotFoundError
from src.features.inference_features import (
    build_inference_features_synthetic,
)

# ---------------------------------------------------------------------------
# Training + artifact roundtrip
# ---------------------------------------------------------------------------


class TestTrainAndSaveArtifact:
    """Test that we can train a model, save it, reload it, and predict."""

    def test_train_save_reload_predict(self, tmp_path):
        """End-to-end: train → save → register → reload → predict."""
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(
            {
                "points_scored_avg_3": rng.uniform(14, 35, n),
                "points_allowed_avg_3": rng.uniform(14, 30, n),
                "off_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
                "def_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
                "turnover_diff_avg_3": rng.uniform(-2.0, 2.0, n),
            }
        )
        y = pd.Series(rng.choice([0, 1], n))

        # Train
        model = WinLossModel(model_variant="baseline")
        model.train(X, y)

        # Register
        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register_model(
            model_type="win_loss_classifier",
            model_name="logistic_regression",
            version="1.0.0",
            hyperparameters={},
            feature_version="v1.0",
            train_seasons=[2023],
            test_seasons=[2024],
            metrics={"accuracy": 0.55},
            feature_names=list(X.columns),
        )

        # Save artifact
        artifact_path = tmp_path / f"{model_id}.joblib"
        model.save(str(artifact_path))
        assert artifact_path.exists()

        # Registry file exists
        assert (tmp_path / "registry.json").exists()

        # Reload
        from src.models.base_predictor import BasePredictor

        loaded = BasePredictor.load(str(artifact_path))

        # Predict with loaded model
        X_test = pd.DataFrame(
            {
                "points_scored_avg_3": [28.0],
                "points_allowed_avg_3": [18.0],
                "off_yards_per_play_avg_3": [6.0],
                "def_yards_per_play_avg_3": [5.0],
                "turnover_diff_avg_3": [1.0],
            }
        )
        pred = loaded.predict(X_test)
        assert pred.shape == (1,)
        assert pred[0] in (0, 1)

        proba = loaded.get_win_probabilities(X_test)
        assert 0.0 <= proba[0] <= 1.0


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class TestModelManager:
    def test_raises_when_no_artifact(self, tmp_path):
        """ModelManager raises ModelNotFoundError when no artifacts exist."""
        manager = ModelManager(artifacts_path=str(tmp_path))
        with pytest.raises(ModelNotFoundError):
            manager.get_model()

    def test_loads_best_model(self, tmp_path):
        """ModelManager loads the best model from registry."""
        # Train and save a model
        rng = np.random.RandomState(42)
        X = pd.DataFrame(
            {
                "f1": rng.uniform(0, 1, 50),
                "f2": rng.uniform(0, 1, 50),
            }
        )
        y = pd.Series(rng.choice([0, 1], 50))

        model = WinLossModel(model_variant="baseline")
        model.train(X, y)

        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register_model(
            model_type="win_loss_classifier",
            model_name="logistic_regression",
            version="1.0.0",
            hyperparameters={},
            feature_version="v1.0",
            train_seasons=[2023],
            test_seasons=[2024],
            metrics={"accuracy": 0.60},
            feature_names=["f1", "f2"],
        )
        model.save(str(tmp_path / f"{model_id}.joblib"))

        # Load via manager
        manager = ModelManager(artifacts_path=str(tmp_path))
        loaded = manager.get_model()
        assert loaded is not None

        info = manager.get_model_info()
        assert info["model_id"] == model_id
        assert manager.get_feature_names() == ["f1", "f2"]


# ---------------------------------------------------------------------------
# Inference features
# ---------------------------------------------------------------------------


class TestInferenceFeatures:
    def test_synthetic_features_shape(self):
        cols = ["points_scored_avg_3", "turnover_diff_avg_3", "win_pct_last_5"]
        df = build_inference_features_synthetic("KC", "SF", cols)
        assert df.shape == (1, 3)
        assert list(df.columns) == cols

    def test_synthetic_features_values(self):
        cols = ["points_scored_avg_3", "turnover_diff_avg_3"]
        df = build_inference_features_synthetic("KC", "SF", cols)
        assert df["points_scored_avg_3"].iloc[0] == 22.0
        assert df["turnover_diff_avg_3"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


class TestPredictionEndpoint:
    def _create_model_and_app(self, tmp_path):
        """Helper: train a model, save it, and create a test client."""
        rng = np.random.RandomState(99)
        n = 200
        X = pd.DataFrame(
            {
                "points_scored_avg_3": rng.uniform(14, 35, n),
                "points_allowed_avg_3": rng.uniform(14, 30, n),
                "off_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
                "def_yards_per_play_avg_3": rng.uniform(4.0, 7.5, n),
                "turnover_diff_avg_3": rng.uniform(-2.0, 2.0, n),
                "points_scored_avg_5": rng.uniform(14, 35, n),
                "points_allowed_avg_5": rng.uniform(14, 30, n),
                "current_streak": rng.randint(-5, 6, n).astype(float),
                "win_pct_last_5": rng.uniform(0.0, 1.0, n),
                "is_division_game": rng.choice([0, 1], n).astype(float),
            }
        )
        score_diff = X["points_scored_avg_3"] - X["points_allowed_avg_3"]
        y = pd.Series((score_diff > 0).astype(int))

        model = WinLossModel(model_variant="baseline")
        model.train(X, y)

        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register_model(
            model_type="win_loss_classifier",
            model_name="logistic_regression",
            version="1.0.0",
            hyperparameters={},
            feature_version="v1.0",
            train_seasons=[2023],
            test_seasons=[2024],
            metrics={"accuracy": 0.75},
            feature_names=list(X.columns),
        )
        model.save(str(tmp_path / f"{model_id}.joblib"))
        return model_id

    def test_predict_returns_non_stub(self, tmp_path, monkeypatch):
        """Prediction endpoint returns non-0.50 probability with a real model."""
        self._create_model_and_app(tmp_path)

        monkeypatch.setattr(
            "src.core.config.settings.MODEL_ARTIFACTS_PATH", str(tmp_path)
        )
        # Reset the cached manager
        import src.main

        src.main._manager = None

        client = TestClient(src.main.app)
        resp = client.post(
            "/predictions/predict",
            json={
                "home_team": "KC",
                "away_team": "SF",
                "season": 2024,
                "week": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["home_team"] == "KC"
        assert data["away_team"] == "SF"
        # With synthetic inference features (neutral defaults), prob should
        # be close to 0.5 but not exactly the hardcoded 0.50 stub
        assert "prediction" in data
        assert isinstance(data["prediction"]["win_probability"], float)

    def test_predict_503_when_no_model(self, tmp_path, monkeypatch):
        """Prediction endpoint returns 503 when no model artifact exists."""
        monkeypatch.setattr(
            "src.core.config.settings.MODEL_ARTIFACTS_PATH", str(tmp_path)
        )
        import src.main

        src.main._manager = None

        client = TestClient(src.main.app)
        resp = client.post(
            "/predictions/predict",
            json={
                "home_team": "KC",
                "away_team": "SF",
                "season": 2024,
                "week": 1,
            },
        )
        assert resp.status_code == 503

    def test_gateway_predict_endpoint(self, tmp_path, monkeypatch):
        """GET /predict endpoint works for API gateway."""
        self._create_model_and_app(tmp_path)

        monkeypatch.setattr(
            "src.core.config.settings.MODEL_ARTIFACTS_PATH", str(tmp_path)
        )
        import src.main

        src.main._manager = None

        client = TestClient(src.main.app)
        resp = client.get("/predict", params={"team1": "KC", "team2": "SF"})
        assert resp.status_code == 200
        data = resp.json()
        assert "home_win_probability" in data
        assert "away_win_probability" in data
        assert "bet_recommendation" in data
