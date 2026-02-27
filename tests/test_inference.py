"""
Tests for model inference: ModelManager, feature builder, and /predictions/predict endpoint.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.base_predictor import BasePredictor
from src.models.model_registry import ModelRegistry
from src.models.win_loss_model import WinLossModel
from src.service.model_manager import ModelManager, ModelNotFoundError


@pytest.fixture
def trained_model_dir(tmp_path):
    """Create a temp dir with a trained model and registry."""
    # Train a tiny model
    np.random.seed(42)
    X = pd.DataFrame({
        "diff_pf": np.random.randn(100),
        "diff_pa": np.random.randn(100),
        "diff_mov": np.random.randn(100),
        "diff_srs": np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], 100))

    model = WinLossModel(model_variant="baseline", version="1.0.0")
    model.train(X, y)

    # Register and save
    registry = ModelRegistry(str(tmp_path))
    model_id = registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={"C": 1.0},
        feature_version="1.0.0",
        train_seasons=[2020, 2021, 2022],
        test_seasons=[2023],
        metrics={"accuracy": 0.60, "log_loss": 0.68},
        feature_names=list(X.columns),
    )
    model.save(str(tmp_path / f"{model_id}.joblib"))

    return tmp_path, model_id


class TestModelManager:
    """Tests for ModelManager."""

    def test_loads_best_model(self, trained_model_dir):
        tmp_path, model_id = trained_model_dir
        manager = ModelManager(str(tmp_path))
        m = manager.model
        assert m.is_trained
        assert manager.model_meta["model_id"] == model_id

    def test_raises_when_no_registry(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        with pytest.raises(ModelNotFoundError, match="No registry.json"):
            _ = manager.model

    def test_raises_when_no_models(self, tmp_path):
        # Empty registry
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps({"models": []}))
        manager = ModelManager(str(tmp_path))
        with pytest.raises(ModelNotFoundError, match="No win_loss_classifier"):
            _ = manager.model

    def test_raises_when_artifact_missing(self, tmp_path):
        # Registry with entry but no .joblib file
        registry = ModelRegistry(str(tmp_path))
        registry.register_model(
            model_type="win_loss_classifier",
            model_name="lr",
            version="1.0.0",
            hyperparameters={},
            feature_version="1.0.0",
            train_seasons=[2020],
            test_seasons=[2023],
            metrics={"accuracy": 0.55},
            feature_names=["f1"],
        )
        manager = ModelManager(str(tmp_path))
        with pytest.raises(ModelNotFoundError, match="artifact not found"):
            _ = manager.model


class TestModelSaveLoad:
    """Tests for model save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y = pd.Series(np.random.choice([0, 1], 50))

        model = WinLossModel(model_variant="baseline")
        model.train(X, y)

        path = str(tmp_path / "test_model.joblib")
        model.save(path)

        loaded = BasePredictor.load(path)
        assert loaded.is_trained
        assert loaded.feature_names == ["a", "b"]

        # Predictions should be identical
        orig_pred = model.predict_proba(X)
        loaded_pred = loaded.predict_proba(X)
        np.testing.assert_array_equal(orig_pred, loaded_pred)


class TestPredictEndpoint:
    """Tests for /predictions/predict endpoint."""

    def test_returns_503_when_no_model(self):
        with patch("src.main._manager") as mock_mgr:
            mock_mgr.model = property(
                lambda self: (_ for _ in ()).throw(ModelNotFoundError("no model"))
            )
            type(mock_mgr).model = property(
                lambda self: (_ for _ in ()).throw(ModelNotFoundError("no model"))
            )
            from src.main import app

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/predictions/predict",
                json={"home_team": "KC", "away_team": "BUF", "season": 2023},
            )
            assert resp.status_code == 503

    def test_returns_prediction_with_model(self, trained_model_dir):
        tmp_path, _ = trained_model_dir
        manager = ModelManager(str(tmp_path))

        with patch("src.main._manager", manager):
            # Also mock the DB feature builder to return zeros
            with patch("src.main.build_inference_features") as mock_feat:
                features = model_features = manager.model.feature_names
                mock_feat.return_value = pd.DataFrame(
                    [{f: 0.0 for f in features}]
                )
                from src.main import app

                client = TestClient(app)
                resp = client.post(
                    "/predictions/predict",
                    json={"home_team": "KC", "away_team": "BUF", "season": 2023},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["home_team"] == "KC"
                assert data["away_team"] == "BUF"
                assert "prediction" in data
                assert "winner" in data["prediction"]
                assert 0.0 <= data["prediction"]["win_probability"] <= 1.0
                assert data["prediction"]["confidence"] in ("low", "medium", "high")

    def test_health_endpoint(self):
        from src.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "beat-books-model"
