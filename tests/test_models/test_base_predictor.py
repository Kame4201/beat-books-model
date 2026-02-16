"""Tests for BasePredictor abstract base class."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.base_predictor import BasePredictor


class ConcretePredictor(BasePredictor):
    """Concrete implementation for testing abstract base class."""

    def __init__(self):
        super().__init__(model_type="test_model", version="1.0.0")
        self.mock_model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._store_training_metadata(X_train, y_train, {"param": "value"})
        self.mock_model = "trained"
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_features(X)
        return np.ones(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_features(X)
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        self._validate_features(X_test)
        return {"accuracy": 1.0}


def test_base_predictor_initialization():
    """Test BasePredictor initialization."""
    predictor = ConcretePredictor()
    assert predictor.model_type == "test_model"
    assert predictor.version == "1.0.0"
    assert predictor.is_trained is False
    assert predictor.feature_names is None


def test_base_predictor_train():
    """Test training stores metadata correctly."""
    predictor = ConcretePredictor()
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    })
    y_train = pd.Series([0, 1, 0])

    predictor.train(X_train, y_train)

    assert predictor.is_trained is True
    assert predictor.feature_names == ["feature1", "feature2"]
    assert predictor.training_metadata["n_samples"] == 3
    assert predictor.training_metadata["n_features"] == 2


def test_base_predictor_validate_features():
    """Test feature validation."""
    predictor = ConcretePredictor()
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    })
    y_train = pd.Series([0, 1, 0])
    predictor.train(X_train, y_train)

    # Valid features
    X_test = pd.DataFrame({
        "feature1": [7, 8],
        "feature2": [9, 10],
    })
    predictor.predict(X_test)  # Should not raise

    # Missing feature
    X_missing = pd.DataFrame({"feature1": [7, 8]})
    with pytest.raises(ValueError, match="Missing features"):
        predictor.predict(X_missing)


def test_base_predictor_save_load():
    """Test model serialization."""
    predictor = ConcretePredictor()
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    })
    y_train = pd.Series([0, 1, 0])
    predictor.train(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.joblib"
        predictor.save(str(save_path))

        # Load and verify
        loaded = ConcretePredictor.load(str(save_path))
        assert loaded.is_trained is True
        assert loaded.feature_names == ["feature1", "feature2"]
        assert loaded.model_type == "test_model"


def test_base_predictor_save_untrained():
    """Test that saving untrained model raises error."""
    predictor = ConcretePredictor()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.joblib"
        with pytest.raises(RuntimeError, match="Cannot save untrained"):
            predictor.save(str(save_path))


def test_base_predictor_predict_untrained():
    """Test that predicting with untrained model raises error."""
    predictor = ConcretePredictor()
    X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

    with pytest.raises(RuntimeError, match="has not been trained yet"):
        predictor.predict(X)
