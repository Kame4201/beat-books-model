"""Tests for SpreadModel."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.spread_model import SpreadModel


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    n_train = 100
    n_test = 20

    # Training data
    X_train = pd.DataFrame(
        {
            "points_scored_avg": np.random.uniform(20, 35, n_train),
            "points_allowed_avg": np.random.uniform(15, 25, n_train),
            "yards_per_play": np.random.uniform(5.0, 7.0, n_train),
            "turnover_diff": np.random.randint(-3, 4, n_train),
        }
    )
    y_train = pd.Series(np.random.uniform(-14, 14, n_train))  # Spread values

    # Test data
    X_test = pd.DataFrame(
        {
            "points_scored_avg": np.random.uniform(20, 35, n_test),
            "points_allowed_avg": np.random.uniform(15, 25, n_test),
            "yards_per_play": np.random.uniform(5.0, 7.0, n_test),
            "turnover_diff": np.random.randint(-3, 4, n_test),
        }
    )
    y_test = pd.Series(np.random.uniform(-14, 14, n_test))

    return X_train, y_train, X_test, y_test


def test_spread_model_initialization():
    """Test SpreadModel initialization."""
    model = SpreadModel(model_variant="baseline")
    assert model.model_type == "spread_regressor"
    assert model.model_variant == "baseline"
    assert model.is_trained is False


def test_spread_model_invalid_variant():
    """Test that invalid model variant raises error."""
    with pytest.raises(ValueError, match="Invalid model_variant"):
        SpreadModel(model_variant="invalid")


def test_spread_model_train_baseline(sample_data):
    """Test training baseline linear regression model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_spread_model_train_ridge(sample_data):
    """Test training ridge regression model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="ridge")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_spread_model_train_xgboost(sample_data):
    """Test training XGBoost model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="xgboost")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_spread_model_train_lightgbm(sample_data):
    """Test training LightGBM model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="lightgbm")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_spread_model_predict(sample_data):
    """Test making predictions."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)


def test_spread_model_predict_with_confidence(sample_data):
    """Test predicting with confidence scores."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    result = model.predict_with_confidence(X_test)

    assert len(result) == len(X_test)
    assert "predicted_spread" in result.columns
    assert "confidence" in result.columns
    assert np.all(result["confidence"] >= 0)
    assert np.all(result["confidence"] <= 1)


def test_spread_model_predict_home_favorite(sample_data):
    """Test predicting home favorite."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    is_home_favored = model.predict_home_favorite(X_test)

    assert len(is_home_favored) == len(X_test)
    assert is_home_favored.dtype == bool


def test_spread_model_evaluate(sample_data):
    """Test model evaluation."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "median_ae" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_spread_model_feature_importance_baseline(sample_data):
    """Test feature importance for baseline model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    importance = model.get_feature_importance(top_n=3)

    assert len(importance) <= 3
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_spread_model_feature_importance_xgboost(sample_data):
    """Test feature importance for XGBoost model."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="xgboost")
    model.train(X_train, y_train)

    importance = model.get_feature_importance(top_n=3)

    assert len(importance) <= 3
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_spread_model_save_load(sample_data):
    """Test model serialization."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.joblib"
        model.save(str(save_path))

        # Load and verify
        loaded_model = SpreadModel.load(str(save_path))
        assert loaded_model.is_trained is True
        assert loaded_model.model_variant == "baseline"

        # Verify predictions match
        orig_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        assert np.allclose(orig_pred, loaded_pred)


def test_spread_model_train_empty_data():
    """Test training with empty data raises error."""
    model = SpreadModel(model_variant="baseline")
    X_empty = pd.DataFrame()
    y_empty = pd.Series()

    with pytest.raises(ValueError, match="Training data is empty"):
        model.train(X_empty, y_empty)


def test_spread_model_custom_hyperparameters(sample_data):
    """Test using custom hyperparameters."""
    X_train, y_train, X_test, y_test = sample_data
    custom_params = {"alpha": 2.0}
    model = SpreadModel(model_variant="ridge", custom_hyperparameters=custom_params)

    model.train(X_train, y_train)

    assert model.hyperparameters["alpha"] == 2.0


def test_spread_model_negative_spread_means_home_favored(sample_data):
    """Test that negative spread correctly indicates home team is favored."""
    X_train, y_train, X_test, y_test = sample_data
    model = SpreadModel(model_variant="baseline")
    model.train(X_train, y_train)

    # Create a test case where we expect home to be favored
    X_single = X_test.iloc[[0]]
    spread = model.predict(X_single)[0]
    is_home_favored = model.predict_home_favorite(X_single)[0]

    # If spread is negative, home should be favored
    if spread < 0:
        assert is_home_favored == True  # noqa: E712 (numpy bool)
    else:
        assert is_home_favored == False  # noqa: E712 (numpy bool)
