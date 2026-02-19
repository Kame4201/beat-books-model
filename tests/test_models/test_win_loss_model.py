"""Tests for WinLossModel."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.win_loss_model import WinLossModel


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
    y_train = pd.Series(np.random.choice([0, 1], n_train))

    # Test data
    X_test = pd.DataFrame(
        {
            "points_scored_avg": np.random.uniform(20, 35, n_test),
            "points_allowed_avg": np.random.uniform(15, 25, n_test),
            "yards_per_play": np.random.uniform(5.0, 7.0, n_test),
            "turnover_diff": np.random.randint(-3, 4, n_test),
        }
    )
    y_test = pd.Series(np.random.choice([0, 1], n_test))

    return X_train, y_train, X_test, y_test


def test_win_loss_model_initialization():
    """Test WinLossModel initialization."""
    model = WinLossModel(model_variant="baseline")
    assert model.model_type == "win_loss_classifier"
    assert model.model_variant == "baseline"
    assert model.is_trained is False


def test_win_loss_model_invalid_variant():
    """Test that invalid model variant raises error."""
    with pytest.raises(ValueError, match="Invalid model_variant"):
        WinLossModel(model_variant="invalid")


def test_win_loss_model_train_baseline(sample_data):
    """Test training baseline logistic regression model."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_win_loss_model_train_xgboost(sample_data):
    """Test training XGBoost model."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="xgboost")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_win_loss_model_train_lightgbm(sample_data):
    """Test training LightGBM model."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="lightgbm")

    model.train(X_train, y_train)

    assert model.is_trained is True
    assert model.feature_names == list(X_train.columns)


def test_win_loss_model_predict(sample_data):
    """Test making predictions."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_win_loss_model_predict_proba(sample_data):
    """Test predicting probabilities."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    proba = model.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(proba >= 0) and np.all(proba <= 1)  # Between 0 and 1


def test_win_loss_model_get_win_probabilities(sample_data):
    """Test getting win probabilities."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    win_probs = model.get_win_probabilities(X_test)

    assert len(win_probs) == len(X_test)
    assert np.all(win_probs >= 0) and np.all(win_probs <= 1)


def test_win_loss_model_evaluate(sample_data):
    """Test model evaluation."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)

    assert "accuracy" in metrics
    assert "log_loss" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_win_loss_model_feature_importance_baseline(sample_data):
    """Test feature importance for baseline model."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    importance = model.get_feature_importance(top_n=3)

    assert len(importance) <= 3
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_win_loss_model_feature_importance_xgboost(sample_data):
    """Test feature importance for XGBoost model."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="xgboost")
    model.train(X_train, y_train)

    importance = model.get_feature_importance(top_n=3)

    assert len(importance) <= 3
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_win_loss_model_save_load(sample_data):
    """Test model serialization."""
    X_train, y_train, X_test, y_test = sample_data
    model = WinLossModel(model_variant="baseline")
    model.train(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.joblib"
        model.save(str(save_path))

        # Load and verify
        loaded_model = WinLossModel.load(str(save_path))
        assert loaded_model.is_trained is True
        assert loaded_model.model_variant == "baseline"

        # Verify predictions match
        orig_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        assert np.array_equal(orig_pred, loaded_pred)


def test_win_loss_model_train_empty_data():
    """Test training with empty data raises error."""
    model = WinLossModel(model_variant="baseline")
    X_empty = pd.DataFrame()
    y_empty = pd.Series()

    with pytest.raises(ValueError, match="Training data is empty"):
        model.train(X_empty, y_empty)


def test_win_loss_model_train_invalid_target():
    """Test training with non-binary target raises error."""
    model = WinLossModel(model_variant="baseline")
    X = pd.DataFrame({"feat": [1, 2, 3]})
    y = pd.Series([0, 1, 2])  # Should be binary

    with pytest.raises(ValueError, match="Target must be binary"):
        model.train(X, y)


def test_win_loss_model_custom_hyperparameters(sample_data):
    """Test using custom hyperparameters."""
    X_train, y_train, X_test, y_test = sample_data
    custom_params = {"max_iter": 500, "C": 0.5}
    model = WinLossModel(model_variant="baseline", custom_hyperparameters=custom_params)

    model.train(X_train, y_train)

    assert model.hyperparameters["max_iter"] == 500
    assert model.hyperparameters["C"] == 0.5
