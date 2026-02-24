"""Tests for ModelRegistry."""

import pytest
import tempfile
from pathlib import Path

from src.models.model_registry import ModelRegistry


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_dir=tmpdir)
        yield registry


def test_model_registry_initialization(temp_registry):
    """Test ModelRegistry initialization."""
    assert temp_registry.registry_file.exists()
    assert "models" in temp_registry.registry


def test_model_registry_register_model(temp_registry):
    """Test registering a new model."""
    model_id = temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={"C": 1.0, "max_iter": 1000},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58, "log_loss": 0.65},
        feature_names=["feature1", "feature2"],
        notes="Baseline model",
    )

    assert model_id is not None
    assert len(temp_registry.registry["models"]) == 1

    # Verify model was saved
    model = temp_registry.get_model(model_id)
    assert model["model_type"] == "win_loss_classifier"
    assert model["model_name"] == "logistic_regression"
    assert model["metrics"]["accuracy"] == 0.58


def test_model_registry_get_model(temp_registry):
    """Test retrieving a model by ID."""
    model_id = temp_registry.register_model(
        model_type="spread_regressor",
        model_name="linear_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"mae": 12.5},
        feature_names=["feature1"],
    )

    model = temp_registry.get_model(model_id)
    assert model is not None
    assert model["model_id"] == model_id


def test_model_registry_get_nonexistent_model(temp_registry):
    """Test retrieving a model that doesn't exist."""
    model = temp_registry.get_model("nonexistent-id")
    assert model is None


def test_model_registry_list_models(temp_registry):
    """Test listing all models."""
    # Register multiple models
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )
    temp_registry.register_model(
        model_type="spread_regressor",
        model_name="linear_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"mae": 12.5},
        feature_names=["feat1"],
    )

    all_models = temp_registry.list_models()
    assert len(all_models) == 2


def test_model_registry_list_models_filtered(temp_registry):
    """Test listing models with filters."""
    # Register multiple models
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="xgboost_classifier",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.62},
        feature_names=["feat1"],
    )
    temp_registry.register_model(
        model_type="spread_regressor",
        model_name="linear_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"mae": 12.5},
        feature_names=["feat1"],
    )

    # Filter by model_type
    win_loss_models = temp_registry.list_models(model_type="win_loss_classifier")
    assert len(win_loss_models) == 2

    # Filter by model_name
    xgboost_models = temp_registry.list_models(model_name="xgboost_classifier")
    assert len(xgboost_models) == 1


def test_model_registry_get_best_model(temp_registry):
    """Test getting best model by metric."""
    # Register models with different accuracy
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="xgboost_classifier",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.62},
        feature_names=["feat1"],
    )

    best = temp_registry.get_best_model("win_loss_classifier", "accuracy")
    assert best is not None
    assert best["metrics"]["accuracy"] == 0.62


def test_model_registry_get_best_model_minimize(temp_registry):
    """Test getting best model by minimizing metric."""
    # Register models with different MAE
    temp_registry.register_model(
        model_type="spread_regressor",
        model_name="linear_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"mae": 12.5},
        feature_names=["feat1"],
    )
    temp_registry.register_model(
        model_type="spread_regressor",
        model_name="xgboost_regressor",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"mae": 10.2},
        feature_names=["feat1"],
    )

    best = temp_registry.get_best_model("spread_regressor", "mae", minimize=True)
    assert best is not None
    assert best["metrics"]["mae"] == 10.2


def test_model_registry_get_artifact_path(temp_registry):
    """Test getting artifact path for a model."""
    model_id = temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )

    artifact_path = temp_registry.get_artifact_path(model_id)
    assert artifact_path is not None
    assert artifact_path.name == f"{model_id}.joblib"


def test_model_registry_delete_model(temp_registry):
    """Test deleting a model."""
    model_id = temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )

    # Delete model
    success = temp_registry.delete_model(model_id, delete_artifact=False)
    assert success is True

    # Verify model is gone
    model = temp_registry.get_model(model_id)
    assert model is None


def test_model_registry_export_summary(temp_registry):
    """Test exporting registry summary."""
    # Register a model
    temp_registry.register_model(
        model_type="win_loss_classifier",
        model_name="logistic_regression",
        version="1.0.0",
        hyperparameters={"C": 1.0},
        feature_version="1.0.0",
        train_seasons=list(range(2000, 2023)),
        test_seasons=[2023, 2024],
        metrics={"accuracy": 0.58},
        feature_names=["feat1"],
    )

    summary = temp_registry.export_summary()
    assert "MODEL REGISTRY SUMMARY" in summary
    assert "logistic_regression" in summary
    assert "accuracy" in summary


def test_model_registry_persistence():
    """Test that registry persists across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create registry and add a model
        registry1 = ModelRegistry(registry_dir=tmpdir)
        model_id = registry1.register_model(
            model_type="win_loss_classifier",
            model_name="logistic_regression",
            version="1.0.0",
            hyperparameters={},
            feature_version="1.0.0",
            train_seasons=list(range(2000, 2023)),
            test_seasons=[2023, 2024],
            metrics={"accuracy": 0.58},
            feature_names=["feat1"],
        )

        # Create new registry instance in same directory
        registry2 = ModelRegistry(registry_dir=tmpdir)
        model = registry2.get_model(model_id)
        assert model is not None
        assert model["model_name"] == "logistic_regression"
