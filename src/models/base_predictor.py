"""
Abstract base class for all prediction models in beat-books-model.

All models (Win/Loss classification, Spread regression) inherit from this base class
to ensure consistent interface for training, prediction, evaluation, and serialization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    Abstract base class for prediction models.

    Defines the interface that all models must implement:
    - train: Fit model on training data
    - predict: Generate predictions on new data
    - predict_proba: Generate probability estimates (for classifiers)
    - evaluate: Compute performance metrics on test data
    - save: Serialize model to disk
    - load: Deserialize model from disk
    """

    def __init__(self, model_type: str, version: str = "1.0.0"):
        """
        Initialize base predictor.

        Args:
            model_type: Type of model (e.g., "win_loss_classifier", "spread_regressor")
            version: Model version string (semantic versioning)
        """
        self.model_type = model_type
        self.version = version
        self.model: Any = None
        self.is_trained = False
        self.feature_names: Optional[list] = None
        self.training_metadata: Dict[str, Any] = {}

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on provided data.

        Args:
            X_train: Feature matrix (DataFrame with feature columns)
            y_train: Target variable (Series)

        Raises:
            ValueError: If input data is invalid
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on new data.

        Args:
            X: Feature matrix (DataFrame with same columns as training data)

        Returns:
            Array of predictions

        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input features don't match training features
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability estimates for predictions.

        For classifiers: returns class probabilities.
        For regressors: may return prediction intervals or confidence scores.

        Args:
            X: Feature matrix (DataFrame)

        Returns:
            Array of probability estimates

        Raises:
            RuntimeError: If model has not been trained
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test target variable

        Returns:
            Dictionary of metric names and values (e.g., {"accuracy": 0.65, "log_loss": 0.45})

        Raises:
            RuntimeError: If model has not been trained
        """
        pass

    def save(self, path: str) -> None:
        """
        Save model to disk using joblib.

        Args:
            path: File path to save model (should end in .joblib)

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(f"Cannot save untrained {self.model_type} model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save entire predictor object (includes model, metadata, feature_names)
        joblib.dump(self, save_path)

    @classmethod
    def load(cls, path: str) -> "BasePredictor":
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Loaded predictor instance

        Raises:
            FileNotFoundError: If model file does not exist
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        return joblib.load(load_path)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that input features match training features.

        Args:
            X: Feature DataFrame to validate

        Raises:
            RuntimeError: If model has not been trained
            ValueError: If features don't match training features
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_type} model has not been trained yet")

        if self.feature_names is None:
            raise RuntimeError(f"{self.model_type} model feature names not set")

        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Missing features in input data: {missing_features}. "
                f"Expected: {self.feature_names}"
            )

        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            # Warning only - we'll select the correct columns
            pass

    def _store_training_metadata(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
    ) -> None:
        """
        Store metadata about the training process.

        Args:
            X_train: Training features
            y_train: Training target
            hyperparameters: Model hyperparameters
        """
        self.training_metadata = {
            "n_samples": len(X_train),
            "n_features": len(X_train.columns),
            "feature_names": list(X_train.columns),
            "hyperparameters": hyperparameters,
            "model_type": self.model_type,
            "version": self.version,
        }
        self.feature_names = list(X_train.columns)
