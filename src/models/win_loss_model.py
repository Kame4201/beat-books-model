"""
Win/Loss Classification Model.

Predicts whether home team wins or loses using:
- Baseline: Logistic Regression (scikit-learn)
- Advanced: Gradient Boosted Classifiers (XGBoost or LightGBM)

Outputs:
- Binary prediction: 1 (home win), 0 (home loss)
- Probabilities: P(home_win), P(away_win) summing to 1.0
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb
import lightgbm as lgb

from src.models.base_predictor import BasePredictor
from src.models.model_config import get_config


class WinLossModel(BasePredictor):
    """
    Binary classifier for predicting home team win/loss.

    Supports multiple model types:
    - "baseline": Logistic Regression
    - "xgboost": XGBoost Classifier
    - "lightgbm": LightGBM Classifier
    """

    def __init__(
        self,
        model_variant: str = "baseline",
        version: str = "1.0.0",
        custom_hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Win/Loss classifier.

        Args:
            model_variant: Model variant ("baseline", "xgboost", or "lightgbm")
            version: Model version string
            custom_hyperparameters: Optional dict to override default hyperparameters

        Raises:
            ValueError: If model_variant is invalid
        """
        super().__init__(model_type="win_loss_classifier", version=version)

        if model_variant not in ["baseline", "xgboost", "lightgbm"]:
            raise ValueError(
                f"Invalid model_variant '{model_variant}'. "
                f"Valid options: 'baseline', 'xgboost', 'lightgbm'"
            )

        self.model_variant = model_variant
        self.config = get_config("win_loss", model_variant)

        # Override hyperparameters if provided
        if custom_hyperparameters:
            self.config["hyperparameters"].update(custom_hyperparameters)

        self.hyperparameters = self.config["hyperparameters"]
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying sklearn/xgboost/lightgbm model."""
        if self.model_variant == "baseline":
            self.model = LogisticRegression(**self.hyperparameters)
        elif self.model_variant == "xgboost":
            self.model = xgb.XGBClassifier(**self.hyperparameters)
        elif self.model_variant == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Win/Loss classifier.

        Args:
            X_train: Feature matrix (DataFrame with feature columns)
            y_train: Binary target (1 = home win, 0 = home loss)

        Raises:
            ValueError: If input data is invalid
        """
        if X_train.empty:
            raise ValueError("Training data is empty")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
            )

        # Validate target is binary
        unique_values = y_train.unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(
                f"Target must be binary (0 or 1). Found values: {unique_values}"
            )

        # Store feature names and metadata
        self._store_training_metadata(X_train, y_train, self.hyperparameters)

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict home team win/loss.

        Args:
            X: Feature matrix (DataFrame with same columns as training data)

        Returns:
            Array of binary predictions (1 = home win, 0 = home loss)

        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input features don't match training features
        """
        self._validate_features(X)

        # Ensure column order matches training
        X_ordered = X[self.feature_names]

        return self.model.predict(X_ordered)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for home win/loss.

        Args:
            X: Feature matrix (DataFrame)

        Returns:
            Array of shape (n_samples, 2) with [P(loss), P(win)] for each sample

        Raises:
            RuntimeError: If model has not been trained
        """
        self._validate_features(X)

        # Ensure column order matches training
        X_ordered = X[self.feature_names]

        return self.model.predict_proba(X_ordered)

    def get_win_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get home team win probabilities (convenience method).

        Args:
            X: Feature matrix

        Returns:
            Array of P(home_win) values

        Examples:
            >>> model = WinLossModel()
            >>> model.train(X_train, y_train)
            >>> win_probs = model.get_win_probabilities(X_test)
            >>> win_probs[0]  # Probability home team wins game 0
            0.62
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # Column 1 is P(win)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate Win/Loss model on test data.

        Computes:
        - accuracy: Overall classification accuracy
        - log_loss: Logarithmic loss (lower is better)
        - precision: Precision for home wins
        - recall: Recall for home wins
        - f1_score: F1 score for home wins

        Args:
            X_test: Test feature matrix
            y_test: Test target (binary: 0 or 1)

        Returns:
            Dictionary of metric names and values

        Raises:
            RuntimeError: If model has not been trained
        """
        self._validate_features(X_test)

        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        return metrics

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance scores (for tree-based models).

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with columns ["feature", "importance"] sorted by importance

        Raises:
            RuntimeError: If model has not been trained
            AttributeError: If model doesn't support feature importance

        Examples:
            >>> model = WinLossModel(model_variant="xgboost")
            >>> model.train(X_train, y_train)
            >>> importance = model.get_feature_importance(top_n=5)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")

        if self.model_variant == "baseline":
            # Logistic regression uses coefficients instead
            coef = self.model.coef_[0]
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": np.abs(coef),  # Use absolute value
                }
            )
        elif self.model_variant in ["xgboost", "lightgbm"]:
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
        else:
            raise AttributeError(
                f"Feature importance not available for {self.model_variant}"
            )

        # Sort and return top N
        importance_df = importance_df.sort_values("importance", ascending=False)
        return importance_df.head(top_n).reset_index(drop=True)
