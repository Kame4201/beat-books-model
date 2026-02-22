"""
Point Spread Regression Model.

Predicts point spread (home_score - away_score) using:
- Baseline: Linear Regression
- Baseline (regularized): Ridge Regression
- Advanced: Gradient Boosted Regressors (XGBoost or LightGBM)

Outputs:
- Predicted spread: negative = home favored, positive = away favored
- Confidence intervals (for applicable models)
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.models.base_predictor import BasePredictor
from src.models.model_config import get_config


class SpreadModel(BasePredictor):
    """
    Regression model for predicting point spread (home_score - away_score).

    Negative spread = home team favored
    Positive spread = away team favored

    Supports multiple model types:
    - "baseline": Linear Regression
    - "ridge": Ridge Regression (L2 regularization)
    - "xgboost": XGBoost Regressor
    - "lightgbm": LightGBM Regressor
    """

    def __init__(
        self,
        model_variant: str = "baseline",
        version: str = "1.0.0",
        custom_hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Spread regressor.

        Args:
            model_variant: Model variant ("baseline", "ridge", "xgboost", or "lightgbm")
            version: Model version string
            custom_hyperparameters: Optional dict to override default hyperparameters

        Raises:
            ValueError: If model_variant is invalid
        """
        super().__init__(model_type="spread_regressor", version=version)

        if model_variant not in ["baseline", "ridge", "xgboost", "lightgbm"]:
            raise ValueError(
                f"Invalid model_variant '{model_variant}'. "
                f"Valid options: 'baseline', 'ridge', 'xgboost', 'lightgbm'"
            )

        self.model_variant = model_variant
        self.config = get_config("spread", model_variant)

        # Override hyperparameters if provided
        if custom_hyperparameters:
            self.config["hyperparameters"].update(custom_hyperparameters)

        self.hyperparameters = self.config["hyperparameters"]
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying sklearn/xgboost/lightgbm model."""
        if self.model_variant == "baseline":
            self.model = LinearRegression(**self.hyperparameters)
        elif self.model_variant == "ridge":
            self.model = Ridge(**self.hyperparameters)
        elif self.model_variant == "xgboost":
            self.model = xgb.XGBRegressor(**self.hyperparameters)
        elif self.model_variant == "lightgbm":
            self.model = lgb.LGBMRegressor(**self.hyperparameters)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Spread regressor.

        Args:
            X_train: Feature matrix (DataFrame with feature columns)
            y_train: Target spread values (home_score - away_score)

        Raises:
            ValueError: If input data is invalid
        """
        if X_train.empty:
            raise ValueError("Training data is empty")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
            )

        # Store feature names and metadata
        self._store_training_metadata(X_train, y_train, self.hyperparameters)

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict point spread.

        Args:
            X: Feature matrix (DataFrame with same columns as training data)

        Returns:
            Array of predicted spreads (negative = home favored)

        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input features don't match training features

        Examples:
            >>> model = SpreadModel()
            >>> model.train(X_train, y_train)
            >>> spreads = model.predict(X_test)
            >>> spreads[0]  # e.g., -7.5 means home favored by 7.5 points
            -7.5
        """
        self._validate_features(X)

        # Ensure column order matches training
        X_ordered = X[self.feature_names]

        return self.model.predict(X_ordered)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predictions with confidence (for compatibility with base class).

        For regression models, this returns the predictions themselves.
        In future versions, could return prediction intervals.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (same as predict())
        """
        return self.predict(X)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict spread with confidence scores.

        For linear models, confidence is based on distance from decision boundary.
        For tree models, could use prediction variance across trees.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with columns ["predicted_spread", "confidence"]
            where confidence is in [0, 1] (higher = more confident)

        Examples:
            >>> model = SpreadModel()
            >>> model.train(X_train, y_train)
            >>> predictions = model.predict_with_confidence(X_test)
            >>> predictions.head()
               predicted_spread  confidence
            0             -7.5        0.75
            1              3.2        0.60
        """
        predictions = self.predict(X)

        # Simple confidence heuristic: higher absolute spread = higher confidence
        # This is a placeholder - could be improved with proper calibration
        confidence = np.clip(np.abs(predictions) / 14.0, 0.0, 1.0)

        return pd.DataFrame(
            {
                "predicted_spread": predictions,
                "confidence": confidence,
            }
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate Spread model on test data.

        Computes:
        - mae: Mean Absolute Error (lower is better)
        - rmse: Root Mean Squared Error (lower is better)
        - r2: R-squared coefficient (higher is better)
        - median_ae: Median Absolute Error (robust to outliers)

        Args:
            X_test: Test feature matrix
            y_test: Test target (actual spread values)

        Returns:
            Dictionary of metric names and values

        Raises:
            RuntimeError: If model has not been trained
        """
        self._validate_features(X_test)

        # Predictions
        y_pred = self.predict(X_test)

        # Metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "median_ae": np.median(np.abs(y_test - y_pred)),
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
            >>> model = SpreadModel(model_variant="xgboost")
            >>> model.train(X_train, y_train)
            >>> importance = model.get_feature_importance(top_n=5)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")

        if self.model_variant in ["baseline", "ridge"]:
            # Linear models use coefficients
            coef = self.model.coef_
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

    def predict_home_favorite(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether home team is favored (convenience method).

        Args:
            X: Feature matrix

        Returns:
            Array of booleans (True = home favored, False = away favored)

        Examples:
            >>> model = SpreadModel()
            >>> model.train(X_train, y_train)
            >>> is_home_favored = model.predict_home_favorite(X_test)
            >>> is_home_favored[0]
            True  # Home team predicted to win
        """
        spreads = self.predict(X)
        return spreads < 0  # Negative spread = home favored
