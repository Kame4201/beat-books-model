"""
Hyperparameter configurations for all prediction models.

Defines baseline and advanced configurations for:
- Win/Loss classification (Logistic Regression, XGBoost, LightGBM)
- Spread regression (Linear Regression, XGBoost, LightGBM)
"""

import copy
from typing import Any, Dict

# ============================================================================
# Win/Loss Classification Configs
# ============================================================================

LOGISTIC_REGRESSION_CONFIG: Dict[str, Any] = {
    "model_name": "logistic_regression",
    "model_class": "sklearn.linear_model.LogisticRegression",
    "hyperparameters": {
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs",
        "C": 1.0,  # Inverse regularization strength
    },
    "description": "Baseline logistic regression classifier for Win/Loss prediction",
}

XGBOOST_CLASSIFIER_CONFIG: Dict[str, Any] = {
    "model_name": "xgboost_classifier",
    "model_class": "xgboost.XGBClassifier",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "description": "Advanced gradient boosted classifier (XGBoost) for Win/Loss prediction",
}

LIGHTGBM_CLASSIFIER_CONFIG: Dict[str, Any] = {
    "model_name": "lightgbm_classifier",
    "model_class": "lightgbm.LGBMClassifier",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    },
    "description": "Advanced gradient boosted classifier (LightGBM) for Win/Loss prediction",
}


# ============================================================================
# Spread Regression Configs
# ============================================================================

LINEAR_REGRESSION_CONFIG: Dict[str, Any] = {
    "model_name": "linear_regression",
    "model_class": "sklearn.linear_model.LinearRegression",
    "hyperparameters": {
        "fit_intercept": True,
        "copy_X": True,
    },
    "description": "Baseline linear regression for point spread prediction",
}

RIDGE_REGRESSION_CONFIG: Dict[str, Any] = {
    "model_name": "ridge_regression",
    "model_class": "sklearn.linear_model.Ridge",
    "hyperparameters": {
        "alpha": 1.0,  # Regularization strength
        "fit_intercept": True,
        "random_state": 42,
    },
    "description": "Baseline ridge regression (L2 regularized) for point spread prediction",
}

XGBOOST_REGRESSOR_CONFIG: Dict[str, Any] = {
    "model_name": "xgboost_regressor",
    "model_class": "xgboost.XGBRegressor",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    },
    "description": "Advanced gradient boosted regressor (XGBoost) for spread prediction",
}

LIGHTGBM_REGRESSOR_CONFIG: Dict[str, Any] = {
    "model_name": "lightgbm_regressor",
    "model_class": "lightgbm.LGBMRegressor",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
    },
    "description": "Advanced gradient boosted regressor (LightGBM) for spread prediction",
}


# ============================================================================
# Configuration Registry
# ============================================================================

WIN_LOSS_CONFIGS = {
    "baseline": LOGISTIC_REGRESSION_CONFIG,
    "xgboost": XGBOOST_CLASSIFIER_CONFIG,
    "lightgbm": LIGHTGBM_CLASSIFIER_CONFIG,
}

SPREAD_CONFIGS = {
    "baseline": LINEAR_REGRESSION_CONFIG,
    "ridge": RIDGE_REGRESSION_CONFIG,
    "xgboost": XGBOOST_REGRESSOR_CONFIG,
    "lightgbm": LIGHTGBM_REGRESSOR_CONFIG,
}


def get_config(model_type: str, config_name: str) -> Dict[str, Any]:
    """
    Retrieve configuration for a specific model type and config name.

    Args:
        model_type: Either "win_loss" or "spread"
        config_name: Configuration name (e.g., "baseline", "xgboost", "lightgbm")

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If model_type or config_name is invalid

    Examples:
        >>> config = get_config("win_loss", "baseline")
        >>> config["model_name"]
        'logistic_regression'
    """
    if model_type == "win_loss":
        if config_name not in WIN_LOSS_CONFIGS:
            raise ValueError(
                f"Invalid config_name '{config_name}' for win_loss. "
                f"Valid options: {list(WIN_LOSS_CONFIGS.keys())}"
            )
        return copy.deepcopy(WIN_LOSS_CONFIGS[config_name])
    elif model_type == "spread":
        if config_name not in SPREAD_CONFIGS:
            raise ValueError(
                f"Invalid config_name '{config_name}' for spread. "
                f"Valid options: {list(SPREAD_CONFIGS.keys())}"
            )
        return copy.deepcopy(SPREAD_CONFIGS[config_name])
    else:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Valid options: 'win_loss', 'spread'"
        )
