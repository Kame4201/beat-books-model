"""
Models package for beat-books-model.

Provides:
- BasePredictor: Abstract base class for all models
- WinLossModel: Binary classifier for home team win/loss
- SpreadModel: Regressor for point spread prediction
- ModelRegistry: Model versioning and metadata tracking
- Model configurations and hyperparameters
"""

from src.models.base_predictor import BasePredictor
from src.models.win_loss_model import WinLossModel
from src.models.spread_model import SpreadModel
from src.models.model_registry import ModelRegistry
from src.models.model_config import (
    get_config,
    WIN_LOSS_CONFIGS,
    SPREAD_CONFIGS,
)

__all__ = [
    "BasePredictor",
    "WinLossModel",
    "SpreadModel",
    "ModelRegistry",
    "get_config",
    "WIN_LOSS_CONFIGS",
    "SPREAD_CONFIGS",
]
