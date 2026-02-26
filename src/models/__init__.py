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
    # Core classes (lazy to avoid requiring DATABASE_URL at import time)
    "BasePredictor",
    "WinLossModel",
    "SpreadModel",
    "ModelRegistry",
    # Configuration
    "get_config",
    "WIN_LOSS_CONFIGS",
    "SPREAD_CONFIGS",
]


def __getattr__(name: str):
    if name == "BasePredictor":
        from src.models.base_predictor import BasePredictor

        return BasePredictor
    if name == "WinLossModel":
        from src.models.win_loss_model import WinLossModel

        return WinLossModel
    if name == "SpreadModel":
        from src.models.spread_model import SpreadModel

        return SpreadModel
    if name == "ModelRegistry":
        from src.models.model_registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
