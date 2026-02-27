"""
Model manager: loads the best trained model from the registry and caches it.

Used by the /predictions/predict endpoint for real inference.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from src.models.base_predictor import BasePredictor
from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """No trained model artifact is available."""


class ModelManager:
    """Loads and caches the best win/loss model from the registry."""

    def __init__(self, artifacts_path: str):
        self._artifacts_path = artifacts_path
        self._model: Optional[BasePredictor] = None
        self._model_meta: Optional[dict] = None

    @property
    def model(self) -> BasePredictor:
        if self._model is None:
            self._load()
        return self._model

    @property
    def model_meta(self) -> dict:
        if self._model_meta is None:
            self._load()
        return self._model_meta

    def _load(self) -> None:
        """Load the best model from the registry, or a specific one via MODEL_ID env."""
        registry_path = Path(self._artifacts_path) / "registry.json"
        if not registry_path.exists():
            raise ModelNotFoundError(
                f"No registry.json found at {self._artifacts_path}. "
                "Train a model first: python scripts/train_baseline.py"
            )

        registry = ModelRegistry(self._artifacts_path)

        model_id = os.environ.get("MODEL_ID")
        if model_id:
            meta = registry.get_model(model_id)
            if meta is None:
                raise ModelNotFoundError(f"MODEL_ID={model_id} not found in registry")
        else:
            meta = registry.get_best_model("win_loss_classifier", "accuracy")
            if meta is None:
                raise ModelNotFoundError(
                    "No win_loss_classifier models in registry. "
                    "Train a model first: python scripts/train_baseline.py"
                )

        artifact_path = Path(self._artifacts_path) / meta["artifact_path"]
        if not artifact_path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found: {artifact_path}. "
                "Re-train or restore the .joblib file."
            )

        logger.info(
            "Loading model %s (accuracy=%.3f) from %s",
            meta["model_id"],
            meta["metrics"].get("accuracy", 0),
            artifact_path,
        )
        self._model = BasePredictor.load(str(artifact_path))
        self._model_meta = meta

    def reload(self) -> None:
        """Force reload the model (e.g. after re-training)."""
        self._model = None
        self._model_meta = None
        self._load()
