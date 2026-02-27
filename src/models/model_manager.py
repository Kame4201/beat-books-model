"""
ModelManager: loads and caches a trained model artifact for inference.

Reads from the model registry to find the best (or specified) model,
loads the .joblib file, and caches it for repeated predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.models.base_predictor import BasePredictor
from src.models.model_registry import ModelRegistry


class ModelNotFoundError(Exception):
    """Raised when no trained model artifact is available."""


class ModelManager:
    """
    Singleton-style manager that loads and caches a trained model.

    Usage:
        manager = ModelManager()
        model = manager.get_model()          # loads best or env-specified model
        info = manager.get_model_info()       # metadata dict
    """

    def __init__(
        self,
        artifacts_path: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self._artifacts_path = artifacts_path or settings.MODEL_ARTIFACTS_PATH
        self._requested_model_id = model_id
        self._model: Optional[BasePredictor] = None
        self._model_meta: Optional[dict] = None

    def get_model(self) -> BasePredictor:
        """Return the cached model, loading it on first call."""
        if self._model is not None:
            return self._model

        registry = ModelRegistry(registry_dir=self._artifacts_path)
        meta = self._resolve_model(registry)
        if meta is None:
            raise ModelNotFoundError(
                f"No trained model found in {self._artifacts_path}. "
                "Run `python scripts/train_baseline.py --synthetic` to create one."
            )

        artifact_path = Path(self._artifacts_path) / meta["artifact_path"]
        if not artifact_path.exists():
            raise ModelNotFoundError(
                f"Model artifact {artifact_path} not found on disk."
            )

        self._model = BasePredictor.load(str(artifact_path))
        self._model_meta = meta
        return self._model

    def get_model_info(self) -> dict:
        """Return metadata for the loaded model."""
        if self._model_meta is None:
            self.get_model()  # triggers load
        return self._model_meta  # type: ignore[return-value]

    def get_feature_names(self) -> list[str]:
        """Return the feature column names the model expects."""
        info = self.get_model_info()
        return info.get("feature_names", [])

    def _resolve_model(self, registry: ModelRegistry) -> Optional[dict]:
        """Find the right model entry from registry."""
        # Explicit model ID from env or constructor
        model_id = self._requested_model_id
        if model_id:
            return registry.get_model(model_id)

        # Best by accuracy
        best = registry.get_best_model(
            model_type="win_loss_classifier",
            metric_name="accuracy",
            minimize=False,
        )
        return best
