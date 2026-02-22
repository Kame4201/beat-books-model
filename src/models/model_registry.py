"""
Model registry for tracking trained model versions, metadata, and artifacts.

Stores:
- Model metadata: model_id, model_type, version, train_date, hyperparameters
- Training info: feature_version, train_seasons, test_seasons
- Performance metrics: accuracy, MAE, log_loss, etc.
- Artifact paths: where joblib model files are saved

Registry is stored as JSON file in model_artifacts/ directory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.core.config import settings


class ModelRegistry:
    """
    Manages model versioning and metadata storage.

    Each trained model gets:
    - Unique model_id (UUID)
    - Metadata JSON file with training info and metrics
    - Binary artifact file (.joblib) with trained model
    """

    def __init__(self, registry_dir: Optional[str] = None):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store registry and artifacts.
                          Defaults to settings.MODEL_ARTIFACTS_PATH
        """
        self.registry_dir = Path(registry_dir or settings.MODEL_ARTIFACTS_PATH)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self._load_or_create_registry()

    def _load_or_create_registry(self) -> None:
        """Load existing registry or create new one if it doesn't exist."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": []}
            self._save_registry()

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_type: str,
        model_name: str,
        version: str,
        hyperparameters: Dict[str, Any],
        feature_version: str,
        train_seasons: List[int],
        test_seasons: List[int],
        metrics: Dict[str, float],
        feature_names: List[str],
        notes: str = "",
    ) -> str:
        """
        Register a newly trained model.

        Args:
            model_type: Type of model ("win_loss_classifier", "spread_regressor")
            model_name: Specific model name ("logistic_regression", "xgboost_classifier", etc.)
            version: Model version (e.g., "1.0.0")
            hyperparameters: Model hyperparameters used during training
            feature_version: Version of feature engineering pipeline used
            train_seasons: List of seasons used for training (e.g., [2000, 2001, ..., 2022])
            test_seasons: List of seasons used for testing (e.g., [2023, 2024])
            metrics: Performance metrics (e.g., {"accuracy": 0.58, "log_loss": 0.65})
            feature_names: List of feature column names used
            notes: Optional notes about this model

        Returns:
            model_id: Unique identifier for this trained model

        Examples:
            >>> registry = ModelRegistry()
            >>> model_id = registry.register_model(
            ...     model_type="win_loss_classifier",
            ...     model_name="logistic_regression",
            ...     version="1.0.0",
            ...     hyperparameters={"C": 1.0, "max_iter": 1000},
            ...     feature_version="1.0.0",
            ...     train_seasons=list(range(2000, 2023)),
            ...     test_seasons=[2023, 2024],
            ...     metrics={"accuracy": 0.58, "log_loss": 0.65},
            ...     feature_names=["points_scored_avg_3", "points_allowed_avg_3"],
            ... )
        """
        model_id = str(uuid.uuid4())
        train_date = datetime.now(timezone.utc).isoformat()

        model_entry = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "version": version,
            "train_date": train_date,
            "hyperparameters": hyperparameters,
            "feature_version": feature_version,
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
            "metrics": metrics,
            "feature_names": feature_names,
            "artifact_path": f"{model_id}.joblib",
            "notes": notes,
        }

        self.registry["models"].append(model_entry)
        self._save_registry()

        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata by ID.

        Args:
            model_id: Unique model identifier

        Returns:
            Model metadata dictionary, or None if not found
        """
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None

    def list_models(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all registered models, optionally filtered.

        Args:
            model_type: Filter by model type (e.g., "win_loss_classifier")
            model_name: Filter by model name (e.g., "xgboost_classifier")

        Returns:
            List of model metadata dictionaries

        Examples:
            >>> registry = ModelRegistry()
            >>> all_models = registry.list_models()
            >>> win_loss_models = registry.list_models(model_type="win_loss_classifier")
            >>> xgboost_models = registry.list_models(model_name="xgboost_classifier")
        """
        models = self.registry["models"]

        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        if model_name:
            models = [m for m in models if m["model_name"] == model_name]

        return models

    def get_best_model(
        self,
        model_type: str,
        metric_name: str,
        minimize: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best-performing model of a given type.

        Args:
            model_type: Model type to filter by
            metric_name: Metric to optimize (e.g., "accuracy", "mae")
            minimize: If True, select model with lowest metric value.
                     If False, select model with highest metric value.

        Returns:
            Best model metadata, or None if no models found

        Examples:
            >>> registry = ModelRegistry()
            >>> best_win_loss = registry.get_best_model("win_loss_classifier", "accuracy")
            >>> best_spread = registry.get_best_model("spread_regressor", "mae", minimize=True)
        """
        models = self.list_models(model_type=model_type)
        if not models:
            return None

        # Filter models that have the requested metric
        models_with_metric = [m for m in models if metric_name in m.get("metrics", {})]
        if not models_with_metric:
            return None

        if minimize:
            return min(models_with_metric, key=lambda m: m["metrics"][metric_name])
        else:
            return max(models_with_metric, key=lambda m: m["metrics"][metric_name])

    def get_artifact_path(self, model_id: str) -> Optional[Path]:
        """
        Get full path to model artifact file.

        Args:
            model_id: Unique model identifier

        Returns:
            Path to .joblib artifact file, or None if model not found
        """
        model = self.get_model(model_id)
        if not model:
            return None

        return self.registry_dir / model["artifact_path"]

    def delete_model(self, model_id: str, delete_artifact: bool = True) -> bool:
        """
        Delete model from registry (and optionally delete artifact file).

        Args:
            model_id: Unique model identifier
            delete_artifact: If True, also delete the .joblib artifact file

        Returns:
            True if model was deleted, False if not found
        """
        model = self.get_model(model_id)
        if not model:
            return False

        # Remove from registry
        self.registry["models"] = [
            m for m in self.registry["models"] if m["model_id"] != model_id
        ]
        self._save_registry()

        # Optionally delete artifact file
        if delete_artifact:
            artifact_path = self.registry_dir / model["artifact_path"]
            if artifact_path.exists():
                artifact_path.unlink()

        return True

    def export_summary(self, output_path: Optional[str] = None) -> str:
        """
        Export human-readable summary of all registered models.

        Args:
            output_path: Optional path to save summary. If None, returns string.

        Returns:
            Summary string

        Examples:
            >>> registry = ModelRegistry()
            >>> print(registry.export_summary())
        """
        lines = ["=" * 80, "MODEL REGISTRY SUMMARY", "=" * 80, ""]

        for model in self.registry["models"]:
            lines.append(f"Model ID: {model['model_id']}")
            lines.append(f"Type: {model['model_type']}")
            lines.append(f"Name: {model['model_name']}")
            lines.append(f"Version: {model['version']}")
            lines.append(f"Trained: {model['train_date']}")
            lines.append(
                f"Train Seasons: {model['train_seasons'][0]}-{model['train_seasons'][-1]}"
            )
            lines.append(f"Test Seasons: {model['test_seasons']}")
            lines.append("Metrics:")
            for metric_name, metric_value in model["metrics"].items():
                lines.append(f"  {metric_name}: {metric_value:.4f}")
            lines.append("")
            lines.append("-" * 80)
            lines.append("")

        summary = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(summary)

        return summary
