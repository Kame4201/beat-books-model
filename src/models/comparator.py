"""
Model comparison framework.

Runs multiple model variants on identical train/test splits and produces a
ranked summary table.  Works with any BasePredictor subclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.win_loss_model import WinLossModel
from src.models.spread_model import SpreadModel


@dataclass
class ComparisonEntry:
    """Result of a single model evaluation."""

    model_type: str  # "win_loss" or "spread"
    variant: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass
class ComparisonReport:
    """Aggregated comparison across models."""

    entries: List[ComparisonEntry] = field(default_factory=list)

    # ------- helpers -------
    def to_dataframe(self) -> pd.DataFrame:
        """Flatten entries into a DataFrame for easy inspection."""
        rows = []
        for e in self.entries:
            row: dict[str, object] = {"model_type": e.model_type, "variant": e.variant}
            row.update(e.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def best(self, metric: str, higher_is_better: bool = True) -> ComparisonEntry:
        """Return the entry with the best value for *metric*."""
        if not self.entries:
            raise ValueError("No entries in comparison report")
        return (max if higher_is_better else min)(
            self.entries, key=lambda e: e.metrics.get(metric, float("-inf"))
        )


# ---------- public API ----------

_MODEL_CLASSES = {
    "win_loss": WinLossModel,
    "spread": SpreadModel,
}

_DEFAULT_VARIANTS = {
    "win_loss": ["baseline", "xgboost", "lightgbm"],
    "spread": ["baseline", "ridge", "xgboost", "lightgbm"],
}


def compare_models(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    variants: Optional[List[str]] = None,
    custom_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ComparisonReport:
    """
    Train and evaluate multiple model variants on the same data split.

    Args:
        model_type: "win_loss" or "spread"
        X_train / y_train: training data
        X_test / y_test: test data
        variants: list of variant names to compare (default: all)
        custom_hyperparameters: optional per-variant overrides,
            e.g. ``{"xgboost": {"n_estimators": 200}}``

    Returns:
        ComparisonReport with one entry per variant
    """
    if model_type not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {list(_MODEL_CLASSES)}"
        )

    cls = _MODEL_CLASSES[model_type]
    variants = variants or _DEFAULT_VARIANTS[model_type]
    custom_hyperparameters = custom_hyperparameters or {}

    report = ComparisonReport()
    for variant in variants:
        hp = custom_hyperparameters.get(variant)
        model = cls(model_variant=variant, custom_hyperparameters=hp)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        report.entries.append(
            ComparisonEntry(
                model_type=model_type,
                variant=variant,
                hyperparameters=model.training_metadata.get("hyperparameters", {}),
                metrics=metrics,
            )
        )

    return report
