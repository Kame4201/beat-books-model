"""
Hyperparameter tuning via grid search with walk-forward-safe evaluation.

Iterates over a parameter grid for a given model variant, trains each
configuration, evaluates on a held-out set, and returns the best
hyperparameters together with all trial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.win_loss_model import WinLossModel
from src.models.spread_model import SpreadModel


@dataclass
class TuningTrial:
    """Single hyperparameter trial."""

    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass
class TuningResult:
    """Outcome of a tuning run."""

    model_type: str
    variant: str
    optimize_metric: str
    higher_is_better: bool
    trials: List[TuningTrial] = field(default_factory=list)

    @property
    def best_trial(self) -> TuningTrial:
        if not self.trials:
            raise ValueError("No trials recorded")
        key = lambda t: t.metrics.get(self.optimize_metric, float("-inf"))
        return (max if self.higher_is_better else min)(self.trials, key=key)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for t in self.trials:
            row = dict(t.hyperparameters)
            row.update(t.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


_MODEL_CLASSES = {
    "win_loss": WinLossModel,
    "spread": SpreadModel,
}


def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of parameter lists."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def grid_search(
    model_type: str,
    variant: str,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimize_metric: str = "accuracy",
    higher_is_better: bool = True,
) -> TuningResult:
    """
    Exhaustive grid search over *param_grid* for a single model variant.

    Args:
        model_type: "win_loss" or "spread"
        variant: model variant name (e.g. "xgboost")
        param_grid: dict mapping param name → list of values
        X_train / y_train: training data
        X_test / y_test: evaluation data
        optimize_metric: metric name returned by model.evaluate()
        higher_is_better: direction of improvement

    Returns:
        TuningResult containing all trials and the best set of
        hyperparameters.
    """
    if model_type not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {list(_MODEL_CLASSES)}"
        )

    cls = _MODEL_CLASSES[model_type]
    combos = _expand_grid(param_grid)

    result = TuningResult(
        model_type=model_type,
        variant=variant,
        optimize_metric=optimize_metric,
        higher_is_better=higher_is_better,
    )

    for hp in combos:
        model = cls(model_variant=variant, custom_hyperparameters=hp)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        result.trials.append(TuningTrial(hyperparameters=hp, metrics=metrics))

    return result
