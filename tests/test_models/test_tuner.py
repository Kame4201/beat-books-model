"""Tests for the hyperparameter tuning module."""

import numpy as np
import pandas as pd
import pytest

from src.models.tuner import grid_search, TuningResult, _expand_grid


@pytest.fixture
def dataset():
    rng = np.random.RandomState(1)
    n = 100
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y_cls = pd.Series((X["f1"] + rng.randn(n) * 0.5 > 0).astype(int))
    y_reg = pd.Series(X["f1"] * 2 + rng.randn(n))
    return X.iloc[:70], y_cls.iloc[:70], y_reg.iloc[:70], X.iloc[70:], y_cls.iloc[70:], y_reg.iloc[70:]


class TestExpandGrid:
    def test_simple(self):
        grid = {"a": [1, 2], "b": ["x", "y"]}
        combos = _expand_grid(grid)
        assert len(combos) == 4

    def test_single_param(self):
        combos = _expand_grid({"a": [1, 2, 3]})
        assert len(combos) == 3

    def test_empty(self):
        combos = _expand_grid({})
        assert len(combos) == 1  # single empty dict


class TestGridSearch:
    def test_win_loss_baseline(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        result = grid_search(
            "win_loss", "baseline",
            {"C": [0.1, 1.0, 10.0]},
            Xtr, yctr, Xte, ycte,
            optimize_metric="accuracy",
        )
        assert isinstance(result, TuningResult)
        assert len(result.trials) == 3
        assert result.best_trial is not None
        assert "accuracy" in result.best_trial.metrics

    def test_spread_ridge(self, dataset):
        Xtr, _, yrtr, Xte, _, yrte = dataset
        result = grid_search(
            "spread", "ridge",
            {"alpha": [0.1, 1.0]},
            Xtr, yrtr, Xte, yrte,
            optimize_metric="mae",
            higher_is_better=False,
        )
        assert len(result.trials) == 2
        best = result.best_trial
        # best should have lower MAE
        worst = max(result.trials, key=lambda t: t.metrics["mae"])
        assert best.metrics["mae"] <= worst.metrics["mae"]

    def test_to_dataframe(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        result = grid_search(
            "win_loss", "baseline",
            {"C": [0.5, 1.0]},
            Xtr, yctr, Xte, ycte,
        )
        df = result.to_dataframe()
        assert len(df) == 2
        assert "C" in df.columns
        assert "accuracy" in df.columns

    def test_invalid_model_type(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        with pytest.raises(ValueError, match="Unknown model_type"):
            grid_search("bad", "baseline", {"C": [1]}, Xtr, yctr, Xte, ycte)
