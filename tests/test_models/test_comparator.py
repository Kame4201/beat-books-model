"""Tests for the model comparison framework."""

import numpy as np
import pandas as pd
import pytest

from src.models.comparator import compare_models, ComparisonReport


@pytest.fixture
def dataset():
    rng = np.random.RandomState(0)
    n = 120
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
        }
    )
    y_cls = pd.Series((X["feat_a"] + rng.randn(n) * 0.3 > 0).astype(int))
    y_reg = pd.Series(X["feat_a"] * 3 + rng.randn(n))
    return X.iloc[:80], y_cls.iloc[:80], y_reg.iloc[:80], X.iloc[80:], y_cls.iloc[80:], y_reg.iloc[80:]


class TestCompareModels:
    def test_win_loss_all_variants(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        report = compare_models("win_loss", Xtr, yctr, Xte, ycte)
        assert isinstance(report, ComparisonReport)
        assert len(report.entries) == 3  # baseline, xgboost, lightgbm

    def test_spread_selected_variants(self, dataset):
        Xtr, _, yrtr, Xte, _, yrte = dataset
        report = compare_models("spread", Xtr, yrtr, Xte, yrte, variants=["baseline", "ridge"])
        assert len(report.entries) == 2

    def test_to_dataframe(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        report = compare_models("win_loss", Xtr, yctr, Xte, ycte, variants=["baseline"])
        df = report.to_dataframe()
        assert "variant" in df.columns
        assert "accuracy" in df.columns

    def test_best(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        report = compare_models("win_loss", Xtr, yctr, Xte, ycte)
        best = report.best("accuracy")
        assert best.variant in ["baseline", "xgboost", "lightgbm"]

    def test_custom_hyperparameters(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        report = compare_models(
            "win_loss", Xtr, yctr, Xte, ycte,
            variants=["baseline"],
            custom_hyperparameters={"baseline": {"C": 0.5}},
        )
        assert len(report.entries) == 1

    def test_invalid_model_type(self, dataset):
        Xtr, yctr, _, Xte, ycte, _ = dataset
        with pytest.raises(ValueError, match="Unknown model_type"):
            compare_models("invalid", Xtr, yctr, Xte, ycte)
