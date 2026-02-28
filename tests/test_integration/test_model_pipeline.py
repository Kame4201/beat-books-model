"""
Integration tests for end-to-end model pipeline.

These tests exercise the full train -> predict -> evaluate -> registry cycle
using synthetic data.  They are marked ``integration`` so CI can skip them
when DATABASE_URL is unavailable, but they do NOT require a real database --
they test the in-memory pipeline only.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

from src.models.win_loss_model import WinLossModel
from src.models.spread_model import SpreadModel
from src.models.model_registry import ModelRegistry
from src.features.feature_store import FeatureStore


def _make_dataset(n: int = 200):
    """Generate a deterministic synthetic dataset for testing."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        {
            "points_scored_avg_3": rng.uniform(17, 35, n),
            "points_allowed_avg_3": rng.uniform(15, 30, n),
            "yards_per_play": rng.uniform(4.5, 7.0, n),
            "turnover_diff": rng.randint(-3, 4, n).astype(float),
            "home_indicator": rng.choice([0, 1], n).astype(float),
        }
    )
    diff = X["points_scored_avg_3"] - X["points_allowed_avg_3"]
    noise = rng.normal(0, 3, n)
    y_win = (diff + 3 * X["home_indicator"] + noise > 0).astype(int)
    y_spread = -(diff + 3 * X["home_indicator"] + noise)
    return X, pd.Series(y_win, name="home_win"), pd.Series(y_spread, name="spread")


@pytest.mark.integration
class TestWinLossEndToEnd:
    """Full lifecycle for the Win/Loss classifier."""

    def test_train_predict_evaluate(self):
        X, y_win, _ = _make_dataset()
        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y_win.iloc[:150], y_win.iloc[150:]

        model = WinLossModel(model_variant="baseline")
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

        probas = model.predict_proba(X_test)
        assert probas.shape[0] == len(X_test)

        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_load_roundtrip(self, tmp_path):
        X, y_win, _ = _make_dataset(100)
        model = WinLossModel(model_variant="baseline")
        model.train(X, y_win)

        path = str(tmp_path / "wl_model.joblib")
        model.save(path)

        loaded = WinLossModel.load(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


@pytest.mark.integration
class TestSpreadEndToEnd:
    """Full lifecycle for the Spread regressor."""

    def test_train_predict_evaluate(self):
        X, _, y_spread = _make_dataset()
        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y_spread.iloc[:150], y_spread.iloc[150:]

        model = SpreadModel(model_variant="ridge")
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

        metrics = model.evaluate(X_test, y_test)
        assert "mae" in metrics
        assert metrics["mae"] >= 0


@pytest.mark.integration
class TestFeatureStoreRoundtrip:
    """Full save -> load -> metadata cycle through the feature store."""

    def test_versioned_roundtrip(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            store = FeatureStore(base_path=tmp_dir)
            df = pd.DataFrame(
                {
                    "season": [2024] * 5,
                    "week": list(range(1, 6)),
                    "feat_a": np.random.rand(5),
                }
            )
            store.save(df, version="v_int_test", description="integration test")

            loaded, meta = store.load(version="v_int_test")
            pd.testing.assert_frame_equal(loaded, df)
            assert meta.version == "v_int_test"
            assert meta.row_count == 5
        finally:
            shutil.rmtree(tmp_dir)


@pytest.mark.integration
class TestRegistryLifecycle:
    """Register -> retrieve -> list models via ModelRegistry."""

    def test_register_and_retrieve(self, tmp_path):
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))

        X, y_win, _ = _make_dataset(80)
        model = WinLossModel(model_variant="baseline")
        model.train(X, y_win)

        feature_names = list(X.columns)
        model_id = registry.register_model(
            model_type="win_loss",
            model_name="test_wl",
            version="1.0.0",
            hyperparameters={"solver": "lbfgs"},
            feature_version="v1.0",
            train_seasons=[2022, 2023],
            test_seasons=[2024],
            metrics={"accuracy": 0.70},
            feature_names=feature_names,
            notes="integration test",
        )
        assert model_id is not None

        entry = registry.get_model(model_id)
        assert entry["model_name"] == "test_wl"

        entries = registry.list_models()
        assert len(entries) >= 1
