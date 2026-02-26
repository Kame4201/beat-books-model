"""Tests for feature store (save/load features)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.features.feature_store import FeatureStore, _EXT
from src.features.feature_config import FeatureMetadata


@pytest.fixture
def temp_feature_store():
    """Create temporary feature store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = FeatureStore(base_path=temp_dir)

    yield store

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_features_df():
    """Create sample features DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "team": ["KC"] * 10,
            "season": [2024] * 10,
            "week": list(range(1, 11)),
            "points_scored_avg_3": np.random.uniform(20, 30, 10),
            "points_allowed_avg_3": np.random.uniform(15, 25, 10),
            "home_win_pct": np.random.uniform(0.4, 0.6, 10),
        }
    )


def test_save_features(temp_feature_store, sample_features_df):
    """Test saving features to store."""
    feature_file = temp_feature_store.save(
        sample_features_df, version="v1.0", description="Test features"
    )

    # Check that files were created
    assert feature_file.exists()
    assert feature_file.suffix == _EXT

    # Check metadata file
    metadata_file = feature_file.parent / feature_file.name.replace(
        "features_", "metadata_"
    ).replace(_EXT, ".json")
    assert metadata_file.exists()


def test_load_features(temp_feature_store, sample_features_df):
    """Test loading features from store."""
    # Save features first
    temp_feature_store.save(
        sample_features_df, version="v1.0", description="Test features"
    )

    # Load features
    loaded_df, metadata = temp_feature_store.load(version="v1.0")

    # Check DataFrame matches
    pd.testing.assert_frame_equal(loaded_df, sample_features_df)

    # Check metadata
    assert metadata.version == "v1.0"
    assert metadata.description == "Test features"
    assert metadata.row_count == len(sample_features_df)
    assert metadata.season_range == (2024, 2024)


def test_list_versions(temp_feature_store, sample_features_df):
    """Test listing available versions."""
    # Initially empty
    assert temp_feature_store.list_versions() == []

    # Save v1.0
    temp_feature_store.save(sample_features_df, version="v1.0")
    assert "v1.0" in temp_feature_store.list_versions()

    # Save v1.1
    temp_feature_store.save(sample_features_df, version="v1.1")
    versions = temp_feature_store.list_versions()
    assert "v1.0" in versions
    assert "v1.1" in versions


def test_list_timestamps(temp_feature_store, sample_features_df):
    """Test listing timestamps for a version."""
    # Save multiple times
    temp_feature_store.save(sample_features_df, version="v1.0")
    temp_feature_store.save(sample_features_df, version="v1.0")

    timestamps = temp_feature_store.list_timestamps(version="v1.0")
    assert len(timestamps) == 2


def test_get_metadata_only(temp_feature_store, sample_features_df):
    """Test loading only metadata without full DataFrame."""
    temp_feature_store.save(
        sample_features_df,
        version="v1.0",
        description="Test features",
        additional_info={"test": "value"},
    )

    # Load metadata only (should be fast)
    metadata = temp_feature_store.get_metadata(version="v1.0")

    assert isinstance(metadata, FeatureMetadata)
    assert metadata.version == "v1.0"
    assert metadata.row_count == len(sample_features_df)
    assert metadata.additional_info["test"] == "value"


def test_load_specific_timestamp(temp_feature_store, sample_features_df):
    """Test loading a specific timestamp."""
    # Save first version
    temp_feature_store.save(sample_features_df, version="v1.0")

    # Get timestamp
    timestamps = temp_feature_store.list_timestamps(version="v1.0")
    timestamp = timestamps[0]

    # Modify DataFrame
    modified_df = sample_features_df.copy()
    modified_df["new_feature"] = 999

    # Save modified version
    temp_feature_store.save(modified_df, version="v1.0")

    # Load first timestamp (should not have new_feature)
    loaded_df, metadata = temp_feature_store.load(version="v1.0", timestamp=timestamp)
    assert "new_feature" not in loaded_df.columns

    # Load latest (should have new_feature)
    loaded_df_latest, _ = temp_feature_store.load(version="v1.0")
    assert "new_feature" in loaded_df_latest.columns


def test_delete_version(temp_feature_store, sample_features_df):
    """Test deleting a version."""
    temp_feature_store.save(sample_features_df, version="v1.0")
    temp_feature_store.save(sample_features_df, version="v2.0")

    # Check both exist
    versions = temp_feature_store.list_versions()
    assert "v1.0" in versions
    assert "v2.0" in versions

    # Delete v1.0
    temp_feature_store.delete_version("v1.0")

    # Check v1.0 is gone but v2.0 remains
    versions = temp_feature_store.list_versions()
    assert "v1.0" not in versions
    assert "v2.0" in versions


def test_load_nonexistent_version(temp_feature_store):
    """Test that loading nonexistent version raises error."""
    with pytest.raises(FileNotFoundError, match="No features found for version"):
        temp_feature_store.load(version="v99.0")


def test_feature_versioning_immutability(temp_feature_store, sample_features_df):
    """Test that old versions are never overwritten."""
    # Save v1.0
    temp_feature_store.save(sample_features_df, version="v1.0")
    original_df, _ = temp_feature_store.load(version="v1.0")

    # Save v1.0 again (should create new timestamp, not overwrite)
    modified_df = sample_features_df.copy()
    modified_df["extra"] = 1
    temp_feature_store.save(modified_df, version="v1.0")

    # Should now have 2 timestamps
    timestamps = temp_feature_store.list_timestamps(version="v1.0")
    assert len(timestamps) == 2

    # Load first timestamp - should be unchanged
    loaded_df, _ = temp_feature_store.load(version="v1.0", timestamp=timestamps[0])
    pd.testing.assert_frame_equal(loaded_df, original_df)
