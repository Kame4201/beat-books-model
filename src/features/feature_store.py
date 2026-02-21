"""
Feature store for saving and loading computed features.

Supports versioning and metadata tracking to ensure reproducibility.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.features.feature_config import FeatureMetadata, FEATURE_VERSION


class FeatureStore:
    """
    Manages storage and retrieval of computed features.

    Features are stored as parquet files with accompanying metadata JSON.
    Versioning ensures old feature sets are never overwritten.
    """

    def __init__(self, base_path: str = "model_artifacts/features"):
        """
        Initialize feature store.

        Args:
            base_path: Base directory for storing features
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        features_df: pd.DataFrame,
        version: str = FEATURE_VERSION,
        description: str = "",
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save features with metadata.

        Args:
            features_df: DataFrame with computed features
            version: Feature version string
            description: Human-readable description
            additional_info: Additional metadata to store

        Returns:
            Path to saved feature file
        """
        # Create version directory
        version_dir = self.base_path / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename (microseconds to avoid collisions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        feature_file = version_dir / f"features_{timestamp}.parquet"
        metadata_file = version_dir / f"metadata_{timestamp}.json"

        # Extract season range if available
        season_range: tuple[int | None, int | None] = (None, None)
        if "season" in features_df.columns:
            season_range = (
                int(features_df["season"].min()),
                int(features_df["season"].max()),
            )

        # Create metadata
        metadata = FeatureMetadata(
            version=version,
            creation_date=datetime.now(),
            feature_names=features_df.columns.tolist(),
            row_count=len(features_df),
            season_range=season_range,
            description=description,
            additional_info=additional_info or {},
        )

        # Save features as parquet (efficient columnar format)
        features_df.to_parquet(feature_file, index=False, compression="snappy")

        # Save metadata as JSON
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f"Saved {len(features_df)} rows with {len(features_df.columns)} features")
        print(f"  Features: {feature_file}")
        print(f"  Metadata: {metadata_file}")

        return feature_file

    def load(
        self, version: str = FEATURE_VERSION, timestamp: Optional[str] = None
    ) -> tuple[pd.DataFrame, FeatureMetadata]:
        """
        Load features and metadata.

        Args:
            version: Feature version to load
            timestamp: Specific timestamp to load (None = latest)

        Returns:
            Tuple of (features DataFrame, metadata)

        Raises:
            FileNotFoundError: If no features found for version
        """
        version_dir = self.base_path / version

        if not version_dir.exists():
            raise FileNotFoundError(f"No features found for version {version}")

        # Find feature files
        feature_files = sorted(version_dir.glob("features_*.parquet"))

        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {version_dir}")

        # Select file
        if timestamp:
            feature_file = version_dir / f"features_{timestamp}.parquet"
            metadata_file = version_dir / f"metadata_{timestamp}.json"

            if not feature_file.exists():
                raise FileNotFoundError(f"No features found for timestamp {timestamp}")
        else:
            # Load latest
            feature_file = feature_files[-1]
            # Get corresponding metadata file
            timestamp_str = feature_file.stem.replace("features_", "")
            metadata_file = version_dir / f"metadata_{timestamp_str}.json"

        # Load features
        features_df = pd.read_parquet(feature_file)

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata_dict = json.load(f)
            metadata = FeatureMetadata.from_dict(metadata_dict)

        print(
            f"Loaded {len(features_df)} rows with {len(features_df.columns)} features"
        )
        print(f"  Version: {metadata.version}")
        print(f"  Created: {metadata.creation_date}")
        print(f"  Season range: {metadata.season_range}")

        return features_df, metadata

    def list_versions(self) -> List[str]:
        """
        List all available feature versions.

        Returns:
            List of version strings
        """
        if not self.base_path.exists():
            return []

        versions = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        return sorted(versions)

    def list_timestamps(self, version: str = FEATURE_VERSION) -> List[str]:
        """
        List all available timestamps for a version.

        Args:
            version: Feature version to check

        Returns:
            List of timestamp strings
        """
        version_dir = self.base_path / version

        if not version_dir.exists():
            return []

        feature_files = sorted(version_dir.glob("features_*.parquet"))
        timestamps = [f.stem.replace("features_", "") for f in feature_files]

        return timestamps

    def get_metadata(
        self, version: str = FEATURE_VERSION, timestamp: Optional[str] = None
    ) -> FeatureMetadata:
        """
        Load only metadata without loading full feature DataFrame.

        Args:
            version: Feature version
            timestamp: Specific timestamp (None = latest)

        Returns:
            FeatureMetadata object
        """
        version_dir = self.base_path / version

        if not version_dir.exists():
            raise FileNotFoundError(f"No features found for version {version}")

        # Find metadata files
        metadata_files = sorted(version_dir.glob("metadata_*.json"))

        if not metadata_files:
            raise FileNotFoundError(f"No metadata found in {version_dir}")

        # Select file
        if timestamp:
            metadata_file = version_dir / f"metadata_{timestamp}.json"
        else:
            metadata_file = metadata_files[-1]

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata_dict = json.load(f)
            return FeatureMetadata.from_dict(metadata_dict)

    def delete_version(self, version: str) -> None:
        """
        Delete an entire feature version directory.

        WARNING: This permanently deletes all features and metadata for the version.

        Args:
            version: Version to delete
        """
        version_dir = self.base_path / version

        if not version_dir.exists():
            print(f"Version {version} does not exist")
            return

        # Delete all files in version directory
        for file in version_dir.iterdir():
            file.unlink()

        # Delete directory
        version_dir.rmdir()

        print(f"Deleted version {version}")
