import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_team_features():
    """Sample feature DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "season": [2024] * 10,
        "week": list(range(1, 11)),
        "team": ["KC"] * 10,
        "points_scored_avg_3": np.random.uniform(20, 35, 10),
        "points_allowed_avg_3": np.random.uniform(15, 25, 10),
        "yards_per_play": np.random.uniform(5.0, 7.0, 10),
        "turnover_diff": np.random.randint(-3, 4, 10),
        "home_indicator": np.random.choice([0, 1], 10),
    })


@pytest.fixture
def sample_game_outcomes():
    """Sample game outcomes for testing models."""
    return pd.DataFrame({
        "game_id": range(1, 11),
        "home_win": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "home_score": [27, 17, 31, 24, 14, 28, 20, 35, 21, 10],
        "away_score": [24, 21, 28, 17, 24, 21, 24, 28, 17, 20],
        "spread": [-3, 4, -3, -7, 10, -7, 4, -7, -4, 10],
    })
