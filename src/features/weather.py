"""
Weather feature provider for NFL game predictions.

Fetches historical weather data for game locations and converts them into
predictive features.  Uses the Open-Meteo Archive API (free, no key required)
with a configurable fallback to a null provider for dome stadiums and
environments without network access.

CRITICAL: This module only READS external data.  It never creates or
modifies database tables.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GameWeather:
    """Weather snapshot for a single game."""

    game_id: str
    temperature_f: float
    wind_speed_mph: float
    precipitation_inches: float
    humidity_pct: float
    is_dome: bool

    # Derived features (filled by compute_derived)
    wind_chill: Optional[float] = None

    def compute_derived(self) -> "GameWeather":
        """Compute derived weather features in-place and return self."""
        if self.temperature_f <= 50 and self.wind_speed_mph > 3:
            # NWS wind chill formula
            t = self.temperature_f
            v = self.wind_speed_mph
            self.wind_chill = (
                35.74 + 0.6215 * t - 35.75 * (v ** 0.16) + 0.4275 * t * (v ** 0.16)
            )
        else:
            self.wind_chill = self.temperature_f
        return self

    def to_feature_dict(self) -> Dict[str, float]:
        """Return flat dict suitable for merging into a feature DataFrame."""
        self.compute_derived()
        return {
            "weather_temperature_f": self.temperature_f,
            "weather_wind_speed_mph": self.wind_speed_mph,
            "weather_precipitation_in": self.precipitation_inches,
            "weather_humidity_pct": self.humidity_pct,
            "weather_is_dome": float(self.is_dome),
            "weather_wind_chill": self.wind_chill if self.wind_chill is not None else self.temperature_f,
        }


# ---------------------------------------------------------------------------
# NFL Stadium coordinates (subset — extend as needed)
# ---------------------------------------------------------------------------

STADIUM_COORDINATES: Dict[str, tuple[float, float, bool]] = {
    # team_abbr: (latitude, longitude, is_dome)
    "ARI": (33.5276, -112.2626, True),
    "ATL": (33.7554, -84.4010, True),
    "BAL": (39.2780, -76.6227, False),
    "BUF": (42.7738, -78.7870, False),
    "CAR": (35.2258, -80.8528, False),
    "CHI": (41.8623, -87.6167, False),
    "CIN": (39.0954, -84.5160, False),
    "CLE": (41.5061, -81.6995, False),
    "DAL": (32.7473, -97.0945, True),
    "DEN": (39.7439, -105.0201, False),
    "DET": (42.3400, -83.0456, True),
    "GB": (44.5013, -88.0622, False),
    "HOU": (29.6847, -95.4107, True),
    "IND": (39.7601, -86.1639, True),
    "JAX": (30.3239, -81.6373, False),
    "KC": (39.0489, -94.4839, False),
    "LA": (33.9534, -118.3387, True),
    "LAC": (33.9534, -118.3387, True),
    "LV": (36.0908, -115.1833, True),
    "MIA": (25.9580, -80.2389, False),
    "MIN": (44.9736, -93.2575, True),
    "NE": (42.0909, -71.2643, False),
    "NO": (29.9511, -90.0812, True),
    "NYG": (40.8128, -74.0742, False),
    "NYJ": (40.8128, -74.0742, False),
    "PHI": (39.9008, -75.1675, False),
    "PIT": (40.4468, -80.0158, False),
    "SEA": (47.5952, -122.3316, False),
    "SF": (37.4032, -121.9698, False),
    "TB": (27.9759, -82.5033, False),
    "TEN": (36.1665, -86.7713, False),
    "WAS": (38.9076, -76.8645, False),
}


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class WeatherProvider(ABC):
    """Abstract base for weather data backends."""

    @abstractmethod
    def get_game_weather(
        self, game_id: str, home_team: str, game_date: str
    ) -> GameWeather:
        """Return weather for a single game."""

    def get_bulk_weather(
        self, games: List[Dict[str, str]]
    ) -> List[GameWeather]:
        """Fetch weather for multiple games. Default: iterate."""
        return [
            self.get_game_weather(g["game_id"], g["home_team"], g["game_date"])
            for g in games
        ]


class NullWeatherProvider(WeatherProvider):
    """
    Fallback provider that returns neutral / dome-like weather.

    Use this when:
    - No API key / network access is available
    - Running unit tests with deterministic data
    """

    def get_game_weather(
        self, game_id: str, home_team: str, game_date: str
    ) -> GameWeather:
        is_dome = STADIUM_COORDINATES.get(home_team, (0, 0, False))[2]
        return GameWeather(
            game_id=game_id,
            temperature_f=72.0 if is_dome else 65.0,
            wind_speed_mph=0.0 if is_dome else 5.0,
            precipitation_inches=0.0,
            humidity_pct=50.0,
            is_dome=is_dome,
        )


class OpenMeteoWeatherProvider(WeatherProvider):
    """
    Fetches historical weather from the Open-Meteo Archive API.

    Requires network access but **no API key**.
    https://open-meteo.com/en/docs/historical-weather-api
    """

    API_URL = "https://archive-api.open-meteo.com/v1/archive"

    def get_game_weather(
        self, game_id: str, home_team: str, game_date: str
    ) -> GameWeather:
        coords = STADIUM_COORDINATES.get(home_team)
        if coords is None:
            return NullWeatherProvider().get_game_weather(game_id, home_team, game_date)

        lat, lon, is_dome = coords

        if is_dome:
            return GameWeather(
                game_id=game_id,
                temperature_f=72.0,
                wind_speed_mph=0.0,
                precipitation_inches=0.0,
                humidity_pct=50.0,
                is_dome=True,
            )

        try:
            import requests  # optional dependency

            resp = requests.get(
                self.API_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": game_date,
                    "end_date": game_date,
                    "daily": "temperature_2m_mean,windspeed_10m_max,precipitation_sum,relative_humidity_2m_mean",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                    "precipitation_unit": "inch",
                    "timezone": "America/New_York",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("daily", {})

            return GameWeather(
                game_id=game_id,
                temperature_f=data.get("temperature_2m_mean", [65.0])[0] or 65.0,
                wind_speed_mph=data.get("windspeed_10m_max", [5.0])[0] or 5.0,
                precipitation_inches=data.get("precipitation_sum", [0.0])[0] or 0.0,
                humidity_pct=data.get("relative_humidity_2m_mean", [50.0])[0] or 50.0,
                is_dome=False,
            )
        except Exception:
            # Graceful degradation to null provider
            return NullWeatherProvider().get_game_weather(game_id, home_team, game_date)


def get_weather_provider(use_api: bool = False) -> WeatherProvider:
    """Factory function. Returns NullWeatherProvider by default."""
    if use_api:
        return OpenMeteoWeatherProvider()
    return NullWeatherProvider()
