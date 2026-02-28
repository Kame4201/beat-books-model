"""Tests for weather feature provider."""

import pytest
from src.features.weather import (
    GameWeather,
    NullWeatherProvider,
    OpenMeteoWeatherProvider,
    get_weather_provider,
    STADIUM_COORDINATES,
)


class TestGameWeather:
    def test_feature_dict_keys(self):
        gw = GameWeather(
            game_id="g1", temperature_f=35.0, wind_speed_mph=15.0,
            precipitation_inches=0.1, humidity_pct=60.0, is_dome=False,
        )
        d = gw.to_feature_dict()
        expected_keys = {
            "weather_temperature_f", "weather_wind_speed_mph",
            "weather_precipitation_in", "weather_humidity_pct",
            "weather_is_dome", "weather_wind_chill",
        }
        assert set(d.keys()) == expected_keys

    def test_wind_chill_cold_windy(self):
        gw = GameWeather(
            game_id="g1", temperature_f=30.0, wind_speed_mph=20.0,
            precipitation_inches=0.0, humidity_pct=40.0, is_dome=False,
        )
        gw.compute_derived()
        assert gw.wind_chill is not None
        assert gw.wind_chill < gw.temperature_f  # wind chill should be lower

    def test_wind_chill_warm(self):
        gw = GameWeather(
            game_id="g1", temperature_f=72.0, wind_speed_mph=10.0,
            precipitation_inches=0.0, humidity_pct=50.0, is_dome=False,
        )
        gw.compute_derived()
        assert gw.wind_chill == gw.temperature_f

    def test_dome_features(self):
        gw = GameWeather(
            game_id="g1", temperature_f=72.0, wind_speed_mph=0.0,
            precipitation_inches=0.0, humidity_pct=50.0, is_dome=True,
        )
        d = gw.to_feature_dict()
        assert d["weather_is_dome"] == 1.0
        assert d["weather_wind_speed_mph"] == 0.0


class TestNullProvider:
    def test_dome_team(self):
        provider = NullWeatherProvider()
        gw = provider.get_game_weather("g1", "DAL", "2024-12-01")
        assert gw.is_dome is True
        assert gw.temperature_f == 72.0
        assert gw.wind_speed_mph == 0.0

    def test_outdoor_team(self):
        provider = NullWeatherProvider()
        gw = provider.get_game_weather("g1", "GB", "2024-12-01")
        assert gw.is_dome is False
        assert gw.temperature_f == 65.0

    def test_unknown_team(self):
        provider = NullWeatherProvider()
        gw = provider.get_game_weather("g1", "UNKNOWN", "2024-12-01")
        assert gw.is_dome is False  # default

    def test_bulk_weather(self):
        provider = NullWeatherProvider()
        games = [
            {"game_id": "g1", "home_team": "KC", "game_date": "2024-09-08"},
            {"game_id": "g2", "home_team": "DAL", "game_date": "2024-09-08"},
        ]
        results = provider.get_bulk_weather(games)
        assert len(results) == 2


class TestOpenMeteoProvider:
    def test_dome_bypass(self):
        """Dome stadiums should return fixed weather without API call."""
        provider = OpenMeteoWeatherProvider()
        gw = provider.get_game_weather("g1", "ATL", "2024-12-01")
        assert gw.is_dome is True
        assert gw.temperature_f == 72.0

    def test_fallback_on_missing_team(self):
        """Unknown team should fall back to null provider."""
        provider = OpenMeteoWeatherProvider()
        gw = provider.get_game_weather("g1", "INVALID", "2024-12-01")
        assert gw.game_id == "g1"

    def test_fallback_on_network_error(self):
        """API failure should degrade to null provider."""
        provider = OpenMeteoWeatherProvider()
        provider.API_URL = "http://localhost:1/nonexistent"
        gw = provider.get_game_weather("g1", "GB", "2024-12-01")
        assert gw.game_id == "g1"
        assert gw.temperature_f == 65.0  # null provider default


class TestFactory:
    def test_default_is_null(self):
        p = get_weather_provider()
        assert isinstance(p, NullWeatherProvider)

    def test_api_flag(self):
        p = get_weather_provider(use_api=True)
        assert isinstance(p, OpenMeteoWeatherProvider)


class TestStadiumCoordinates:
    def test_all_32_teams(self):
        assert len(STADIUM_COORDINATES) == 32

    def test_coordinate_format(self):
        for team, (lat, lon, dome) in STADIUM_COORDINATES.items():
            assert -90 <= lat <= 90, f"{team} lat out of range"
            assert -180 <= lon <= 180, f"{team} lon out of range"
            assert isinstance(dome, bool)
