"""Tests for the pipeline runner CLI and orchestration logic."""

import pytest
from src.pipeline.runner import (
    parse_season_range,
    split_train_test,
    build_parser,
)


class TestParseSeasonRange:
    def test_range_format(self):
        assert parse_season_range("2020-2024") == [2020, 2021, 2022, 2023, 2024]

    def test_single_season(self):
        assert parse_season_range("2024") == [2024]

    def test_comma_separated(self):
        assert parse_season_range("2020,2022,2024") == [2020, 2022, 2024]

    def test_range_single_year(self):
        assert parse_season_range("2023-2023") == [2023]


class TestSplitTrainTest:
    def test_basic_split(self):
        seasons = [2020, 2021, 2022, 2023, 2024]
        train, test = split_train_test(seasons, backtest_start=2023)
        assert train == [2020, 2021, 2022]
        assert test == [2023, 2024]

    def test_all_test(self):
        seasons = [2023, 2024]
        train, test = split_train_test(seasons, backtest_start=2023)
        assert train == []
        assert test == [2023, 2024]

    def test_all_train(self):
        seasons = [2020, 2021, 2022]
        train, test = split_train_test(seasons, backtest_start=2025)
        assert train == [2020, 2021, 2022]
        assert test == []


class TestBuildParser:
    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(["--seasons", "2020-2024"])
        assert args.seasons == "2020-2024"
        assert args.model == "win_loss"
        assert args.variant == "baseline"

    def test_all_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--seasons", "2020-2024",
            "--model", "spread",
            "--variant", "xgboost",
            "--backtest-start", "2023",
            "--output-dir", "/tmp/out",
            "--no-save",
        ])
        assert args.model == "spread"
        assert args.variant == "xgboost"
        assert args.backtest_start == 2023
        assert args.output_dir == "/tmp/out"
        assert args.no_save is True

    def test_help_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
