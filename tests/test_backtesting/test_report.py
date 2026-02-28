"""Tests for backtest HTML report generation."""

import pytest
from pathlib import Path

from src.backtesting.types import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    BetSizingMethod,
)
from src.backtesting.report import generate_html_report, save_report


@pytest.fixture
def sample_result():
    config = BacktestConfig(
        model_id="test_model",
        feature_version="v1.0",
        start_date="2023-09-07",
        end_date="2024-02-11",
    )
    metrics = BacktestMetrics(
        overall_accuracy=0.58,
        home_accuracy=0.62,
        away_accuracy=0.54,
        favorite_accuracy=0.60,
        underdog_accuracy=0.55,
        log_loss=0.68,
        brier_score=0.24,
        total_bets=150,
        total_wagered=12000.0,
        total_profit=960.0,
        roi=8.0,
        win_rate=0.55,
        max_drawdown=1500.0,
        max_drawdown_pct=12.5,
        sharpe_ratio=1.2,
    )
    return BacktestResult(
        run_id="test-run-001",
        timestamp="2024-03-01T10:00:00",
        config=config,
        actual_start_date="2023-09-07",
        actual_end_date="2024-02-11",
        total_games=272,
        metrics=metrics,
        bankroll_curve={"week1": 10000, "week2": 10200, "week3": 10050, "week4": 10500},
        notes="Test run for report generation",
    )


class TestGenerateHtmlReport:
    def test_returns_valid_html(self, sample_result):
        html = generate_html_report(sample_result)
        assert "<!DOCTYPE html>" in html
        assert "Backtest Report" in html

    def test_contains_run_id(self, sample_result):
        html = generate_html_report(sample_result)
        assert "test-run-001" in html

    def test_contains_metrics(self, sample_result):
        html = generate_html_report(sample_result)
        assert "0.5800" in html  # overall accuracy
        assert "8.00" in html  # ROI
        assert "150" in html  # total bets

    def test_contains_bankroll_section(self, sample_result):
        html = generate_html_report(sample_result)
        # Should have either an image (matplotlib) or a text table
        assert "Bankroll" in html

    def test_notes_rendered(self, sample_result):
        html = generate_html_report(sample_result)
        assert "Test run for report generation" in html

    def test_no_notes(self, sample_result):
        sample_result.notes = None
        html = generate_html_report(sample_result)
        assert "<!DOCTYPE html>" in html

    def test_empty_bankroll(self, sample_result):
        sample_result.bankroll_curve = {}
        html = generate_html_report(sample_result)
        assert "No bankroll data" in html


class TestSaveReport:
    def test_creates_file(self, sample_result, tmp_path):
        path = str(tmp_path / "report.html")
        result_path = save_report(sample_result, path)
        assert result_path.exists()
        content = result_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(self, sample_result, tmp_path):
        path = str(tmp_path / "subdir" / "deep" / "report.html")
        result_path = save_report(sample_result, path)
        assert result_path.exists()
