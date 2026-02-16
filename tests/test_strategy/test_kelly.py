"""
Unit tests for Kelly Criterion calculator.

Tests include hand-verified expected values for correctness.
"""
import pytest
from src.strategy.kelly import (
    KellyCriterion,
    KellyFraction,
    american_to_decimal,
    calculate_edge,
    calculate_kelly,
    decimal_to_implied_probability,
    get_bet_recommendation,
)


class TestOddsConversion:
    """Test American odds to decimal odds conversion."""

    def test_favorite_odds(self):
        """Test conversion of favorite odds (negative)."""
        # -110 odds: bet $110 to win $100 (win $210 total)
        # Decimal: 210/110 = 1.909...
        assert round(american_to_decimal(-110), 4) == 1.9091

        # -200 odds: bet $200 to win $100 (win $300 total)
        # Decimal: 300/200 = 1.5
        assert american_to_decimal(-200) == 1.5

        # -150 odds
        assert round(american_to_decimal(-150), 4) == 1.6667

    def test_underdog_odds(self):
        """Test conversion of underdog odds (positive)."""
        # +150 odds: bet $100 to win $150 (win $250 total)
        # Decimal: 250/100 = 2.5
        assert american_to_decimal(+150) == 2.5

        # +200 odds: bet $100 to win $200 (win $300 total)
        # Decimal: 300/100 = 3.0
        assert american_to_decimal(+200) == 3.0

        # +100 odds (even money)
        assert american_to_decimal(+100) == 2.0

    def test_implied_probability(self):
        """Test decimal odds to implied probability."""
        # Even odds (2.0) = 50%
        assert decimal_to_implied_probability(2.0) == 0.5

        # -110 odds (1.909) = 52.4%
        assert round(decimal_to_implied_probability(1.9091), 3) == 0.524

        # +150 odds (2.5) = 40%
        assert decimal_to_implied_probability(2.5) == 0.4


class TestEdgeCalculation:
    """Test edge calculation."""

    def test_positive_edge(self):
        """Test when model has positive edge over market."""
        # Model: 55% win probability
        # Market: -110 odds = 52.4% implied
        # Edge: 55% - 52.4% = 2.6%
        edge = calculate_edge(0.55, -110)
        assert round(edge, 3) == 0.026

    def test_negative_edge(self):
        """Test when model has negative edge (worse than market)."""
        # Model: 50% win probability
        # Market: -110 odds = 52.4% implied
        # Edge: 50% - 52.4% = -2.4%
        edge = calculate_edge(0.50, -110)
        assert round(edge, 3) == -0.024

    def test_zero_edge(self):
        """Test when model matches market exactly."""
        # Model: 52.38% win probability
        # Market: -110 odds = 52.38% implied
        # Edge: ~0%
        market_implied = decimal_to_implied_probability(american_to_decimal(-110))
        edge = calculate_edge(market_implied, -110)
        assert abs(edge) < 0.001  # Very close to zero


class TestKellyFormula:
    """Test Kelly Criterion formula."""

    def test_kelly_with_edge(self):
        """
        Test Kelly calculation with positive edge.

        Hand calculation:
        - Model prob (p): 60%
        - Market odds: -110 (decimal 1.909, b = 0.909)
        - q = 1 - 0.6 = 0.4
        - Full Kelly: (0.909 * 0.6 - 0.4) / 0.909 = 0.1587
        - Half Kelly: 0.1587 / 2 = 0.0794
        """
        kelly = calculate_kelly(
            model_prob=0.60, american_odds=-110, kelly_fraction=0.5, min_edge=0.01
        )
        assert round(kelly, 4) == 0.0794

    def test_kelly_full_variant(self):
        """Test full Kelly (no fraction)."""
        # Same as above but full Kelly
        kelly = calculate_kelly(
            model_prob=0.60, american_odds=-110, kelly_fraction=1.0, min_edge=0.01
        )
        assert round(kelly, 4) == 0.1587

    def test_kelly_quarter_variant(self):
        """Test quarter Kelly (conservative)."""
        kelly = calculate_kelly(
            model_prob=0.60, american_odds=-110, kelly_fraction=0.25, min_edge=0.01
        )
        assert round(kelly, 4) == 0.0397

    def test_kelly_no_edge(self):
        """Test that Kelly returns 0 when edge is below minimum."""
        # Model: 52% (very small edge over -110 odds)
        kelly = calculate_kelly(
            model_prob=0.52, american_odds=-110, kelly_fraction=0.5, min_edge=0.02
        )
        assert kelly == 0.0

    def test_kelly_negative_edge(self):
        """Test that Kelly returns 0 when edge is negative."""
        # Model: 50% (negative edge vs -110 odds)
        kelly = calculate_kelly(
            model_prob=0.50, american_odds=-110, kelly_fraction=0.5, min_edge=0.01
        )
        assert kelly == 0.0

    def test_kelly_underdog_with_edge(self):
        """
        Test Kelly for underdog bet.

        Hand calculation:
        - Model prob (p): 45%
        - Market odds: +200 (decimal 3.0, b = 2.0)
        - q = 0.55
        - Market implied: 33.33%
        - Edge: 45% - 33.33% = 11.67% (strong edge!)
        - Full Kelly: (2.0 * 0.45 - 0.55) / 2.0 = 0.175
        - Half Kelly: 0.0875
        """
        kelly = calculate_kelly(
            model_prob=0.45, american_odds=+200, kelly_fraction=0.5, min_edge=0.02
        )
        assert round(kelly, 4) == 0.0875


class TestBetRecommendation:
    """Test bet recommendation logic."""

    def test_pass_recommendation(self):
        """Test PASS when edge is too small."""
        rec = get_bet_recommendation(edge=0.01, kelly_size=0.005, min_edge=0.02)
        assert rec == "PASS"

    def test_pass_zero_kelly(self):
        """Test PASS when Kelly size is zero."""
        rec = get_bet_recommendation(edge=0.03, kelly_size=0.0, min_edge=0.02)
        assert rec == "PASS"

    def test_bet_recommendation(self):
        """Test BET for moderate edge."""
        rec = get_bet_recommendation(edge=0.03, kelly_size=0.02, min_edge=0.02)
        assert rec == "BET"

    def test_strong_bet_high_edge(self):
        """Test STRONG BET for high edge."""
        rec = get_bet_recommendation(
            edge=0.06, kelly_size=0.02, min_edge=0.02, strong_edge_threshold=0.05
        )
        assert rec == "STRONG BET"

    def test_strong_bet_high_kelly(self):
        """Test STRONG BET for high Kelly size."""
        rec = get_bet_recommendation(
            edge=0.03,
            kelly_size=0.04,
            min_edge=0.02,
            strong_kelly_threshold=0.03,
        )
        assert rec == "STRONG BET"


class TestKellyCriterionClass:
    """Test KellyCriterion class."""

    def test_initialization(self):
        """Test KellyCriterion initialization."""
        kc = KellyCriterion(
            kelly_fraction=0.5, min_edge=0.02, max_bet_pct=0.05, min_bet_pct=0.01
        )
        assert kc.kelly_fraction == 0.5
        assert kc.min_edge == 0.02
        assert kc.max_bet_pct == 0.05
        assert kc.min_bet_pct == 0.01

    def test_invalid_kelly_fraction(self):
        """Test that invalid Kelly fraction raises error."""
        with pytest.raises(ValueError, match="kelly_fraction must be between"):
            KellyCriterion(kelly_fraction=1.5)

        with pytest.raises(ValueError, match="kelly_fraction must be between"):
            KellyCriterion(kelly_fraction=-0.1)

    def test_invalid_edge(self):
        """Test that invalid min_edge raises error."""
        with pytest.raises(ValueError, match="min_edge must be between"):
            KellyCriterion(min_edge=1.5)

    def test_invalid_max_bet(self):
        """Test that invalid max_bet_pct raises error."""
        with pytest.raises(ValueError, match="max_bet_pct must be between"):
            KellyCriterion(max_bet_pct=0.0)

        with pytest.raises(ValueError, match="max_bet_pct must be between"):
            KellyCriterion(max_bet_pct=1.5)

    def test_invalid_min_bet(self):
        """Test that invalid min_bet_pct raises error."""
        with pytest.raises(ValueError, match="min_bet_pct must be between"):
            KellyCriterion(min_bet_pct=0.0)

        # min_bet > max_bet
        with pytest.raises(ValueError, match="min_bet_pct must be between"):
            KellyCriterion(min_bet_pct=0.1, max_bet_pct=0.05)

    def test_calculate_bet_size_output(self):
        """Test that calculate_bet_size returns correct structure."""
        kc = KellyCriterion(kelly_fraction=0.5, min_edge=0.02, max_bet_pct=0.05)
        result = kc.calculate_bet_size(model_prob=0.60, american_odds=-110)

        # Check all required keys
        assert "edge_vs_market" in result
        assert "recommended_bet_size" in result
        assert "kelly_fraction" in result
        assert "confidence" in result
        assert "bet_recommendation" in result
        assert "raw_kelly" in result

        # Check values
        assert result["kelly_fraction"] == 0.5
        assert result["confidence"] == 0.60
        assert result["bet_recommendation"] in ["BET", "PASS", "STRONG BET"]

    def test_max_bet_cap_applied(self):
        """
        Test that max bet cap overrides Kelly when Kelly suggests more.

        Hand calculation:
        - Model: 70% (very strong edge)
        - Market: -110 (52.4% implied)
        - Edge: 17.6%
        - Half Kelly would be ~9.7%
        - But max_bet_pct=5% should cap it
        """
        kc = KellyCriterion(kelly_fraction=0.5, min_edge=0.02, max_bet_pct=0.05)
        result = kc.calculate_bet_size(model_prob=0.70, american_odds=-110)

        assert result["recommended_bet_size"] == 0.05  # Capped at max
        assert result["raw_kelly"] > 0.05  # Raw Kelly is higher

    def test_min_bet_threshold(self):
        """Test that bets below minimum threshold are rounded to zero."""
        kc = KellyCriterion(
            kelly_fraction=0.5, min_edge=0.01, max_bet_pct=0.05, min_bet_pct=0.01
        )
        # Small edge scenario
        result = kc.calculate_bet_size(model_prob=0.53, american_odds=-110)

        # Kelly might suggest 0.5% but min is 1%, so should be 0
        if result["raw_kelly"] < 0.01:
            assert result["recommended_bet_size"] == 0.0

    def test_pass_with_no_edge(self):
        """Test PASS recommendation when no edge."""
        kc = KellyCriterion(kelly_fraction=0.5, min_edge=0.02, max_bet_pct=0.05)
        result = kc.calculate_bet_size(model_prob=0.50, american_odds=-110)

        assert result["bet_recommendation"] == "PASS"
        assert result["recommended_bet_size"] == 0.0

    def test_strong_bet_recommendation(self):
        """Test STRONG BET for high edge scenario."""
        kc = KellyCriterion(kelly_fraction=0.5, min_edge=0.02, max_bet_pct=0.05)
        result = kc.calculate_bet_size(model_prob=0.60, american_odds=-110)

        # 60% vs 52.4% is strong edge, should get STRONG BET
        assert result["bet_recommendation"] == "STRONG BET"
        assert result["recommended_bet_size"] > 0


class TestKellyFractionEnum:
    """Test KellyFraction enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert KellyFraction.FULL.value == 1.0
        assert KellyFraction.HALF.value == 0.5
        assert KellyFraction.QUARTER.value == 0.25

    def test_using_enum_in_calculator(self):
        """Test using enum values in KellyCriterion."""
        kc = KellyCriterion(kelly_fraction=KellyFraction.HALF.value)
        assert kc.kelly_fraction == 0.5

        kc2 = KellyCriterion(kelly_fraction=KellyFraction.QUARTER.value)
        assert kc2.kelly_fraction == 0.25


class TestAcceptanceCriteria:
    """Test acceptance criteria from requirements."""

    def test_kelly_formula_correctness(self):
        """
        Verify Kelly formula produces correct values for known test case.

        Known case:
        - Model: 60% win probability
        - Odds: -110 (decimal 1.909)
        - Expected full Kelly: (0.909 * 0.6 - 0.4) / 0.909 = 15.87%
        - Expected half Kelly: 7.94%
        """
        kelly_full = calculate_kelly(0.60, -110, kelly_fraction=1.0, min_edge=0.0)
        kelly_half = calculate_kelly(0.60, -110, kelly_fraction=0.5, min_edge=0.0)

        assert round(kelly_full, 4) == 0.1587
        assert round(kelly_half, 4) == 0.0794

    def test_negative_edge_returns_zero(self):
        """Verify negative edge always returns bet size of 0."""
        kc = KellyCriterion()
        result = kc.calculate_bet_size(model_prob=0.45, american_odds=-110)
        assert result["recommended_bet_size"] == 0.0

    def test_hard_cap_overrides_kelly(self):
        """Verify 5% hard cap overrides Kelly when Kelly suggests more."""
        kc = KellyCriterion(max_bet_pct=0.05)
        # Very high edge scenario
        result = kc.calculate_bet_size(model_prob=0.75, american_odds=-110)
        assert result["recommended_bet_size"] <= 0.05

    def test_fractional_kelly_variants(self):
        """Verify fractional Kelly variants work correctly."""
        test_cases = [
            (1.0, 0.1587),  # Full Kelly
            (0.5, 0.0794),  # Half Kelly
            (0.25, 0.0397),  # Quarter Kelly
        ]

        for fraction, expected in test_cases:
            kelly = calculate_kelly(0.60, -110, kelly_fraction=fraction, min_edge=0.0)
            assert round(kelly, 4) == expected
