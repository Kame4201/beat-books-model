"""
Kelly Criterion bet sizing calculator.

Implements optimal bet sizing using Kelly Criterion formula:
f* = (bp - q) / b

where:
- f* = fraction of bankroll to bet
- b = decimal odds - 1 (net odds received)
- p = probability of winning (model's predicted probability)
- q = probability of losing (1 - p)
"""

from enum import Enum


class KellyFraction(Enum):
    """Kelly Criterion fraction variants."""

    FULL = 1.0  # Full Kelly (highest variance)
    HALF = 0.5  # Half Kelly (recommended default)
    QUARTER = 0.25  # Quarter Kelly (conservative)


def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        american_odds: American odds format (e.g., -110, +150)
            Negative odds (e.g., -110) mean you bet that amount to win $100
            Positive odds (e.g., +150) mean you win that amount on a $100 bet

    Returns:
        Decimal odds (e.g., 1.909, 2.5)

    Examples:
        >>> american_to_decimal(-110)
        1.9090909090909092
        >>> american_to_decimal(+150)
        2.5
        >>> american_to_decimal(+100)
        2.0
    """
    if american_odds < 0:
        # Favorite: decimal = (100 / abs(american_odds)) + 1
        return (100 / abs(american_odds)) + 1
    else:
        # Underdog: decimal = (american_odds / 100) + 1
        return (american_odds / 100) + 1


def decimal_to_implied_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds (e.g., 1.909, 2.5)

    Returns:
        Implied probability (0-1 scale)

    Examples:
        >>> decimal_to_implied_probability(2.0)
        0.5
        >>> round(decimal_to_implied_probability(1.909), 3)
        0.524
    """
    return 1 / decimal_odds


def calculate_edge(model_prob: float, american_odds: int) -> float:
    """
    Calculate edge over the market.

    Edge = model_probability - implied_probability

    Args:
        model_prob: Model's predicted probability (0-1 scale)
        american_odds: Market odds in American format

    Returns:
        Edge as a decimal (e.g., 0.05 = 5% edge)

    Examples:
        >>> # Model predicts 55% chance, market implies 52.4%
        >>> round(calculate_edge(0.55, -110), 3)
        0.026
    """
    decimal_odds = american_to_decimal(american_odds)
    implied_prob = decimal_to_implied_probability(decimal_odds)
    return model_prob - implied_prob


def calculate_kelly(
    model_prob: float,
    american_odds: int,
    kelly_fraction: float = 0.5,
    min_edge: float = 0.02,
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Kelly formula: f* = (bp - q) / b
    where:
    - b = decimal odds - 1 (net odds)
    - p = probability of winning
    - q = probability of losing (1 - p)

    Args:
        model_prob: Model's predicted win probability (0-1 scale)
        american_odds: Market odds in American format
        kelly_fraction: Fraction of Kelly to use (default 0.5 for half Kelly)
        min_edge: Minimum edge required to bet (default 0.02 = 2%)

    Returns:
        Fraction of bankroll to bet (0-1 scale)
        Returns 0 if edge is negative or below minimum threshold

    Examples:
        >>> # 60% win prob, -110 odds, half Kelly
        >>> round(calculate_kelly(0.60, -110, kelly_fraction=0.5), 4)
        0.0764
        >>> # No edge, should return 0
        >>> calculate_kelly(0.50, -110)
        0.0
    """
    # Check for edge
    edge = calculate_edge(model_prob, american_odds)
    if edge < min_edge:
        return 0.0

    # Convert to decimal odds and calculate net odds (b)
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1  # Net odds received on winning bet

    # Kelly formula: f* = (bp - q) / b
    p = model_prob
    q = 1 - p
    kelly_full = (b * p - q) / b

    # Apply Kelly fraction
    kelly_bet = max(0.0, kelly_full * kelly_fraction)

    return kelly_bet


def get_bet_recommendation(
    edge: float,
    kelly_size: float,
    min_edge: float = 0.02,
    strong_edge_threshold: float = 0.05,
    strong_kelly_threshold: float = 0.03,
) -> str:
    """
    Get bet recommendation based on edge and Kelly size.

    Args:
        edge: Edge over market (0-1 scale)
        kelly_size: Recommended Kelly bet size (0-1 scale)
        min_edge: Minimum edge to recommend betting
        strong_edge_threshold: Edge threshold for strong bet (default 5%)
        strong_kelly_threshold: Kelly size threshold for strong bet (default 3%)

    Returns:
        "PASS", "BET", or "STRONG BET"

    Examples:
        >>> get_bet_recommendation(0.01, 0.005)
        'PASS'
        >>> get_bet_recommendation(0.03, 0.02)
        'BET'
        >>> get_bet_recommendation(0.06, 0.04)
        'STRONG BET'
    """
    if edge < min_edge or kelly_size == 0:
        return "PASS"
    elif edge >= strong_edge_threshold or kelly_size >= strong_kelly_threshold:
        return "STRONG BET"
    else:
        return "BET"


class KellyCriterion:
    """
    Kelly Criterion calculator with configurable parameters.

    This class provides a convenient interface for calculating bet sizes
    and includes all the betting logic in one place.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_edge: float = 0.02,
        max_bet_pct: float = 0.05,
        min_bet_pct: float = 0.01,
    ):
        """
        Initialize Kelly Criterion calculator.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.0-1.0)
                0.25 = Quarter Kelly (conservative)
                0.5 = Half Kelly (recommended default)
                1.0 = Full Kelly (aggressive)
            min_edge: Minimum edge required to bet (default 2%)
            max_bet_pct: Maximum bet as fraction of bankroll (default 5%)
            min_bet_pct: Minimum bet as fraction of bankroll (default 1%)
        """
        if not 0.0 <= kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be between 0.0 and 1.0")
        if not 0.0 <= min_edge <= 1.0:
            raise ValueError("min_edge must be between 0.0 and 1.0")
        if not 0.0 < max_bet_pct <= 1.0:
            raise ValueError("max_bet_pct must be between 0.0 and 1.0")
        if not 0.0 < min_bet_pct <= max_bet_pct:
            raise ValueError("min_bet_pct must be between 0.0 and max_bet_pct")

        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct

    def calculate_bet_size(
        self,
        model_prob: float,
        american_odds: int,
    ) -> dict:
        """
        Calculate recommended bet size and return detailed prediction output.

        Args:
            model_prob: Model's predicted win probability (0-1 scale)
            american_odds: Market odds in American format

        Returns:
            Dictionary with:
            - edge_vs_market: Model probability - implied probability
            - recommended_bet_size: Kelly-optimal fraction (capped at max_bet_pct)
            - kelly_fraction: Fraction variant used
            - confidence: Model's raw probability
            - bet_recommendation: "BET" / "PASS" / "STRONG BET"
            - raw_kelly: Uncapped Kelly size (for analysis)

        Example:
            >>> kc = KellyCriterion(kelly_fraction=0.5, min_edge=0.02)
            >>> result = kc.calculate_bet_size(0.60, -110)
            >>> result['bet_recommendation']
            'STRONG BET'
        """
        # Calculate edge
        edge = calculate_edge(model_prob, american_odds)

        # Calculate raw Kelly size
        raw_kelly = calculate_kelly(
            model_prob=model_prob,
            american_odds=american_odds,
            kelly_fraction=self.kelly_fraction,
            min_edge=self.min_edge,
        )

        # Apply hard cap
        recommended_size = min(raw_kelly, self.max_bet_pct)

        # If below minimum bet size, round to zero (don't bother)
        if 0 < recommended_size < self.min_bet_pct:
            recommended_size = 0.0

        # Get recommendation
        bet_rec = get_bet_recommendation(
            edge=edge,
            kelly_size=recommended_size,
            min_edge=self.min_edge,
        )

        return {
            "edge_vs_market": edge,
            "recommended_bet_size": recommended_size,
            "kelly_fraction": self.kelly_fraction,
            "confidence": model_prob,
            "bet_recommendation": bet_rec,
            "raw_kelly": raw_kelly,  # For analysis/debugging
        }
