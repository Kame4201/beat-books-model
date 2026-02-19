"""
Bankroll simulation and bet sizing strategies.
"""

from typing import Optional, Tuple
from src.backtesting.types import BacktestConfig, BetSizingMethod


def calculate_bet_size(
    config: BacktestConfig,
    bankroll: float,
    model_prob: float,
    market_prob: Optional[float] = None,
    market_odds: Optional[float] = None,
) -> float:
    """
    Calculate bet size based on configured strategy.

    Args:
        config: Backtest configuration
        bankroll: Current bankroll
        model_prob: Model's predicted probability (0-1)
        market_prob: Market implied probability (0-1), if available
        market_odds: American odds, if available

    Returns:
        Bet size in dollars
    """
    if config.bet_sizing_method == BetSizingMethod.FLAT:
        bet_size = config.flat_bet_size or 100.0

    elif config.bet_sizing_method == BetSizingMethod.PERCENTAGE:
        pct = config.percentage_bet_size or 0.02
        bet_size = bankroll * pct

    elif config.bet_sizing_method == BetSizingMethod.KELLY:
        if market_prob is None or market_odds is None:
            # Fallback to percentage if market data missing
            bet_size = bankroll * 0.02
        else:
            kelly_fraction = calculate_kelly_criterion(
                model_prob=model_prob,
                market_prob=market_prob,
                market_odds=market_odds,
            )
            # Apply fractional Kelly
            bet_size = bankroll * kelly_fraction * config.kelly_fraction

    else:
        raise ValueError(f"Unknown bet sizing method: {config.bet_sizing_method}")

    # Apply constraints
    bet_size = max(bet_size, config.min_bet_size)
    bet_size = min(bet_size, config.max_bet_size)
    bet_size = min(bet_size, bankroll * config.max_bet_fraction)

    return bet_size


def calculate_kelly_criterion(
    model_prob: float,
    market_prob: float,
    market_odds: float,
) -> float:
    """
    Calculate Kelly Criterion fraction.

    Kelly Criterion: f* = (bp - q) / b
    where:
        b = decimal odds - 1 (net odds)
        p = probability of winning
        q = probability of losing = 1 - p

    Args:
        model_prob: Our model's probability
        market_prob: Market implied probability
        market_odds: American odds (e.g., -110, +150)

    Returns:
        Kelly fraction (0-1), or 0 if no edge
    """
    # Convert American odds to decimal odds
    if market_odds > 0:
        decimal_odds = (market_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(market_odds)) + 1

    # Net odds (profit per dollar wagered)
    b = decimal_odds - 1

    # Our probabilities
    p = model_prob
    q = 1 - p

    # Kelly fraction
    kelly = (b * p - q) / b

    # Only bet if we have an edge (kelly > 0)
    return max(0.0, kelly)


def should_place_bet(
    config: BacktestConfig,
    model_prob: float,
    market_prob: Optional[float] = None,
) -> bool:
    """
    Determine if we should place a bet based on edge threshold.

    Args:
        config: Backtest configuration
        model_prob: Model's predicted probability
        market_prob: Market implied probability

    Returns:
        True if we should bet, False otherwise
    """
    if market_prob is None:
        return False

    # Calculate edge
    edge = abs(model_prob - market_prob)

    # Only bet if edge exceeds threshold
    return edge >= config.min_edge


def calculate_bet_payout(
    bet_amount: float,
    odds: float,
    won: bool,
) -> Tuple[float, float]:
    """
    Calculate payout and profit from a bet.

    Args:
        bet_amount: Amount wagered
        odds: American odds (e.g., -110, +150)
        won: Whether the bet won

    Returns:
        (payout, profit) where profit = payout - bet_amount
    """
    if not won:
        return 0.0, -bet_amount

    # Calculate profit from American odds
    if odds > 0:
        profit = bet_amount * (odds / 100)
    else:
        profit = bet_amount * (100 / abs(odds))

    payout = bet_amount + profit

    return payout, profit


def american_odds_to_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def probability_to_american_odds(prob: float) -> float:
    """
    Convert probability to American odds.

    Args:
        prob: Probability (0-1)

    Returns:
        American odds
    """
    if prob >= 0.5:
        # Favorite (negative odds)
        return -(prob / (1 - prob)) * 100
    else:
        # Underdog (positive odds)
        return ((1 - prob) / prob) * 100


class BankrollTracker:
    """
    Track bankroll changes over time during backtesting.
    """

    def __init__(self, starting_bankroll: float):
        """
        Initialize bankroll tracker.

        Args:
            starting_bankroll: Initial bankroll amount
        """
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.history = [(0, starting_bankroll)]  # (bet_number, bankroll)
        self.peak_bankroll = starting_bankroll

        # Summary statistics
        self.total_bets = 0
        self.total_wagered = 0.0
        self.total_profit = 0.0
        self.wins = 0
        self.losses = 0

    def place_bet(self, bet_amount: float, profit: float, won: bool):
        """
        Record a bet and update bankroll.

        Args:
            bet_amount: Amount wagered
            profit: Profit from bet (negative if loss)
            won: Whether bet won
        """
        self.total_bets += 1
        self.total_wagered += bet_amount
        self.total_profit += profit
        self.current_bankroll += profit

        if won:
            self.wins += 1
        else:
            self.losses += 1

        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

        # Record history
        self.history.append((self.total_bets, self.current_bankroll))

    def get_bankroll_curve(self) -> dict:
        """
        Get bankroll curve as dictionary.

        Returns:
            Dict mapping bet_number to bankroll
        """
        return {str(bet_num): bankroll for bet_num, bankroll in self.history}

    def get_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown.

        Returns:
            (max_drawdown_dollars, max_drawdown_percentage)
        """
        if not self.history:
            return 0.0, 0.0

        peak = self.history[0][1]
        max_dd = 0.0

        for _, bankroll in self.history:
            if bankroll > peak:
                peak = bankroll
            dd = peak - bankroll
            if dd > max_dd:
                max_dd = dd

        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def get_roi(self) -> float:
        """
        Calculate ROI percentage.

        Returns:
            ROI as percentage
        """
        if self.total_wagered == 0:
            return 0.0
        return (self.total_profit / self.total_wagered) * 100

    def get_win_rate(self) -> float:
        """
        Calculate win rate.

        Returns:
            Win rate as percentage
        """
        if self.total_bets == 0:
            return 0.0
        return (self.wins / self.total_bets) * 100

    def get_summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Dict with summary statistics
        """
        max_dd, max_dd_pct = self.get_max_drawdown()

        return {
            "starting_bankroll": self.starting_bankroll,
            "ending_bankroll": self.current_bankroll,
            "total_profit": self.total_profit,
            "total_bets": self.total_bets,
            "total_wagered": self.total_wagered,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.get_win_rate(),
            "roi": self.get_roi(),
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "peak_bankroll": self.peak_bankroll,
        }
