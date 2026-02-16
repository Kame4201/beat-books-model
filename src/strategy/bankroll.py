"""
Bankroll management rules and position sizing.

Implements safeguards to protect bankroll from ruin:
- Maximum bet limits
- Stop-loss rules
- Daily exposure limits
"""
from datetime import date, datetime
from typing import Optional


class BankrollManager:
    """
    Manages bankroll and enforces betting limits.

    This class tracks bankroll changes over time and enforces risk management
    rules to prevent over-betting and catastrophic losses.
    """

    def __init__(
        self,
        starting_bankroll: float,
        max_bet_pct: float = 0.05,
        min_bet_pct: float = 0.01,
        stop_loss_pct: float = 0.50,
        daily_exposure_pct: float = 0.15,
    ):
        """
        Initialize bankroll manager.

        Args:
            starting_bankroll: Initial bankroll amount ($)
            max_bet_pct: Maximum bet as fraction of current bankroll (default 5%)
            min_bet_pct: Minimum bet as fraction of current bankroll (default 1%)
            stop_loss_pct: Stop trading if bankroll drops below this fraction
                          of starting amount (default 50%)
            daily_exposure_pct: Maximum daily exposure as fraction of bankroll
                               (default 15%)
        """
        if starting_bankroll <= 0:
            raise ValueError("starting_bankroll must be positive")
        if not 0.0 < max_bet_pct <= 1.0:
            raise ValueError("max_bet_pct must be between 0 and 1")
        if not 0.0 < min_bet_pct <= max_bet_pct:
            raise ValueError("min_bet_pct must be between 0 and max_bet_pct")
        if not 0.0 < stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must be between 0 and 1")
        if not 0.0 < daily_exposure_pct <= 1.0:
            raise ValueError("daily_exposure_pct must be between 0 and 1")

        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.stop_loss_pct = stop_loss_pct
        self.daily_exposure_pct = daily_exposure_pct

        # Track daily exposure
        self._daily_exposure: dict[date, float] = {}  # date -> total at-risk amount
        self._is_stopped = False

    @property
    def is_stopped(self) -> bool:
        """Check if trading is stopped due to stop-loss."""
        return self._is_stopped

    @property
    def stop_loss_threshold(self) -> float:
        """Get the dollar amount that triggers stop-loss."""
        return self.starting_bankroll * self.stop_loss_pct

    def check_stop_loss(self) -> bool:
        """
        Check if stop-loss has been triggered.

        Returns:
            True if bankroll is below stop-loss threshold

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.5)
            >>> bm.current_bankroll = 4500
            >>> bm.check_stop_loss()
            True
        """
        if self.current_bankroll < self.stop_loss_threshold:
            self._is_stopped = True
            return True
        return False

    def get_max_bet_size(self, kelly_size: float) -> float:
        """
        Get maximum bet size in dollars, applying all constraints.

        Applies:
        1. Kelly size (as fraction of bankroll)
        2. Maximum bet percentage cap
        3. Minimum bet percentage floor

        Args:
            kelly_size: Kelly-recommended bet size (0-1 fraction)

        Returns:
            Bet size in dollars, or 0 if below minimum or stopped

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> bm.get_max_bet_size(0.03)  # 3% Kelly
            300.0
            >>> bm.get_max_bet_size(0.10)  # 10% Kelly, capped at 5%
            500.0
        """
        # Stop-loss check
        if self.check_stop_loss():
            return 0.0

        # Apply maximum cap
        capped_size = min(kelly_size, self.max_bet_pct)

        # Check minimum threshold
        if 0 < capped_size < self.min_bet_pct:
            return 0.0

        # Convert to dollars
        bet_dollars = capped_size * self.current_bankroll
        return bet_dollars

    def get_daily_exposure(self, day: Optional[date] = None) -> float:
        """
        Get total exposure (at-risk amount) for a given day.

        Args:
            day: Date to check (defaults to today)

        Returns:
            Total amount at risk on that day

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> from datetime import date
            >>> today = date(2024, 1, 15)
            >>> bm.get_daily_exposure(today)
            0.0
        """
        if day is None:
            day = date.today()
        return self._daily_exposure.get(day, 0.0)

    def can_add_bet(
        self, bet_amount: float, bet_date: Optional[date] = None
    ) -> tuple[bool, str]:
        """
        Check if a bet can be added without violating daily exposure limit.

        Args:
            bet_amount: Proposed bet size in dollars
            bet_date: Date of the bet (defaults to today)

        Returns:
            Tuple of (can_bet: bool, reason: str)

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000, daily_exposure_pct=0.15)
            >>> bm.can_add_bet(1000)
            (True, 'OK')
            >>> bm.add_bet(1000)
            True
            >>> bm.can_add_bet(600)
            (False, 'Daily exposure limit exceeded: $1600.00 > $1500.00')
        """
        if bet_date is None:
            bet_date = date.today()

        # Stop-loss check
        if self.check_stop_loss():
            return False, f"Stop-loss triggered: bankroll ${self.current_bankroll:.2f} < ${self.stop_loss_threshold:.2f}"

        current_exposure = self.get_daily_exposure(bet_date)
        max_daily_exposure = self.daily_exposure_pct * self.current_bankroll
        new_exposure = current_exposure + bet_amount

        if new_exposure > max_daily_exposure:
            return (
                False,
                f"Daily exposure limit exceeded: ${new_exposure:.2f} > ${max_daily_exposure:.2f}",
            )

        return True, "OK"

    def add_bet(self, bet_amount: float, bet_date: Optional[date] = None) -> bool:
        """
        Add a bet to the daily exposure tracker.

        Args:
            bet_amount: Bet size in dollars
            bet_date: Date of the bet (defaults to today)

        Returns:
            True if bet was added, False if it violates limits

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> bm.add_bet(500)
            True
        """
        can_bet, reason = self.can_add_bet(bet_amount, bet_date)
        if not can_bet:
            return False

        if bet_date is None:
            bet_date = date.today()

        self._daily_exposure[bet_date] = self.get_daily_exposure(bet_date) + bet_amount
        return True

    def settle_bet(self, profit: float) -> None:
        """
        Update bankroll after a bet settles.

        Args:
            profit: Net profit/loss from the bet (positive for win, negative for loss)

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> bm.settle_bet(100)  # Won $100
            >>> bm.current_bankroll
            10100.0
            >>> bm.settle_bet(-200)  # Lost $200
            >>> bm.current_bankroll
            9900.0
        """
        self.current_bankroll += profit
        # Ensure bankroll doesn't go negative
        self.current_bankroll = max(0.0, self.current_bankroll)
        # Check stop-loss after settlement
        self.check_stop_loss()

    def get_stats(self) -> dict:
        """
        Get current bankroll statistics.

        Returns:
            Dictionary with bankroll stats

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> bm.current_bankroll = 11500
            >>> stats = bm.get_stats()
            >>> stats['total_return_pct']
            0.15
        """
        return {
            "starting_bankroll": self.starting_bankroll,
            "current_bankroll": self.current_bankroll,
            "total_return": self.current_bankroll - self.starting_bankroll,
            "total_return_pct": (self.current_bankroll - self.starting_bankroll)
            / self.starting_bankroll,
            "stop_loss_threshold": self.stop_loss_threshold,
            "is_stopped": self.is_stopped,
            "max_bet_dollars": self.current_bankroll * self.max_bet_pct,
            "min_bet_dollars": self.current_bankroll * self.min_bet_pct,
        }

    def reset(self) -> None:
        """
        Reset bankroll to starting amount and clear all state.

        Useful for backtesting multiple strategies.

        Example:
            >>> bm = BankrollManager(starting_bankroll=10000)
            >>> bm.current_bankroll = 8000
            >>> bm.reset()
            >>> bm.current_bankroll
            10000.0
        """
        self.current_bankroll = self.starting_bankroll
        self._daily_exposure = {}
        self._is_stopped = False
