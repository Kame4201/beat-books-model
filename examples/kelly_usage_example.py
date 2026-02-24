"""
Example usage of Kelly Criterion bet sizing.

This demonstrates how to use the strategy module for optimal bet sizing
with bankroll management.
"""
from src.strategy import KellyCriterion, BankrollManager


def main():
    """Demonstrate Kelly Criterion bet sizing."""

    # Initialize Kelly calculator (half Kelly, 2% min edge, 5% max bet)
    kelly = KellyCriterion(
        kelly_fraction=0.5,  # Half Kelly (recommended)
        min_edge=0.02,  # Only bet with 2%+ edge
        max_bet_pct=0.05,  # Never bet more than 5% of bankroll
        min_bet_pct=0.01,  # Don't bother with bets < 1%
    )

    # Initialize bankroll manager
    bankroll = BankrollManager(
        starting_bankroll=10000,
        max_bet_pct=0.05,
        min_bet_pct=0.01,
        stop_loss_pct=0.50,  # Stop if bankroll drops below 50%
        daily_exposure_pct=0.15,  # Max 15% at risk per day
    )

    # Example 1: Strong favorite with edge
    print("=" * 60)
    print("Example 1: Strong edge on favorite")
    print("=" * 60)
    model_prob = 0.60  # Model predicts 60% win probability
    market_odds = -110  # Typical NFL odds

    result = kelly.calculate_bet_size(model_prob, market_odds)
    print(f"Model probability: {result['confidence']:.1%}")
    print(f"Edge vs market: {result['edge_vs_market']:.2%}")
    print(f"Kelly fraction used: {result['kelly_fraction']}")
    print(f"Raw Kelly: {result['raw_kelly']:.2%}")
    print(f"Recommended bet size: {result['recommended_bet_size']:.2%}")
    print(f"Bet recommendation: {result['bet_recommendation']}")

    # Calculate dollar amount
    bet_amount = bankroll.get_max_bet_size(result["recommended_bet_size"])
    print(f"Bet amount: ${bet_amount:.2f}")
    print()

    # Example 2: Underdog with edge
    print("=" * 60)
    print("Example 2: Underdog with strong edge")
    print("=" * 60)
    model_prob = 0.45  # Model gives underdog 45% chance
    market_odds = +200  # Market implies 33.3%

    result = kelly.calculate_bet_size(model_prob, market_odds)
    print(f"Model probability: {result['confidence']:.1%}")
    print(f"Edge vs market: {result['edge_vs_market']:.2%}")
    print(f"Recommended bet size: {result['recommended_bet_size']:.2%}")
    print(f"Bet recommendation: {result['bet_recommendation']}")

    bet_amount = bankroll.get_max_bet_size(result["recommended_bet_size"])
    print(f"Bet amount: ${bet_amount:.2f}")
    print()

    # Example 3: No edge - should pass
    print("=" * 60)
    print("Example 3: No edge (should PASS)")
    print("=" * 60)
    model_prob = 0.50  # Model predicts 50%
    market_odds = -110  # Market implies 52.4%

    result = kelly.calculate_bet_size(model_prob, market_odds)
    print(f"Model probability: {result['confidence']:.1%}")
    print(f"Edge vs market: {result['edge_vs_market']:.2%}")
    print(f"Recommended bet size: {result['recommended_bet_size']:.2%}")
    print(f"Bet recommendation: {result['bet_recommendation']}")
    print()

    # Example 4: Bankroll management
    print("=" * 60)
    print("Example 4: Bankroll management with daily limits")
    print("=" * 60)

    from datetime import date

    today = date.today()

    # Try to place three bets
    bets = [500, 600, 500]  # Total $1600, but limit is $1500

    for i, bet_amt in enumerate(bets, 1):
        can_bet, reason = bankroll.can_add_bet(bet_amt, today)
        print(f"Bet {i} (${bet_amt}): {reason}")
        if can_bet:
            bankroll.add_bet(bet_amt, today)

    print(f"Daily exposure: ${bankroll.get_daily_exposure(today):.2f}")
    print()

    # Example 5: Stop-loss
    print("=" * 60)
    print("Example 5: Stop-loss protection")
    print("=" * 60)
    bankroll2 = BankrollManager(starting_bankroll=10000, stop_loss_pct=0.50)
    print(f"Starting bankroll: ${bankroll2.current_bankroll:.2f}")

    # Simulate losses
    bankroll2.settle_bet(-5500)  # Big loss
    print(f"After loss: ${bankroll2.current_bankroll:.2f}")
    print(f"Stop-loss triggered: {bankroll2.is_stopped}")
    print(f"Can still bet: ${bankroll2.get_max_bet_size(0.03):.2f}")

    # Get stats
    print("\nBankroll stats:")
    stats = bankroll2.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}" if "dollars" in key or "bankroll" in key or "threshold" in key else f"  {key}: {value:.2%}" if "pct" in key else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
