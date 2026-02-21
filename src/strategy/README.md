# Strategy Module

Optimal bet sizing using Kelly Criterion with comprehensive bankroll management.

## Overview

This module implements:
- **Kelly Criterion** bet sizing with fractional variants (full, half, quarter Kelly)
- **Edge calculation** against market odds
- **Bankroll management** with safety constraints
- **Position sizing** with daily exposure limits
- **Stop-loss protection** to prevent catastrophic losses

## Quick Start

```python
from src.strategy import KellyCriterion, BankrollManager

# Initialize Kelly calculator (half Kelly recommended)
kelly = KellyCriterion(kelly_fraction=0.5, min_edge=0.02)

# Calculate bet size
result = kelly.calculate_bet_size(model_prob=0.60, american_odds=-110)
print(result['bet_recommendation'])  # "STRONG BET"
print(result['recommended_bet_size'])  # 0.0794 (7.94% of bankroll)

# Initialize bankroll manager
bankroll = BankrollManager(starting_bankroll=10000)
bet_dollars = bankroll.get_max_bet_size(result['recommended_bet_size'])
print(f"Bet ${bet_dollars:.2f}")  # Bet $794.00
```

## Kelly Criterion

### Formula

```
f* = (bp - q) / b
```

Where:
- `f*` = fraction of bankroll to bet
- `b` = decimal odds - 1 (net odds received)
- `p` = probability of winning (model's prediction)
- `q` = probability of losing (1 - p)

### Fractional Kelly Variants

- **Full Kelly** (f = 1.0): Maximum growth rate, highest variance
- **Half Kelly** (f = 0.5): Recommended default, reduces variance significantly
- **Quarter Kelly** (f = 0.25): Conservative, smoother equity curve
- **Custom**: Any fraction between 0.0 and 1.0

### Edge Calculation

Edge = Model Probability - Market Implied Probability

Only bet when edge exceeds minimum threshold (default 2%).

## Bankroll Management

### Safety Constraints

1. **Maximum Bet**: 5% of current bankroll (hard cap)
2. **Minimum Bet**: 1% of bankroll (don't bother with tiny bets)
3. **Stop-Loss**: Halt trading if bankroll drops below 50% of starting amount
4. **Daily Exposure**: Max 15% of bankroll at risk on any single day

### Bet Recommendations

- **PASS**: No edge or edge below minimum threshold
- **BET**: Moderate edge (2-5%)
- **STRONG BET**: Strong edge (5%+ or Kelly size 3%+)

## API Reference

### KellyCriterion

```python
KellyCriterion(
    kelly_fraction: float = 0.5,
    min_edge: float = 0.02,
    max_bet_pct: float = 0.05,
    min_bet_pct: float = 0.01
)
```

**Methods:**

- `calculate_bet_size(model_prob, american_odds)` → dict
  - Returns: edge_vs_market, recommended_bet_size, kelly_fraction, confidence, bet_recommendation, raw_kelly

### BankrollManager

```python
BankrollManager(
    starting_bankroll: float,
    max_bet_pct: float = 0.05,
    min_bet_pct: float = 0.01,
    stop_loss_pct: float = 0.50,
    daily_exposure_pct: float = 0.15
)
```

**Methods:**

- `get_max_bet_size(kelly_size)` → float (bet size in dollars)
- `can_add_bet(bet_amount, bet_date)` → tuple[bool, str]
- `add_bet(bet_amount, bet_date)` → bool
- `settle_bet(profit)` → None
- `get_stats()` → dict
- `reset()` → None

**Properties:**

- `is_stopped` → bool
- `stop_loss_threshold` → float

## Helper Functions

```python
from src.strategy.kelly import (
    american_to_decimal,
    decimal_to_implied_probability,
    calculate_edge,
    calculate_kelly,
    get_bet_recommendation,
)
```

## Examples

See `examples/kelly_usage_example.py` for comprehensive usage examples.

## Testing

Run tests:

```bash
pytest tests/test_strategy/ -v
```

Tests include hand-verified expected values for:
- Odds conversion accuracy
- Kelly formula correctness
- Edge calculation
- Bankroll management rules
- All acceptance criteria

## Integration with Backtesting

```python
from src.strategy import KellyCriterion, BankrollManager

# In backtesting loop
kelly = KellyCriterion(kelly_fraction=0.5)
bankroll = BankrollManager(starting_bankroll=10000)

for game in games:
    model_prob = model.predict(game)
    market_odds = game.closing_line

    result = kelly.calculate_bet_size(model_prob, market_odds)

    if result['bet_recommendation'] != 'PASS':
        bet_size = bankroll.get_max_bet_size(result['recommended_bet_size'])

        if bankroll.add_bet(bet_size):
            # Place bet
            profit = calculate_profit(bet_size, market_odds, game.outcome)
            bankroll.settle_bet(profit)
```

## References

- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Fractional Kelly](https://www.investopedia.com/articles/trading/04/091504.asp)
- [Bet sizing in sports betting](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/the-importance-of-bet-sizing/NSBJN3QLWJB4BPZZ)
