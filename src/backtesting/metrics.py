"""
Evaluation metrics for backtesting.

All metrics functions take predictions and return calculated values.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.backtesting.types import PredictionRecord


def calculate_accuracy(predictions: List[PredictionRecord]) -> float:
    """Calculate overall prediction accuracy."""
    if not predictions:
        return 0.0

    correct = sum(
        1
        for p in predictions
        if p.predicted_home_win_prob >= 0.5
        and p.actual_home_win
        or p.predicted_home_win_prob < 0.5
        and not p.actual_home_win
    )
    return correct / len(predictions)


def calculate_home_away_accuracy(
    predictions: List[PredictionRecord],
) -> Tuple[float, float]:
    """
    Calculate accuracy for home and away predictions separately.

    Returns:
        (home_accuracy, away_accuracy)
    """
    home_correct = 0
    home_total = 0
    away_correct = 0
    away_total = 0

    for p in predictions:
        predicted_home_win = p.predicted_home_win_prob >= 0.5

        if predicted_home_win:
            home_total += 1
            if p.actual_home_win:
                home_correct += 1
        else:
            away_total += 1
            if not p.actual_home_win:
                away_correct += 1

    home_acc = home_correct / home_total if home_total > 0 else 0.0
    away_acc = away_correct / away_total if away_total > 0 else 0.0

    return home_acc, away_acc


def calculate_favorite_underdog_accuracy(
    predictions: List[PredictionRecord],
) -> Tuple[float, float]:
    """
    Calculate accuracy for favorites vs underdogs.

    Favorite = team with spread advantage (negative spread for home).

    Returns:
        (favorite_accuracy, underdog_accuracy)
    """
    favorite_correct = 0
    favorite_total = 0
    underdog_correct = 0
    underdog_total = 0

    for p in predictions:
        if p.market_spread is None:
            continue

        # Negative spread means home is favorite
        home_is_favorite = p.market_spread < 0
        predicted_home_win = p.predicted_home_win_prob >= 0.5

        if home_is_favorite:
            if predicted_home_win:
                favorite_total += 1
                if p.actual_home_win:
                    favorite_correct += 1
            else:
                underdog_total += 1
                if not p.actual_home_win:
                    underdog_correct += 1
        else:
            if predicted_home_win:
                underdog_total += 1
                if p.actual_home_win:
                    underdog_correct += 1
            else:
                favorite_total += 1
                if not p.actual_home_win:
                    favorite_correct += 1

    fav_acc = favorite_correct / favorite_total if favorite_total > 0 else 0.0
    dog_acc = underdog_correct / underdog_total if underdog_total > 0 else 0.0

    return fav_acc, dog_acc


def calculate_log_loss(predictions: List[PredictionRecord]) -> float:
    """
    Calculate log loss (cross-entropy) for probability calibration.

    Lower is better. Perfect predictions = 0, worst = infinity.
    """
    if not predictions:
        return float("inf")

    log_losses = []
    for p in predictions:
        # Clip probabilities to avoid log(0)
        prob = np.clip(p.predicted_home_win_prob, 1e-15, 1 - 1e-15)
        actual = 1.0 if p.actual_home_win else 0.0

        loss = -(actual * np.log(prob) + (1 - actual) * np.log(1 - prob))
        log_losses.append(loss)

    return float(np.mean(log_losses))


def calculate_brier_score(predictions: List[PredictionRecord]) -> float:
    """
    Calculate Brier score for probability calibration.

    Lower is better. Perfect predictions = 0, worst = 1.
    """
    if not predictions:
        return 1.0

    scores = []
    for p in predictions:
        actual = 1.0 if p.actual_home_win else 0.0
        score = (p.predicted_home_win_prob - actual) ** 2
        scores.append(score)

    return float(np.mean(scores))


def calculate_roi(predictions: List[PredictionRecord]) -> Tuple[float, float, float]:
    """
    Calculate ROI and related betting metrics.

    Returns:
        (total_wagered, total_profit, roi_percentage)
    """
    bets = [p for p in predictions if p.bet_placed]

    if not bets:
        return 0.0, 0.0, 0.0

    total_wagered = sum(p.bet_amount for p in bets)
    total_profit = sum(p.bet_profit for p in bets)

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

    return total_wagered, total_profit, roi


def calculate_max_drawdown(
    predictions: List[PredictionRecord],
    starting_bankroll: float,
) -> Tuple[float, float]:
    """
    Calculate maximum drawdown (peak-to-trough decline).

    Returns:
        (max_drawdown_dollars, max_drawdown_percentage)
    """
    if not predictions:
        return 0.0, 0.0

    # Build bankroll curve
    bankroll = starting_bankroll
    bankrolls = [bankroll]

    for p in predictions:
        if p.bet_placed:
            bankroll += p.bet_profit
        bankrolls.append(bankroll)

    # Calculate drawdown
    peak = bankrolls[0]
    max_dd = 0.0

    for b in bankrolls:
        if b > peak:
            peak = b
        dd = peak - b
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0

    return max_dd, max_dd_pct


def calculate_sharpe_ratio(
    predictions: List[PredictionRecord],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).

    Assumes returns are measured per bet (not annualized).
    risk_free_rate is per-bet equivalent (usually ~0 for sports betting).

    Returns:
        Sharpe ratio (higher is better)
    """
    bets = [p for p in predictions if p.bet_placed]

    if len(bets) < 2:
        return 0.0

    # Calculate return per bet
    returns = []
    for p in bets:
        ret = p.bet_profit / p.bet_amount if p.bet_amount > 0 else 0.0
        returns.append(ret)

    returns_arr = np.array(returns)
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    return float(sharpe)


def calculate_clv(
    predictions: List[PredictionRecord],
) -> Tuple[Optional[float], int, int]:
    """
    Calculate Closing Line Value (CLV).

    CLV measures how much better our line was than the closing line.
    Positive CLV = we got a better price than the closing line.

    Returns:
        (average_clv, clv_wins, clv_total)
    """
    clv_games = [
        p
        for p in predictions
        if p.bet_placed and p.closing_line is not None and p.market_spread is not None
    ]

    if not clv_games:
        return None, 0, 0

    clv_values = []
    clv_wins = 0

    for p in clv_games:
        # CLV = line we got - closing line
        # Positive CLV means we got a better line
        our_line = p.market_spread
        closing = p.closing_line

        if p.bet_on_home:
            clv = closing - our_line
        else:
            clv = our_line - closing

        clv_values.append(clv)
        if clv > 0:
            clv_wins += 1

    avg_clv = float(np.mean(clv_values)) if clv_values else 0.0

    return avg_clv, clv_wins, len(clv_games)


def calculate_edge_bucket_accuracy(
    predictions: List[PredictionRecord],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate accuracy by edge bucket.

    Edge = |model_prob - market_prob|

    Returns:
        (edge_1_3_pct_accuracy, edge_3_5_pct_accuracy, edge_5_plus_pct_accuracy)
    """
    edge_1_3 = []
    edge_3_5 = []
    edge_5_plus = []

    for p in predictions:
        if p.market_home_win_prob is None:
            continue

        edge = abs(p.predicted_home_win_prob - p.market_home_win_prob)
        predicted_correct = (
            p.predicted_home_win_prob >= 0.5
            and p.actual_home_win
            or p.predicted_home_win_prob < 0.5
            and not p.actual_home_win
        )

        if 0.01 <= edge < 0.03:
            edge_1_3.append(predicted_correct)
        elif 0.03 <= edge < 0.05:
            edge_3_5.append(predicted_correct)
        elif edge >= 0.05:
            edge_5_plus.append(predicted_correct)

    acc_1_3 = sum(edge_1_3) / len(edge_1_3) if edge_1_3 else None
    acc_3_5 = sum(edge_3_5) / len(edge_3_5) if edge_3_5 else None
    acc_5_plus = sum(edge_5_plus) / len(edge_5_plus) if edge_5_plus else None

    return acc_1_3, acc_3_5, acc_5_plus


def calculate_ats_when_disagree(
    predictions: List[PredictionRecord],
) -> Tuple[Optional[str], Optional[float]]:
    """
    Calculate Against The Spread (ATS) record when model disagrees with market.

    Returns:
        (ats_record, ats_win_rate) where record is "W-L-P" format
    """
    disagree_games = []

    for p in predictions:
        if p.market_home_win_prob is None:
            continue

        model_picks_home = p.predicted_home_win_prob >= 0.5
        market_picks_home = p.market_home_win_prob >= 0.5

        if model_picks_home != market_picks_home:
            disagree_games.append(p)

    if not disagree_games:
        return None, None

    wins = 0
    losses = 0
    pushes = 0

    for p in disagree_games:
        if p.market_spread is None:
            continue

        model_picks_home = p.predicted_home_win_prob >= 0.5
        actual_margin = p.actual_home_score - p.actual_away_score
        covered_spread = actual_margin + p.market_spread

        # Model picked home
        if model_picks_home:
            if covered_spread > 0:
                wins += 1
            elif covered_spread < 0:
                losses += 1
            else:
                pushes += 1
        # Model picked away
        else:
            if covered_spread < 0:
                wins += 1
            elif covered_spread > 0:
                losses += 1
            else:
                pushes += 1

    record = f"{wins}-{losses}-{pushes}"
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    return record, win_rate
