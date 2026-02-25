"""
Backtest report generator.

Produces a standalone HTML report from a ``BacktestResult`` including:
- Summary metrics table
- Bankroll curve (inline SVG via matplotlib if available, else text table)
- Prediction accuracy breakdown
- Betting performance stats

No external template engine required — generates self-contained HTML.
"""

from __future__ import annotations

import html
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.backtesting.types import BacktestResult


# ---------------------------------------------------------------------------
# Lightweight HTML builder
# ---------------------------------------------------------------------------

def _metric_row(label: str, value, fmt: str = ".4f") -> str:
    if value is None:
        return f"<tr><td>{html.escape(label)}</td><td>N/A</td></tr>"
    if isinstance(value, float):
        return f"<tr><td>{html.escape(label)}</td><td>{value:{fmt}}</td></tr>"
    return f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>"


def _try_bankroll_chart_base64(result: BacktestResult) -> Optional[str]:
    """Return base64-encoded PNG of the bankroll curve, or None."""
    if not result.bankroll_curve:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dates = list(result.bankroll_curve.keys())
        values = list(result.bankroll_curve.values())

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(len(values)), values, linewidth=1.5, color="#2563eb")
        ax.set_ylabel("Bankroll ($)")
        ax.set_xlabel("Time Step")
        ax.set_title("Bankroll Over Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except ImportError:
        return None


def _bankroll_text_table(result: BacktestResult) -> str:
    """Fallback ASCII table for bankroll curve."""
    if not result.bankroll_curve:
        return "<p>No bankroll data.</p>"
    rows = []
    for step, (date, val) in enumerate(result.bankroll_curve.items()):
        rows.append(f"<tr><td>{html.escape(str(date))}</td><td>${val:,.2f}</td></tr>")
    return (
        '<table class="tbl"><tr><th>Date</th><th>Bankroll</th></tr>'
        + "\n".join(rows)
        + "</table>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(result: BacktestResult) -> str:
    """
    Generate a self-contained HTML report string from *result*.

    Returns:
        HTML string (UTF-8)
    """
    m = result.metrics

    chart_b64 = _try_bankroll_chart_base64(result)
    if chart_b64:
        bankroll_section = (
            f'<img src="data:image/png;base64,{chart_b64}" '
            f'alt="Bankroll curve" style="max-width:100%"/>'
        )
    else:
        bankroll_section = _bankroll_text_table(result)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Backtest Report — {html.escape(result.run_id)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; color: #1e293b; }}
  h1 {{ color: #0f172a; }}
  h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem; }}
  .tbl {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  .tbl th, .tbl td {{ border: 1px solid #cbd5e1; padding: 0.5rem 0.75rem; text-align: left; }}
  .tbl th {{ background: #f1f5f9; }}
  .meta {{ color: #64748b; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p class="meta">
  Run ID: <strong>{html.escape(result.run_id)}</strong> |
  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} |
  Period: {html.escape(result.actual_start_date)} &rarr; {html.escape(result.actual_end_date)} |
  Games: {result.total_games}
</p>

<h2>Prediction Accuracy</h2>
<table class="tbl">
{_metric_row("Overall Accuracy", m.overall_accuracy)}
{_metric_row("Home Accuracy", m.home_accuracy)}
{_metric_row("Away Accuracy", m.away_accuracy)}
{_metric_row("Favorite Accuracy", m.favorite_accuracy)}
{_metric_row("Underdog Accuracy", m.underdog_accuracy)}
{_metric_row("Log Loss", m.log_loss)}
{_metric_row("Brier Score", m.brier_score)}
</table>

<h2>Betting Performance</h2>
<table class="tbl">
{_metric_row("Total Bets", m.total_bets, "d")}
{_metric_row("Total Wagered", m.total_wagered, ",.2f")}
{_metric_row("Total Profit", m.total_profit, ",.2f")}
{_metric_row("ROI", m.roi, ".2f")}
{_metric_row("Win Rate", m.win_rate)}
</table>

<h2>Risk Metrics</h2>
<table class="tbl">
{_metric_row("Max Drawdown", m.max_drawdown, ",.2f")}
{_metric_row("Max Drawdown %", m.max_drawdown_pct, ".2f")}
{_metric_row("Sharpe Ratio", m.sharpe_ratio)}
{_metric_row("CLV", m.clv)}
</table>

<h2>Bankroll Curve</h2>
{bankroll_section}

{f'<h2>Notes</h2><p>{html.escape(result.notes)}</p>' if result.notes else ""}
</body>
</html>"""


def save_report(result: BacktestResult, path: str) -> Path:
    """Generate and save report to *path*. Returns the Path written."""
    content = generate_html_report(result)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
