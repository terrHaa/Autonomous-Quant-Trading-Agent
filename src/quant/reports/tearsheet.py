"""tearsheet.py — markdown tear-sheet reports for a backtest result.

Renders a ``BacktestResult`` into a markdown document with the standard
sections: header, headline metrics, trading activity, cost breakdown, and
top fills. Output is plain markdown so it renders anywhere (GitHub,
editors, the LLM agent layer when it lands).

Design notes
------------
- **No Jinja2.** The template is short enough that f-string composition
  beats template indirection. We still import jinja2 nowhere — fewer
  moving parts means fewer ways the formatting silently drifts.
- **No matplotlib.** PNG equity-curve / drawdown charts are deferred to
  the ``notebooks`` extra. The markdown table of fills + metrics is
  enough to triage a backtest; chart rendering is a future enhancement.
- **The tearsheet doesn't compute metrics from scratch.** It calls
  ``metrics_for(result)`` so a tearsheet and an interactive Metrics
  query produce the same numbers (no drift between display formats).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.evaluation.metrics import Metrics, metrics_for


def render_tearsheet(
    result: BacktestResult,
    *,
    title: str | None = None,
    top_n_fills: int = 10,
) -> str:
    """Render a markdown tearsheet string.

    Parameters
    ----------
    result
        The backtest result to summarize.
    title
        Title to put at the top of the document. Defaults to the
        strategy's name.
    top_n_fills
        How many fills (by absolute notional) to surface in the
        "Top fills" table. Set to 0 to hide that section.

    Returns
    -------
    str
        Markdown content. Pass to ``write_tearsheet`` or print directly.
    """
    metrics = metrics_for(result)
    parts: list[str] = []

    # ---- Header --------------------------------------------------------
    parts.append(_render_header(result, metrics, title=title))

    # ---- Headline metrics ----------------------------------------------
    parts.append(_render_metrics_table(metrics))

    # ---- Trading activity ----------------------------------------------
    parts.append(_render_trading_activity(result))

    # ---- Cost breakdown ------------------------------------------------
    parts.append(_render_cost_breakdown(result))

    # ---- Top fills -----------------------------------------------------
    if top_n_fills > 0 and not result.fills.empty:
        parts.append(_render_top_fills(result, n=top_n_fills))

    # ---- Footer --------------------------------------------------------
    parts.append(_render_footer())

    # `\n\n` between sections so markdown renders blank lines between
    # headings — most renderers require this for proper block separation.
    return "\n\n".join(p for p in parts if p) + "\n"


def write_tearsheet(
    result: BacktestResult,
    path: Path | str,
    *,
    title: str | None = None,
    top_n_fills: int = 10,
) -> Path:
    """Render and write the tearsheet to ``path``. Returns the resolved path.

    Creates the parent directory if it doesn't exist. Overwrites any
    existing file at ``path`` — a deliberate re-run replaces the prior
    report rather than appending.
    """
    md = render_tearsheet(result, title=title, top_n_fills=top_n_fills)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return out


# ---------------------------------------------------------------------------
# Section renderers (private). Each returns a self-contained markdown block.
# ---------------------------------------------------------------------------


def _render_header(
    result: BacktestResult,
    metrics: Metrics,
    *,
    title: str | None,
) -> str:
    """Title + one-paragraph summary."""
    heading = title or result.strategy_name
    start = result.metadata.get("start_date")
    end = result.metadata.get("end_date")
    return (
        f"# {heading}\n\n"
        f"**Strategy:** `{result.strategy_name}`  \n"
        f"**Window:** {start} → {end} ({metrics.n_days} days)  \n"
        f"**Equity:** ${metrics.starting_equity:,.2f} "
        f"→ ${metrics.ending_equity:,.2f} "
        f"({_pct(metrics.total_return)})"
    )


def _render_metrics_table(metrics: Metrics) -> str:
    """The headline tear-sheet table."""
    rows = [
        ("Total return",   _pct(metrics.total_return)),
        ("CAGR",           _pct(metrics.cagr)),
        ("Annualized vol", _pct(metrics.annualized_vol)),
        ("Sharpe",         f"{metrics.sharpe:+.2f}"),
        ("Sortino",        f"{metrics.sortino:+.2f}"),
        ("Calmar",         f"{metrics.calmar:+.2f}"),
        ("Max drawdown",   _pct(metrics.max_drawdown)),
        ("Max DD days",    f"{metrics.max_drawdown_duration_days}"),
        ("Hit rate",       _pct(metrics.hit_rate)),
    ]
    return "## Headline metrics\n\n" + _md_table(
        headers=["Metric", "Value"],
        rows=rows,
    )


def _render_trading_activity(result: BacktestResult) -> str:
    """Counts of orders and fills + run time."""
    md = result.metadata
    rows = [
        ("Orders generated",   f"{md.get('n_orders', 0)}"),
        ("Fills",              f"{md.get('n_fills', 0)}"),
        ("Bars processed",     f"{md.get('n_bars', 0)}"),
        ("Run time",           f"{md.get('run_time_s', 0):.3f}s"),
    ]
    return "## Trading activity\n\n" + _md_table(
        headers=["Item", "Value"],
        rows=rows,
    )


def _render_cost_breakdown(result: BacktestResult) -> str:
    """Per-component total cost over the whole run."""
    if result.costs.empty:
        return "## Costs paid\n\n_No trading activity._"

    spread = float(result.costs["spread_cost"].sum())
    slippage = float(result.costs["slippage_cost"].sum())
    commission = float(result.costs["commission"].sum())
    total = spread + slippage + commission

    start_equity = float(result.metadata.get("starting_equity", 1.0))
    # bps of equity: a useful "is this strategy paying too much in costs" number.
    bps = (total / start_equity) * 10_000 if start_equity > 0 else 0.0

    rows = [
        ("Spread",     f"${spread:,.2f}"),
        ("Slippage",   f"${slippage:,.2f}"),
        ("Commission", f"${commission:,.2f}"),
        ("**Total**",  f"**${total:,.2f}** ({bps:.1f} bps of starting equity)"),
    ]
    return "## Costs paid\n\n" + _md_table(
        headers=["Component", "Amount"],
        rows=rows,
    )


def _render_top_fills(result: BacktestResult, *, n: int) -> str:
    """Top N fills sorted by absolute notional."""
    fills = result.fills.copy()
    fills["abs_notional"] = fills["notional"].abs()
    top = fills.sort_values("abs_notional", ascending=False).head(n)

    rows = []
    for _, f in top.iterrows():
        rows.append((
            str(f["date"]),
            str(f["symbol"]),
            str(f["side"]),
            f"{int(f['qty']):,}",
            f"${float(f['fill_price']):,.2f}",
            f"${float(f['notional']):,.2f}",
        ))

    return (
        f"## Top {len(top)} fills by notional\n\n"
        + _md_table(
            headers=["Date", "Symbol", "Side", "Qty", "Fill price", "Notional"],
            rows=rows,
        )
    )


def _render_footer() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"---\n\n_Generated {now} by `quant.reports.tearsheet`._"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    """Format a decimal as a signed percentage with 2 decimal places."""
    return f"{x:+.2%}"


def _md_table(
    *,
    headers: list[str],
    rows: list[tuple],
) -> str:
    """Render a markdown table from headers + rows.

    Uses a minimal pipe-delimited format. No alignment fences (`:---:`)
    because most renderers default-align numbers acceptably, and the
    fences add noise.

    Example output:
        | Metric | Value |
        | --- | --- |
        | CAGR | +10.39% |
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [str(c) for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
