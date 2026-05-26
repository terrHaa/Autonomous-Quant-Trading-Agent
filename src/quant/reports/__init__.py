"""reports — markdown report generation.

Generates human-readable reports from backtest and live-run outputs:
- Tear sheets (returns, drawdowns, top fills, cost breakdown).
- Walk-forward summaries (per-fold OOS performance, stability metrics).
  TODO.
- Registry snapshots (what's in research vs paper vs live, and why).
  TODO.

Reports are markdown so they render anywhere (GitHub, editors, the LLM
agent). PNG charts (equity, drawdown) are deferred to the ``notebooks``
extras for now — the markdown numbers are enough to triage.

Currently available:
- ``render_tearsheet(result)`` — return a markdown string.
- ``write_tearsheet(result, path)`` — write to a file.
"""

from quant.reports.tearsheet import render_tearsheet, write_tearsheet

__all__ = ["render_tearsheet", "write_tearsheet"]
