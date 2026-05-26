"""reports — markdown report generation.

Generates human-readable reports from backtest and live-run outputs:
- Tear sheets (returns, drawdowns, exposures, top contributors).
- Walk-forward summaries (per-fold OOS performance, stability metrics).
- Registry snapshots (what's in research vs paper vs live, and why).

Reports are markdown so they render anywhere (GitHub, editors, the LLM agent).
Charts are rendered to PNG and linked inline.
"""
