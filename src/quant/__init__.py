"""quant — quantitative equity research and trading platform.

Top-level package. Submodules:

- data        Market data ingestion, storage, and integrity checks.
- backtest    Event-driven backtest engine with no-look-ahead guarantees.
- strategies  Individual strategy implementations.
- evaluation  Metrics, Deflated Sharpe Ratio, walk-forward analysis.
- allocator   Portfolio construction: HRP, vol targeting, Kelly.
- risk        Risk overlays that wrap allocator output (drawdown kill, caps).
- registry    Strategy registry and promotion gates.
- execution   Broker interface (Alpaca paper/live).
- agent       LLM agent layer (placeholder — not yet implemented).
- reports     Markdown report generation.
"""

__version__ = "0.1.0"
