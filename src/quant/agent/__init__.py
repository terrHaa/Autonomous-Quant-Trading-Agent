"""agent — the autonomous trading agent.

Originally a placeholder for an LLM-driven agent layer; now houses the
mechanical autonomous trader the operator commissioned. Composes the
existing platform pieces — strategies, allocator, risk overlay, executor,
registry — into a daily-cadence loop that:

- Loads the top-100 S&P snapshot universe.
- Runs three strategies via the ENSEMBLE (SMA crossover + mean reversion
  + cross-sectional momentum), combining their per-symbol targets via
  HRP weights persisted in ``EnsembleState``.
- Hard-enforces the 20%-per-trade rule via the risk overlay.
- Submits market entries with atomic 5% stop-loss via Alpaca's OTO
  bracket orders.
- Logs every action to disk and to the registry.
- Emails the operator a daily summary after market close.

Plus weekly + monthly review jobs that self-improve the strategy mix:
- Weekly (Friday after close): refit HRP weights across the three
  strategies from a rolling 252-day backtest.
- Monthly (1st of month): tune cross-sectional momentum parameters
  through a DSR-gated grid search; apply the winner if it passes the
  Sharpe / drawdown / DSR ≥ 0.95 gates against the registered trial
  population.

Currently exported:
- ``EmailSender`` — SMTP wrapper for the report-out path.
- ``ImprovementResult`` from ``improver`` — what the monthly review
  searched and decided.
- ``StrategyParams`` — the cross-sectional-momentum parameter tuple
  used as the improver's candidate data-shape.
"""

from quant.agent.email_sender import EmailConfig, EmailSender
from quant.agent.improver import ImprovementCandidate, ImprovementResult
from quant.agent.params import StrategyParams

__all__ = [
    "EmailConfig",
    "EmailSender",
    "ImprovementCandidate",
    "ImprovementResult",
    "StrategyParams",
]
