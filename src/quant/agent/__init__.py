"""agent — the autonomous trading agent.

Originally a placeholder for an LLM-driven agent layer; now houses the
mechanical autonomous trader the operator commissioned. Composes the
existing platform pieces — strategy, allocator, risk overlay, executor,
registry — into a daily-cadence loop that:

- Loads the top-100 S&P snapshot universe.
- Runs CrossSectionalMomentum to get target weights.
- Hard-enforces the 20%-per-trade rule via the risk overlay.
- Submits market entries with atomic 5% stop-loss via Alpaca's OTO.
- Logs every action to the registry.
- Emails the operator a daily summary after market close.

Plus weekly/monthly review jobs that analyze recent performance and
(with the safety-railed auto-apply gate) recommend / apply parameter
tweaks. See docs/specs/ for design details (TBD).

Currently available:
- ``EmailSender`` — SMTP wrapper for the report-out path.
- ``StrategyParams`` + ``load_params`` / ``save_params`` — tunable
  strategy parameters the auto-improver may swap.
- ``ImprovementResult`` from ``improver`` — what the monthly review
  searched and decided.
"""

from quant.agent.email_sender import EmailConfig, EmailSender
from quant.agent.improver import ImprovementCandidate, ImprovementResult
from quant.agent.params import StrategyParams, load_params, save_params

__all__ = [
    "EmailConfig",
    "EmailSender",
    "ImprovementCandidate",
    "ImprovementResult",
    "StrategyParams",
    "load_params",
    "save_params",
]
