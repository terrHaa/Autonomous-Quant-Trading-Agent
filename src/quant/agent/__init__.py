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
"""

from quant.agent.email_sender import EmailConfig, EmailSender

__all__ = ["EmailConfig", "EmailSender"]
