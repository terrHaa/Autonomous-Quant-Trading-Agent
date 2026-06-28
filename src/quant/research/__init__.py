"""research — the continuous-research substrate (see docs/design/research-desk.md).

The measurement backbone that keeps fast iteration honest: a global trial
ledger so multiple-testing is accounted for across every experiment ever
run, plus (later) the shadow queue, A/B harness, and promotion criteria.

Public API:
  - ``TrialLedger`` — append-only record of every hypothesis tested; its
    ``deflated_sharpe`` is the global multiple-testing-corrected DSR.
"""

from quant.research.ledger import TrialLedger, TrialView

__all__ = ["TrialLedger", "TrialView"]
