"""research — the continuous-research substrate (see docs/design/research-desk.md).

The measurement backbone that keeps fast iteration honest: a global trial
ledger so multiple-testing is accounted for across every experiment ever
run, plus (later) the shadow queue, A/B harness, and promotion criteria.

Public API:
  - ``TrialLedger`` — append-only record of every hypothesis tested; its
    ``deflated_sharpe`` is the global multiple-testing-corrected DSR.
  - ``run_ab_test`` — paired, ledger-deflated A/B test of a structural
    change (variant vs baseline daily returns).
"""

from quant.research.ab_test import ABResult, run_ab_test
from quant.research.ledger import TrialLedger, TrialView
from quant.research.promotion import PromotionDecision, evaluate_promotion
from quant.research.shadow_queue import ShadowCandidate, ShadowQueue

__all__ = [
    "ABResult",
    "PromotionDecision",
    "ShadowCandidate",
    "ShadowQueue",
    "TrialLedger",
    "TrialView",
    "evaluate_promotion",
    "run_ab_test",
]
