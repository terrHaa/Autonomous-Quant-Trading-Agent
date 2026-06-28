"""promotion.py — event-driven promotion criteria (research substrate, phase A4).

Decides when a shadow candidate has earned a live seat. This replaces the
fixed monthly deploy tick with "promote when ready" — but "ready" is a
checklist of evidence, never just "the backtest looked good." A candidate
promotes only when ALL of these hold (design doc §5):

  1. OOS shadow length — enough live paper days to mean something.
  2. Out-of-sample survival — the shadow Sharpe hasn't collapsed versus
     the backtest (the in-sample promise held up live).
  3. Global Deflated Sharpe — the shadow record clears the DSR floor when
     deflated against the WHOLE trial population in the ledger (so a
     candidate found by searching a thousand variants is held to a higher
     bar than one found in three).
  4. Diversification — correlation to the current book under a ceiling; a
     0.6-Sharpe uncorrelated sleeve is worth more than a 1.0-Sharpe clone.

The deployment-rate budget (≤1 promotion / ~30 days) is enforced by the
orchestrator over the *set* of ready candidates, not here — this function
judges one candidate on its own merits.

Pure and side-effect-free: it reads, it decides, it returns. The caller
records the outcome to the ledger and queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.research.ledger import TrialLedger
from quant.research.shadow_queue import ShadowCandidate

TRADING_DAYS = 252


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    promote: bool
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    reason: str = ""

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "HOLD"
        lines = [f"{head} {self.candidate_id}: {self.reason}"]
        for name, (ok, detail) in self.checks.items():
            lines.append(f"  [{'x' if ok else ' '}] {name}: {detail}")
        return "\n".join(lines)


def _ann_sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else 0.0


def evaluate_promotion(
    candidate: ShadowCandidate,
    *,
    ledger: TrialLedger,
    book_returns: pd.Series | None = None,
    min_shadow_days: int = 20,
    dsr_floor: float = 0.60,
    corr_ceiling: float = 0.70,
    max_sharpe_giveback: float = 1.0,
) -> PromotionDecision:
    """Evaluate one shadow candidate against the promotion gates.

    ``book_returns`` (the current live book's daily returns) enables the
    diversification check; if absent, that check is skipped (recorded as
    "n/a — passed") rather than blocking promotion.
    """
    checks: dict[str, tuple[bool, str]] = {}
    shadow = candidate.returns_series()
    n = len(shadow)

    # 1. shadow length
    long_enough = n >= min_shadow_days
    checks["shadow_length"] = (long_enough, f"{n}/{min_shadow_days} OOS days")

    # 2. out-of-sample survival (shadow Sharpe vs backtest)
    shadow_sh = _ann_sharpe(shadow) if n >= 5 else 0.0
    giveback = candidate.backtest_sharpe - shadow_sh
    survived = n >= 5 and shadow_sh > 0 and giveback <= max_sharpe_giveback
    checks["oos_survival"] = (
        survived,
        f"shadow Sharpe {shadow_sh:+.2f} vs backtest "
        f"{candidate.backtest_sharpe:+.2f} (giveback {giveback:+.2f})",
    )

    # 3. global deflated Sharpe
    dsr = 0.0
    if n >= 10:
        try:
            dsr = ledger.deflated_sharpe(
                shadow, kind="strategy", include_self_sharpe=shadow_sh,
            )
        except Exception:
            dsr = 0.0
    dsr_ok = dsr >= dsr_floor
    checks["global_dsr"] = (dsr_ok, f"DSR {dsr:.2f} vs floor {dsr_floor:.2f}")

    # 4. diversification (correlation to book)
    if book_returns is not None and n >= 10:
        joined = pd.concat([shadow.rename("c"), book_returns.rename("b")],
                           axis=1, join="inner").dropna()
        if len(joined) >= 10:
            corr = float(joined["c"].corr(joined["b"]))
            div_ok = abs(corr) <= corr_ceiling
            checks["diversification"] = (
                div_ok, f"corr to book {corr:+.2f} vs ceiling {corr_ceiling:.2f}",
            )
        else:
            checks["diversification"] = (True, "insufficient overlap — skipped")
    else:
        checks["diversification"] = (True, "no book returns — skipped")

    promote = all(ok for ok, _ in checks.values())
    if promote:
        reason = "all gates passed"
    else:
        failed = [k for k, (ok, _) in checks.items() if not ok]
        reason = f"waiting on: {', '.join(failed)}"
    return PromotionDecision(
        candidate_id=candidate.candidate_id, promote=promote,
        checks=checks, reason=reason,
    )
