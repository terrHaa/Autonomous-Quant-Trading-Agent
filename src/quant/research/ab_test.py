"""ab_test.py — backtest A/B harness (research substrate, phase A3).

Scores a structural change honestly: given the daily return series of a
BASELINE config and a VARIANT config over the same window, it answers
"did the change help, and is the difference real or noise?" — and logs
both arms to the trial ledger so the comparison counts against the global
multiple-testing budget.

Why paired, not two independent Sharpes: both arms trade the same market
path, so most of their day-to-day variation is shared market noise. A
PAIRED test on the daily *difference* series cancels that shared noise and
isolates the effect of the change itself. It will call a real 0.1-Sharpe
improvement significant on far less data than comparing two noisy Sharpe
estimates side by side would. This is the correct test for "does this
structural tweak add value."

The harness is deliberately engine-agnostic: it takes two return series,
however they were produced (a pipeline replay like
tools/validate_sizing_changes.py, the backtest engine, or live shadow
records). Its job is the statistics + the ledger bookkeeping, not running
the backtest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass(frozen=True)
class ABResult:
    """Verdict of a paired A/B test between a baseline and a variant."""

    name: str
    n_obs: int
    baseline_sharpe: float
    variant_sharpe: float
    sharpe_delta: float
    mean_daily_delta: float       # mean of (variant - baseline) daily returns
    delta_tstat: float            # paired t-stat on the difference series
    variant_dsr: float            # global deflated Sharpe of the variant (0..1)
    verdict: str                  # "variant better" | "inconclusive" | "variant worse"
    reason: str

    @property
    def adopt(self) -> bool:
        return self.verdict == "variant better"

    def summary(self) -> str:
        return (
            f"A/B [{self.name}]: {self.verdict} — {self.reason}\n"
            f"  Sharpe {self.baseline_sharpe:+.2f} → {self.variant_sharpe:+.2f} "
            f"(Δ{self.sharpe_delta:+.2f}); paired t={self.delta_tstat:+.2f}; "
            f"variant DSR {self.variant_dsr:.2f}; n={self.n_obs}"
        )


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0


def run_ab_test(
    baseline_returns: pd.Series,
    variant_returns: pd.Series,
    *,
    name: str,
    ledger=None,
    family: str = "structural",
    min_tstat: float = 2.0,
    dsr_floor: float = 0.60,
    log_to_ledger: bool = True,
) -> ABResult:
    """Paired A/B test of variant vs baseline daily returns.

    Parameters
    ----------
    baseline_returns, variant_returns
        Date-indexed daily return series. Inner-joined on common dates.
    ledger
        Optional ``TrialLedger``. When given, both arms are recorded as
        ``kind="structural"`` trials, and the variant's Deflated Sharpe is
        computed against the WHOLE ledger population (so this experiment is
        penalized for every trial ever run, not judged in isolation).
    min_tstat
        Paired-t threshold for calling the difference significant.
    dsr_floor
        The variant's global DSR must clear this to be adopted — guards
        against a difference that's significant in-sample but doesn't
        survive the multiple-testing correction.

    Never raises; returns an "inconclusive" verdict on degenerate input.
    """
    joined = pd.concat(
        [baseline_returns.rename("b"), variant_returns.rename("v")],
        axis=1, join="inner",
    ).dropna(how="any")
    n = len(joined)
    if n < 10:
        return ABResult(name, n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        "inconclusive", f"too few overlapping obs (n={n})")

    b, v = joined["b"], joined["v"]
    delta = v - b
    base_sh, var_sh = _sharpe(b), _sharpe(v)
    mean_delta = float(delta.mean())
    sd = float(delta.std(ddof=1))
    tstat = (mean_delta / sd * np.sqrt(n)) if sd > 0 else 0.0

    variant_dsr = 0.0
    if ledger is not None:
        try:
            if log_to_ledger:
                ledger.log_trial(kind="structural", name=f"{name}::baseline",
                                 backtest_sharpe=base_sh, family=family)
                ledger.log_trial(kind="structural", name=f"{name}::variant",
                                 backtest_sharpe=var_sh, family=family)
            variant_dsr = ledger.deflated_sharpe(
                v, kind="structural", include_self_sharpe=var_sh,
            )
        except Exception as e:
            logger.warning("ab_test: ledger/DSR step failed (%s); dsr=0", e)

    # Verdict: the variant must (a) be paired-significantly better AND
    # (b) survive the global multiple-testing deflation. A significant
    # in-sample edge that the DSR floor rejects is "inconclusive", not
    # "better" — exactly the false-positive guard the ledger exists for.
    if tstat >= min_tstat and variant_dsr >= dsr_floor:
        verdict, reason = "variant better", (
            f"paired t {tstat:+.2f} ≥ {min_tstat} and DSR {variant_dsr:.2f} ≥ {dsr_floor}"
        )
    elif tstat <= -min_tstat:
        verdict, reason = "variant worse", f"paired t {tstat:+.2f} ≤ -{min_tstat}"
    elif tstat >= min_tstat and variant_dsr < dsr_floor:
        verdict, reason = "inconclusive", (
            f"significant in-sample (t {tstat:+.2f}) but DSR {variant_dsr:.2f} "
            f"< {dsr_floor} after deflation — likely overfit"
        )
    else:
        verdict, reason = "inconclusive", (
            f"paired t {tstat:+.2f} within ±{min_tstat} — no clear effect"
        )

    return ABResult(
        name=name, n_obs=n,
        baseline_sharpe=base_sh, variant_sharpe=var_sh,
        sharpe_delta=var_sh - base_sh,
        mean_daily_delta=mean_delta, delta_tstat=float(tstat),
        variant_dsr=variant_dsr, verdict=verdict, reason=reason,
    )
