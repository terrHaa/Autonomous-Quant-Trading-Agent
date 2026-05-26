"""evaluation — strategy metrics with overfitting controls.

A naive Sharpe ratio on a backtest is *not* evidence a strategy works. We
explicitly correct for the two biggest research-pipeline biases:

- **Selection bias / multiple testing.** If we try 100 variants and report the
  best Sharpe, that Sharpe is inflated. The Deflated Sharpe Ratio (DSR;
  Bailey & López de Prado, 2014) adjusts for the number of trials and the
  skew/kurtosis of returns.
- **Look-ahead in evaluation.** Walk-forward analysis re-fits on a rolling
  training window and evaluates only on the subsequent out-of-sample window,
  preventing the evaluator from leaking future information.

Currently available:
- ``Metrics``, ``compute_metrics``, ``metrics_for`` — standard performance
  ratios (Sharpe, Sortino, Calmar, max drawdown, etc.)
- DSR and walk-forward to come.

See docs/specs/dsr.md (TBD) for the DSR math and rationale.
"""

from quant.evaluation.dsr import (
    deflated_sharpe_ratio,
    dsr_for,
    estimate_var_sr_from_trials,
    probabilistic_sharpe_ratio,
    psr_for,
)
from quant.evaluation.metrics import Metrics, compute_metrics, metrics_for

__all__ = [
    "Metrics",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "dsr_for",
    "estimate_var_sr_from_trials",
    "metrics_for",
    "probabilistic_sharpe_ratio",
    "psr_for",
]
