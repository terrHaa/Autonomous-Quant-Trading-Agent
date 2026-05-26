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

Other metrics here: probabilistic Sharpe ratio, Calmar, max drawdown, turnover,
hit rate, exposure decomposition.

See docs/specs/dsr.md for the math and rationale.
"""
