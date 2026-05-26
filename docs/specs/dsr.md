# Deflated Sharpe Ratio (DSR) — design spec

**Status:** Draft v1
**Date:** 2026-05-26
**References:** Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" (2014).

## What this is

DSR converts an observed in-sample Sharpe ratio into a **probability** that the strategy has positive true skill, while accounting for two biases that inflate naive Sharpe numbers in practice:

1. **Selection bias from multiple testing.** If you tried 100 variants on the same data and report the best, that Sharpe is a max-of-100, not a single draw. Even under the null of zero skill, the expected max-of-100 is well above zero. DSR deflates the threshold by exactly this expected-max-under-null.

2. **Departure from normality.** Real return distributions have fat tails (high kurtosis) and skew. Sharpe-as-a-Gaussian-statistic overstates significance when these are present.

The output is a probability in [0, 1]:

- **DSR ≈ 0.5** — observed Sharpe matches what luck/null would produce.
- **DSR > 0.95** — ~95% confident the deflated Sharpe is positive (the standard threshold for "this is more than noise").
- **DSR < 0.5** — observed Sharpe is *below* what selection bias on this many trials would have given us. Worse than noise.

## Why this is non-negotiable

Quote from Bailey & López de Prado (2014):

> "Sharpe ratio overstatement is essentially universal in financial research because of selection bias."

If we don't deflate, we will eventually fool ourselves with a backtest that has a Sharpe of 1.5 — because that's roughly the expected best-of-100 under the null. The registry (Step 20) tracks every variant we've tested *to feed DSR honestly*. Skipping registered variants in the trial count = lying to DSR = lying to ourselves.

## The math

### Probabilistic Sharpe Ratio (PSR)

The base ingredient. Given an observed Sharpe `SR_hat` and a benchmark Sharpe `SR_star`, PSR is the probability under the null that the true Sharpe exceeds `SR_star`:

$$
\text{PSR}(SR^*) = \Phi\left(\frac{(\hat{SR} - SR^*) \sqrt{N-1}}{\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4} \hat{SR}^2}}\right)
$$

where:
- `N` — number of return observations.
- `γ_3` — sample skewness (Pearson, bias-corrected).
- `γ_4` — sample kurtosis (Pearson, **not** excess; 3 for a normal distribution).
- `Φ` — standard normal CDF.
- `SR_hat`, `SR_star` — **per-period** Sharpe (not annualized; our API takes annualized values and de-annualizes internally).

For normal returns (γ_3=0, γ_4=3), the denominator simplifies to `sqrt(1 + 0.5 · SR^2)`. The skew/kurtosis terms make the test more conservative for fat-tailed returns.

### DSR — the deflation

DSR is PSR with `SR_star` set to the **expected max Sharpe** under the null of zero skill across `N_trials` independent trials:

$$
SR^*_{DSR} = \sqrt{V[SR]} \cdot \left((1 - \gamma) \cdot \Phi^{-1}\left(1 - \frac{1}{N_{\text{trials}}}\right) + \gamma \cdot \Phi^{-1}\left(1 - \frac{1}{N_{\text{trials}} \cdot e}\right)\right)
$$

where:
- `γ ≈ 0.5772` is the Euler-Mascheroni constant.
- `V[SR]` is the **variance of Sharpe estimates across your trial population**.
- The two `Φ^{-1}` terms encode the asymptotic expected-max of `N` standard normals.

So DSR(SR_hat) = PSR(SR_hat | SR_star = SR*_DSR).

### Intuition

`SR*_DSR` grows roughly as `sqrt(2 log N_trials)`. With more trials, the threshold to beat rises, and your Sharpe-of-1.5 becomes less impressive when 99 of those trials were silently discarded.

## How we use it

### In the registry (Step 20)

Every backtest run is registered with:
- Strategy name + parameters
- Returns series (or summary stats)
- Annualized Sharpe

When considering a strategy for promotion, the registry:
1. Counts how many *related* variants have been tested. (Related = "could have been the one we picked".)
2. Computes V[SR] from the population of trial Sharpes.
3. Asks DSR for the probability the candidate's deflated Sharpe is positive.
4. Requires DSR ≥ 0.95 to promote.

Until the registry exists, the caller must pass `n_trials` and `var_sr_trials_annual` explicitly. This is fine for ad-hoc analysis but not for production decisions.

### Standalone (now)

```python
from quant.evaluation import probabilistic_sharpe_ratio, deflated_sharpe_ratio

# After running 10 SMA variants and computing each one's Sharpe:
trial_sharpes_annual = [0.63, 0.49, 0.37, 0.38, 0.55, 0.41, ...]  # 10 values
var_sr = statistics.variance(trial_sharpes_annual)

# For the variant we want to evaluate (its daily returns):
daily_returns = result.equity_curve.pct_change().dropna()
dsr = deflated_sharpe_ratio(
    daily_returns,
    n_trials=10,
    var_sr_trials_annual=var_sr,
)
print(f"DSR = {dsr:.2%}")  # e.g., 87.43%
```

## What DSR is NOT

- **Not a guarantee.** DSR is a probabilistic statement under the i.i.d. normal null with finite-sample skew/kurtosis. Real returns are non-stationary, autocorrelated, and regime-dependent. A high DSR is *necessary* evidence but not *sufficient*.
- **Not a substitute for walk-forward / OOS testing.** DSR is purely in-sample; it cannot detect a strategy that overfits to the training window without ever being tested OOS. Walk-forward (Step 15) is the other half.
- **Not useful for a single trial.** With `n_trials=1`, `SR*_DSR = 0`, and DSR = PSR(0) — i.e., just the standard t-test on the Sharpe. The deflation does work starting from `n_trials=2`.

## Limitations of our implementation

- **V[SR] must be supplied.** The caller estimates it from the population of trial Sharpes. Until the registry tracks this automatically, easy to under-count trials (and thus under-deflate). A pre-registration habit helps.
- **No clustering of related trials.** López de Prado (2018) refines DSR by clustering similar trials (e.g., neighboring SMA windows) and counting clusters rather than raw trials. We use the simpler Bailey 2014 form; revisit if the registry grows large.
- **Per-period returns assumed i.i.d.** Autocorrelation in returns biases the variance estimate; could under- or over-state significance. Not yet corrected.

## API

Two functions, both in `quant.evaluation.dsr`:

```python
def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe_annual: float = 0.0,
    trading_days_per_year: int = 252,
) -> float: ...

def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    var_sr_trials_annual: float,
    trading_days_per_year: int = 252,
) -> float: ...
```

Plus convenience wrappers `psr_for(result)` and `dsr_for(result, n_trials, var_sr_trials_annual)` that pull config defaults from a `BacktestResult`.
