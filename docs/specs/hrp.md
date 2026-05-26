# HRP — Hierarchical Risk Parity allocator

**Status:** Draft v1
**Date:** 2026-05-26
**Reference:** López de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample" (2016).

## What this is

HRP converts a returns matrix (rows = time, columns = assets or strategies) into a set of **long-only weights summing to 1.0**, designed to be robust to:

- Singular or near-singular covariance matrices (mean-variance breaks; HRP doesn't).
- Duplicate or highly-correlated assets (MV gives wild weights to duplicates; HRP splits cleanly).
- Estimation noise in covariance estimates (HRP uses only the *structure* of correlations for top-level allocation, not exact magnitudes).

Output is a `pd.Series` of weights — one per input column — that the rest of the platform plugs into the engine via a meta-strategy or post-strategy weight blending step (the wiring is the user's; this module just produces weights).

## Why this over mean-variance

Markowitz mean-variance (MV) has three well-known pathologies in practice:

1. **Tiny estimation errors → extreme weights.** Sample covariance is noisy; MV amplifies the noise.
2. **N > T means singular covariance.** Below 252 days of history for 500 assets, MV simply can't solve.
3. **Duplicate assets break it.** Adding a perfect duplicate makes the matrix singular; MV weights become arbitrary.

HRP sidesteps all three. It clusters first using pairwise correlation distances, then allocates recursively along the tree. The result is mathematically defined even when N ≫ T.

The price: HRP is *not* "optimal" in the MV sense — it doesn't maximize Sharpe under the input estimates. But MV's "optimum" is illusory once you account for estimation noise. The paper's eponymous finding is that HRP **out-performs MV out-of-sample** on diverse data.

## The algorithm

Three steps:

### 1. Hierarchical clustering

From correlation $C$, build distance $D$ where

$$D_{ij} = \sqrt{0.5 \cdot (1 - C_{ij})}$$

This metric is 0 when assets are perfectly correlated and 1 when perfectly anti-correlated. Run single-linkage hierarchical clustering on $D$.

### 2. Quasi-diagonalization

Walk the clustering tree to get a leaf ordering where similar assets are adjacent. This gives the covariance matrix a quasi-block-diagonal structure when reordered. We use `scipy.cluster.hierarchy.leaves_list` for this.

### 3. Recursive bisection

Starting with all assets ordered quasi-diagonally:
- Split the list in half: left and right.
- For each half, compute the variance of the inverse-variance-weighted sub-portfolio:

$$\text{var}(\text{cluster}) = w^T \Sigma_{\text{sub}} w \quad \text{where} \quad w_i \propto 1 / \sigma_i^2$$

- Allocate between left and right inversely to variance:

$$\alpha_{\text{left}} = 1 - \frac{\text{var}(\text{left})}{\text{var}(\text{left}) + \text{var}(\text{right})}$$

- Multiply each asset's weight in the left half by $\alpha_{\text{left}}$, each in the right half by $1 - \alpha_{\text{left}}$.
- Recurse on each half. Singletons keep their accumulated weight.

Final weights are non-negative and sum to 1.0 by construction (every step is a convex combination).

## Worked example

Four assets, two clusters:
- A, B: $\sigma=1$, intra-correlation 0.8
- C, D: $\sigma=2$, intra-correlation 0.4
- Zero inter-cluster correlation

Clustering puts (A, B) together and (C, D) together. Quasi-diag order: `[A, B, C, D]`.

Recursive bisection:
- First split: left = (A, B), right = (C, D).
  - $\text{var}(\text{left}) = 0.5^2 \cdot 1 + 0.5^2 \cdot 1 + 2 \cdot 0.5^2 \cdot 0.8 = 0.9$
  - $\text{var}(\text{right}) = 0.5^2 \cdot 4 + 0.5^2 \cdot 4 + 2 \cdot 0.5^2 \cdot 1.6 = 2.8$
  - $\alpha_{\text{left}} = 1 - 0.9 / 3.7 \approx 0.757$
  - Each of A, B carries weight 0.757; each of C, D carries 0.243.
- Within-cluster splits: equal-vol pairs → each gets half.
  - Final: A = B ≈ 0.378, C = D ≈ 0.122. Sum = 1.0.

The low-vol cluster gets 75.7% allocation; within each cluster, equal-vol pairs split evenly. Lower-vol cluster → more capital.

## Limitations

- **Long-only.** HRP weights are non-negative and sum to 1. Use for combining un-leveraged strategies; for net-long-or-short books or leverage, pair HRP with the next layer (vol targeting, Kelly).
- **No view incorporation.** HRP doesn't accept expected-return inputs. It's pure risk-driven. Black-Litterman extensions exist but are out of scope.
- **Single-linkage clustering.** We use the López de Prado default. `complete`, `ward`, etc. produce different trees and slightly different weights — interesting to compare but not a v1 concern.
- **Sample covariance.** No shrinkage (Ledoit-Wolf). For small samples this hurts; revisit if N/T gets aggressive.

## API

```python
from quant.allocator import hrp_weights

# returns: DataFrame with rows=time, columns=strategy/asset names
weights = hrp_weights(returns)
# weights: Series indexed by column names, non-negative, sums to 1.0
```
