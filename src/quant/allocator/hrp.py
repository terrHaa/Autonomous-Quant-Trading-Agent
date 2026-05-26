"""hrp.py — Hierarchical Risk Parity allocator.

HRP converts a returns matrix into long-only weights summing to 1.0, via
three steps: hierarchical clustering of pairwise correlation distances,
quasi-diagonalization of the leaves, then recursive bisection allocating
inversely to cluster variance.

Robust to singular/near-singular covariance, to highly-correlated assets,
and to estimation noise — see ``docs/specs/hrp.md`` for the design spec
and worked example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def hrp_weights(
    returns: pd.DataFrame,
    *,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute HRP portfolio weights from a returns DataFrame.

    Parameters
    ----------
    returns
        Rows = time, columns = asset/strategy names. Must have at least 2
        columns and 2 rows. No NaNs allowed (clean upstream — usually a
        ``returns.dropna()`` is enough).
    linkage_method
        Method passed to ``scipy.cluster.hierarchy.linkage``. Default
        ``"single"`` matches López de Prado (2016); other options:
        ``"complete"``, ``"average"``, ``"ward"``.

    Returns
    -------
    pandas.Series
        Weights indexed by the input column names. Non-negative,
        sums to 1.0 by construction.

    Raises
    ------
    ValueError
        If the input has fewer than 2 columns/rows or contains NaNs.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = pd.DataFrame(rng.normal(0, 0.01, (252, 3)),
    ...                        columns=['A', 'B', 'C'])
    >>> w = hrp_weights(returns)
    >>> assert (w >= 0).all() and abs(w.sum() - 1.0) < 1e-10
    """
    if returns.shape[1] < 2:
        raise ValueError(
            f"HRP requires at least 2 columns; got {returns.shape[1]}"
        )
    if returns.shape[0] < 2:
        raise ValueError(
            f"HRP requires at least 2 rows of returns; got {returns.shape[0]}"
        )
    if returns.isna().any().any():
        # NaNs would propagate into the correlation matrix and silently
        # corrupt the distance metric. Force the caller to be explicit.
        raise ValueError(
            "HRP doesn't support NaN in returns. Drop or fill them upstream."
        )

    cov = returns.cov().values
    corr = returns.corr().values

    # Distance: 0 when corr=1 (identical), 1 when corr=-1 (perfect inverse).
    # Standard HRP "distance from perfect correlation" metric.
    # `np.maximum(..., 0)` guards against tiny negatives from float math
    # when corr is very close to 1.
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    # Diagonal should be exactly 0; insurance against floating-point drift.
    np.fill_diagonal(dist, 0.0)

    # `squareform` converts the NxN distance matrix to a condensed 1D
    # vector that scipy.linkage requires. `checks=False` skips symmetry/
    # zero-diagonal assertions — we know the matrix is well-formed.
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    # Leaf order of the dendrogram = quasi-diagonal order. Adjacent leaves
    # are correlated; the covariance matrix is roughly block-diagonal
    # when permuted to this order.
    order = leaves_list(Z).tolist()

    # Recursive bisection: walk the ordered list, splitting in halves
    # and allocating inversely to each half's variance.
    weights = _recursive_bisection(cov, order)

    return pd.Series(weights, index=returns.columns)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Compute HRP weights via top-down recursive bisection.

    Iterative implementation (queue of clusters) rather than recursion to
    avoid Python's recursion limit on large universes (default ~1000).
    """
    n = cov.shape[0]
    # Start every asset at weight 1.0; each level of the tree multiplies
    # by a cluster allocation ∈ (0, 1), so weights monotonically shrink
    # to their final values as we descend the tree.
    weights = np.ones(n)

    clusters: list[list[int]] = [order]
    while clusters:
        cluster = clusters.pop()
        if len(cluster) <= 1:
            # Singleton: its accumulated weight is final.
            continue

        # Split along the quasi-diagonal. Integer floor of mid; any extra
        # element when len(cluster) is odd goes to the right half (matches
        # López de Prado's reference Python code).
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        var_left = _cluster_var(cov, left)
        var_right = _cluster_var(cov, right)

        # Allocation to the LEFT half: more weight if left has lower
        # variance. Equivalent formulation:
        #   alpha = 1 / (1 + var_left/var_right) when var_right > 0.
        # Below is the López de Prado form, robust to either side being 0.
        alpha = 1.0 - var_left / (var_left + var_right)

        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1.0 - alpha

        clusters.append(left)
        clusters.append(right)

    return weights


def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    """Variance of an inverse-variance-weighted portfolio of the given assets.

    This is the quantity used to decide HOW MUCH to allocate to a sub-cluster
    in the recursive step. We weight intra-cluster by inverse variance (the
    natural choice for "naive risk parity"); the resulting portfolio's
    variance is what we compare across siblings.
    """
    cov_sub = cov[np.ix_(indices, indices)]
    # Inverse-variance weights within the cluster, then normalize to sum 1.
    inv_var = 1.0 / np.diag(cov_sub)
    w = inv_var / inv_var.sum()
    # Portfolio variance under those weights.
    return float(w @ cov_sub @ w)
