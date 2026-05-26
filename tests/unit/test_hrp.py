"""Tests for the HRP allocator.

The tricky tests use small hand-constructable covariance matrices where the
expected weights are computable by hand from the algorithm. The looser tests
check structural properties (sum-to-1, non-negativity, robustness to
duplicates) that hold for ANY HRP implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.allocator import hrp_weights


def _make_returns(cov: np.ndarray, *, n_obs: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Synthesize returns with a given covariance.

    n_obs is large (5000) so the sample covariance is close to the
    population covariance — lets us assert on expected weights with
    tight tolerances.
    """
    rng = np.random.default_rng(seed)
    # Cholesky: returns = L @ N(0,I). Then sample cov ≈ L L^T = cov.
    L = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n_obs, cov.shape[0])) @ L.T
    cols = [chr(ord("A") + i) for i in range(cov.shape[0])]
    return pd.DataFrame(raw, columns=cols)


# ---------------------------------------------------------------------------
# Structural properties (must hold for any valid input)
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_and_are_nonneg() -> None:
    """The fundamental HRP guarantees — non-negative, sums to 1.0."""
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, 4)),
                           columns=["A", "B", "C", "D"])
    w = hrp_weights(returns)
    assert (w >= 0).all(), "HRP weights must be non-negative"
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_weights_indexed_by_input_columns() -> None:
    """Output Series index must match the input column order."""
    rng = np.random.default_rng(42)
    cols = ["alpha", "bravo", "charlie", "delta"]
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, 4)), columns=cols)
    w = hrp_weights(returns)
    assert list(w.index) == cols


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_single_column() -> None:
    """HRP is meaningless with one asset; raise rather than return 1.0."""
    returns = pd.DataFrame({"only": [0.01, -0.005, 0.02]})
    with pytest.raises(ValueError, match="at least 2 columns"):
        hrp_weights(returns)


def test_rejects_single_row() -> None:
    """Need at least 2 observations to estimate covariance."""
    returns = pd.DataFrame({"A": [0.01], "B": [-0.005]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        hrp_weights(returns)


def test_rejects_nans() -> None:
    """Don't let NaNs silently poison the covariance matrix."""
    returns = pd.DataFrame({
        "A": [0.01, -0.005, 0.02, 0.01],
        "B": [0.02, np.nan, 0.01, -0.003],
    })
    with pytest.raises(ValueError, match="NaN"):
        hrp_weights(returns)


# ---------------------------------------------------------------------------
# Algorithmic correctness — two-asset case
# ---------------------------------------------------------------------------


def test_two_uncorrelated_equal_vol_assets_get_equal_weight() -> None:
    """A=B in every way → 50/50 weights."""
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    returns = _make_returns(cov, n_obs=5000)
    w = hrp_weights(returns)
    assert w["A"] == pytest.approx(0.5, abs=0.02)
    assert w["B"] == pytest.approx(0.5, abs=0.02)


def test_two_assets_lower_vol_gets_more_weight() -> None:
    """σ_A=1, σ_B=2 → A gets ~4x B's weight (inverse-variance)."""
    cov = np.array([[1.0, 0.0], [0.0, 4.0]])
    returns = _make_returns(cov, n_obs=5000)
    w = hrp_weights(returns)
    # Theoretical: w_A / w_B = 4 (inverse-variance), w_A + w_B = 1.
    # → w_A = 0.8, w_B = 0.2.
    assert w["A"] == pytest.approx(0.8, abs=0.05)
    assert w["B"] == pytest.approx(0.2, abs=0.05)


# ---------------------------------------------------------------------------
# Algorithmic correctness — four-asset two-cluster case
# ---------------------------------------------------------------------------


def test_low_vol_cluster_receives_more_weight() -> None:
    """A, B (σ=1, intra-corr 0.8) | C, D (σ=2, intra-corr 0.4) | zero cross.

    Hand-computed in docs/specs/hrp.md:
      cluster (A,B) variance under inv-var = 0.9
      cluster (C,D) variance under inv-var = 2.8
      alpha_left = 1 - 0.9/3.7 ≈ 0.757
    Within each cluster, equal-vol pairs → 50/50.
    Expected: A ≈ B ≈ 0.378, C ≈ D ≈ 0.122.
    """
    cov = np.array([
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 1.6],
        [0.0, 0.0, 1.6, 4.0],
    ])
    returns = _make_returns(cov, n_obs=10_000)
    w = hrp_weights(returns)

    # Low-vol cluster gets ~76% allocation.
    assert (w["A"] + w["B"]) > (w["C"] + w["D"])
    # Within each cluster, equal-vol pairs → equal split.
    assert w["A"] == pytest.approx(w["B"], abs=0.03)
    assert w["C"] == pytest.approx(w["D"], abs=0.03)
    # Order of magnitude matches the worked example (loose tolerance).
    assert w["A"] == pytest.approx(0.378, abs=0.05)
    assert w["C"] == pytest.approx(0.122, abs=0.05)


# ---------------------------------------------------------------------------
# Robustness to duplicates (the property MV famously fails)
# ---------------------------------------------------------------------------


def test_duplicate_asset_splits_its_allocation() -> None:
    """Adding a perfect duplicate of B should split B's prior weight.

    Setup: A, B uncorrelated, equal vol. Without duplication, HRP gives
    50/50. With B duplicated as B and C (perfectly correlated), we expect:
        A ≈ 0.5, B + C ≈ 0.5, B ≈ C (the duplicate pair splits evenly).
    This is the MV-killer: mean-variance gives arbitrary weights to B vs C
    because the covariance matrix is singular.
    """
    # Perfect correlation between B and C; A independent.
    cov = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    # Add tiny diagonal noise so cholesky doesn't fail on singular matrix.
    cov = cov + np.eye(3) * 1e-6
    returns = _make_returns(cov, n_obs=10_000)
    w = hrp_weights(returns)

    # A keeps about half.
    assert w["A"] == pytest.approx(0.5, abs=0.05)
    # The duplicate pair splits the other half evenly.
    assert w["B"] + w["C"] == pytest.approx(0.5, abs=0.05)
    assert w["B"] == pytest.approx(w["C"], abs=0.05)
