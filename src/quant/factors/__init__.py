"""Factor library + return attribution — the alpha-vs-beta substrate.

Lets the monthly review answer the central professional question: is the
book producing genuine alpha, or is it being paid known factor premia
(market/momentum/low-vol/reversal) and calling it skill?

Public API:
  - ``compute_factor_returns`` — daily long-short factor return series
    built from OHLCV (no fundamentals required).
  - ``attribute_returns`` — regress a portfolio return series on the
    factors → alpha, factor loadings, t-stats, R².
"""

from quant.factors.attribution import AttributionResult, attribute_returns
from quant.factors.library import FACTOR_NAMES, compute_factor_returns

__all__ = [
    "FACTOR_NAMES",
    "AttributionResult",
    "attribute_returns",
    "compute_factor_returns",
]
