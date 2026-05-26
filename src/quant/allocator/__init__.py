"""allocator — portfolio construction across multiple strategies.

The allocator answers: "given N strategies that each emit target weights,
how much capital does each one get?"

Default pipeline:
1. **HRP (Hierarchical Risk Parity).** Cluster strategies by return
   correlation, then allocate inversely to within-cluster volatility. More
   robust than mean-variance to noisy covariance estimates.
2. **Volatility targeting.** Scale the combined book up or down so realized
   portfolio vol matches `risk.vol_target_annual` from config.
3. **Kelly fraction (optional).** Apply a fractional-Kelly multiplier (often
   0.25–0.5) on top of vol targeting. Pure Kelly is too aggressive in practice
   because expected returns are estimated, not known.

The allocator output is *proposed* weights. The risk overlay
(`quant.risk`) is the final arbiter before orders go out.

Currently available:
- ``hrp_weights`` — Hierarchical Risk Parity (López de Prado 2016).

See docs/specs/hrp.md for the HRP construction details.
"""

from quant.allocator.hrp import hrp_weights

__all__ = ["hrp_weights"]
