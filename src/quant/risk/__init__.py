"""risk — risk overlays sitting between allocator and execution.

The risk module is intentionally *outside* the strategy and allocator code.
Its job is to enforce hard limits regardless of what the strategy or
allocator wants. Think of it as the firm's risk desk: strategies propose,
the risk desk disposes.

Enforced limits (from config):
- Gross leverage cap.
- Net exposure cap.
- Per-name position cap.
- Drawdown kill switch — flatten everything if equity peak-to-trough
  exceeds threshold; require manual re-enable.
- Circuit breakers — e.g. halt trading if intraday loss exceeds X%, or if
  market-wide volatility spikes beyond a threshold.

Critically, the risk overlay is *the* place these checks live. A strategy that
quietly internalizes a "soft" version of the limit is a bug, not a feature:
duplicating the check muddies accountability and creates drift.
"""
