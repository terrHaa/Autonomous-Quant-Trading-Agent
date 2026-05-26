"""data — market data ingestion, storage, and integrity.

Responsibilities:
- Fetch bars (daily/intraday), corporate actions, and reference data from
  upstream sources (initially Alpaca; pluggable later).
- Persist to local cache (Parquet) for fast, reproducible reads.
- Enforce point-in-time correctness: a query for date D must only return
  information that was actually known on D. This is the single most common
  source of look-ahead bias in equity research.
- Universe construction with historical membership (no survivorship bias).

Nothing in this module should depend on `quant.backtest` or `quant.strategies`.
Data is upstream of everything else.
"""
