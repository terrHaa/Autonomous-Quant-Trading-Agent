# Backtest engine — design spec

**Status:** Draft v1
**Date:** 2026-05-26
**Audience:** anyone touching `quant.backtest`, plus future-you trying to remember why we did it this way.

## What this is

A bar-by-bar, **event-driven** backtest engine for systematic equity strategies.

Event-driven means: the engine literally walks forward one trading day at a time. At each step, the strategy receives a snapshot containing only the data that would have been observable at that moment. It returns target weights. The engine generates orders, fills them at the next day's open, and steps forward.

### Why not vectorized?

A "vectorized" backtest (multiply a signal column by a returns column, sum it up) is 100× faster and 100× easier to subtly leak future data into. The most common bug looks like this:

```python
signal = (price.rolling(20).mean() > price.rolling(50).mean()).astype(int)
returns = price.pct_change()
strategy_return = signal * returns          # ← look-ahead by one day, silently
```

The `signal` value at date *T* was computed *with knowledge of T's close*, but `returns[T]` is also based on T's close — so the strategy effectively trades on data it didn't yet have at the moment of the decision. Event-driven cannot make this mistake because the engine doesn't have tomorrow yet.

We're optimizing for **correctness over speed**. A typical S&P 500 daily backtest over 10 years runs in a few seconds; that's fast enough.

## The time loop

Pseudocode for one full backtest run:

```
portfolio = Portfolio(starting_equity)
queued_orders = []

for bar_date in trading_calendar(config.dates.start, config.dates.end):
    bars_today = data.bars_for(bar_date)

    # 1. EXECUTE: fill yesterday's queued orders at today's open.
    for order in queued_orders:
        portfolio.fill(order, fill_price=bars_today[order.symbol]['open'])
    queued_orders = []

    # 2. MARK: revalue positions at today's close.
    portfolio.mark_to_market(bars_today)

    # 3. SIGNAL: hand strategy a snapshot with data through today's close.
    snapshot = Snapshot(data, as_of=bar_date)
    target_weights = strategy.on_bar(snapshot)

    # 4. ORDER: turn target weights into orders for tomorrow's open.
    queued_orders = portfolio.orders_to_reach(target_weights, bars_today)
```

Notes on order:
- Fills happen BEFORE marking — fills update positions, then the new positions get marked.
- Signal happens AFTER marking — strategy sees today's close in equity values.

## The no-leak rule

A strategy must not be able to see any data that wouldn't have been observable on the bar it's deciding for.

We enforce this with a `Snapshot` object passed to `strategy.on_bar`. The snapshot is built from a slice of the cached bars frame, truncated to `bar_date` (inclusive). Accessing `snapshot.bars` returns only rows where `timestamp ≤ bar_date close`. Anything beyond that doesn't exist on the snapshot — not "raises on access," literally not present.

This is stricter than "the strategy promises not to peek." A strategy that tries to peek gets an `IndexError` or `KeyError`, not silently-wrong numbers.

Concretely:

```python
class Snapshot:
    """Data observable as of `as_of` (inclusive of that bar's close)."""

    def __init__(self, full_bars: pd.DataFrame, as_of: date):
        # Pre-slice the frame so the strategy literally cannot reach the future.
        ts = full_bars.index.get_level_values("timestamp")
        self._bars = full_bars[ts.date <= as_of]
        self.as_of = as_of

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars

    def close(self, symbol: str) -> float:
        """Most recent close for symbol on or before as_of."""
        ...
```

We will write a test that fabricates a "cheating strategy" — one that tries to look at `as_of + 1` — and asserts it fails. This test is non-negotiable; it's the guard rail for the whole platform.

## Strategy interface

Strategies are classes implementing:

```python
class Strategy(Protocol):
    name: str   # unique identifier — used by the registry later

    def on_bar(self, snapshot: Snapshot) -> dict[str, float | OrderIntent]:
        """Return desired per-symbol intent.

        Two valid values per symbol:
          - A bare float: target weight (fraction of equity), filled as a
            market-on-open order at the next bar.
          - An OrderIntent: same target weight, plus optional order type
            (limit/stop) and trigger prices.

        Rules:
          - Weights are positive for long, negative for short. Both are
            supported from v1.
          - Symbols not in the returned dict are treated as 'flat in this
            name' — engine will exit existing positions in those symbols
            using a market-on-open order.
          - Weights do not need to sum to 1; the allocator/risk layers
            handle leverage and gross/net caps later. For now the engine
            enforces only |sum(|w|)| <= 1 (no leverage in v1).
        """


@dataclass(frozen=True)
class OrderIntent:
    target_weight: float
    order_type: Literal["market", "limit", "stop"] = "market"
    limit_price: float | None = None    # required if order_type == "limit"
    stop_price: float | None = None     # required if order_type == "stop"
    time_in_force: Literal["DAY", "GTC"] = "DAY"
```

Why target weights, not full order objects?

1. **Composability.** When we add HRP and vol targeting, those layers take strategy weights and produce *adjusted* weights. If strategies emitted only orders, the allocator couldn't intercept cleanly.
2. **Simpler to write.** Strategy authors think in "I want X% in AAPL," not "buy 327 shares."
3. **Decouples sizing from signaling.** The engine turns weights into orders using `equity * weight / signal_close`; the strategy is responsible for the signal only.

The `OrderIntent` escape hatch exists for strategies that genuinely need limit or stop semantics (e.g., a mean-reversion strategy that wants to buy at the prior day's low rather than the next open). Most strategies will just return floats and get market-on-open behavior.

## Fill model

**Base convention: orders submitted at the close of day T are evaluated against the bar of day T+1.** This mirrors what really happens — you decide after today's close, the broker tries to fill tomorrow.

Each order type has its own fill rule against next bar's OHLC:

### Market-on-open (default)

Always fills at next open with bps-based slippage.

```
half_spread_bp   = config.backtest.costs.spread_bps / 2
slippage_bp      = config.backtest.costs.slippage_bps
total_bp         = half_spread_bp + slippage_bp

buy_fill_price   = open * (1 + total_bp / 10_000)
sell_fill_price  = open * (1 - total_bp / 10_000)
```

### Limit

Fills *only if* the limit price was reachable inside the next bar.

| Side | Fills when | Fill price |
|---|---|---|
| Buy limit at `L` | next bar `low <= L` | `L` (we got our price) |
| Sell limit at `L` | next bar `high >= L` | `L` |

If not reached: the order's `time_in_force` decides what's next. `DAY` cancels at next close; `GTC` re-evaluates on each subsequent bar.

No slippage is added for limits — by definition you trade at your stated price. Commission still applies.

### Stop

Triggers when the stop level is crossed, then fills at the stop price.

| Side | Triggers when | Fill price |
|---|---|---|
| Buy stop at `S` | next bar `high >= S` | `S` |
| Sell stop at `S` | next bar `low <= S` | `S` |

**Optimism caveat:** at daily resolution we cannot model **gap-through**. If a stop sits at $95 and tomorrow's bar opens at $90 (overnight gap on bad news), reality would fill us at $90, not $95 — a 5% worse exit. Our engine fills at $95. Documented as a known optimistic bias of the daily-bar engine; if/when we build an intraday engine, this is one of the first things it fixes. To partially compensate, the realistic move is to set `slippage_bps` higher when using stops, or use a market order to exit instead.

### Commission

Commission is **separate** from the price-side costs above (Alpaca is commission-free; some brokers aren't). Deducted from cash on each fill, regardless of order type:

```
commission_dollars = commission_bps / 10_000 * |notional|
```

### Total round-trip cost

For a market-order round trip:
`2 × (half_spread + slippage + commission)` bps
= one full spread + 2× slippage + 2× commission.

For a limit-order round trip you pay only the commissions (in exchange for fill-risk uncertainty). This is part of why limits matter and why we modeled them.

### Why split costs three ways?

Each captures a different real-world cost source:
- `spread_bps` — bid-ask spread, broker-independent, depends on the security.
- `slippage_bps` — impact + adverse selection buffer (open vs realized fill).
- `commission_bps` — per-broker charge, zero for Alpaca.

Keeping them distinct lets us decompose performance attribution by cost source later (e.g., "this strategy's Sharpe is fine but its slippage line is huge — we're rebalancing too often").

## Output: `BacktestResult`

The engine returns one structured object containing everything needed by evaluation, reports, and the registry:

```python
@dataclass(frozen=True)
class BacktestResult:
    config:        Config              # snapshot of inputs (for reproducibility)
    strategy_name: str
    equity_curve:  pd.Series           # daily equity, DatetimeIndex
    positions:     pd.DataFrame        # rows=date, cols=symbol, vals=shares
    weights:       pd.DataFrame        # rows=date, cols=symbol, vals=weight after marking
    orders:        pd.DataFrame        # one row per order; cols: date, symbol, side, target_qty
    fills:         pd.DataFrame        # one row per fill; cols: date, symbol, side, qty, fill_price, costs_paid
    costs:         pd.DataFrame        # per-day cost breakdown: commission, spread, slippage
    metadata:      dict                # n_bars, n_orders, run_time_s, git_sha, etc.
```

Frozen because once a backtest finishes, its result is part of the audit trail — the registry pins each result by hash. Any mutation would invalidate the trail.

## Determinism

Identical inputs (config + data + strategy code) must produce a bit-identical `BacktestResult`. This is what makes the registry's "have we seen this exact variant before?" check work.

Rules:
- No reads of system clock during a run.
- No `os.urandom` or unseeded RNG.
- Any randomness comes from a config-supplied seed.
- Pandas operations on equal inputs already give equal outputs.
- The `metadata` dict has stable key ordering.

## Delisting policy

If a held symbol drops out of the universe mid-backtest (sector reshuffle, merger, bankruptcy), the engine:

1. Detects the disappearance — symbol has no bar for date T but had one prior.
2. Liquidates the position at the **last available close** for that symbol.
3. Logs a `delisting` event into `BacktestResult.fills`.
4. Removes the symbol from `weights`/`positions` going forward.

**Optimistic bias:** "last close" is right for normal removals (S&P swaps in/out, mergers paying at last-print premium) but wrong for bankruptcies — Lehman didn't trade at $3.65 after Sept 15, 2008; holders got pennies. We don't yet have a data source distinguishing the two, so we use last-close as a uniform proxy. Documented; will revisit when corporate-actions data is wired in.

## Non-goals (v1)

Explicitly **not** doing these in the first cut. Each is a real-world thing we'll need eventually; leaving them out keeps v1 small.

- **Intraday.** Daily bars only; minute-bar support is a future engine variant. The known consequence is the stop-fill optimism noted above.
- **Weekly/monthly rebalance.** Config has the field, but v1 supports daily only. Adding the others is ~20 lines once we need them.
- **Short borrow costs.** Modeled as zero. Real shorts pay 0–5%+ annualized.
- **Margin interest.** Same — modeled as zero. The leverage cap in config keeps us under 1.5×, but real margin isn't free.
- **Partial fills.** Every order fills 100% at the modeled price. Real life: liquidity-constrained names may fill partially.
- **Market-impact model.** Slippage is a constant in bps regardless of order size. Reality: big orders move the market.
- **Corporate actions at fill time.** We rely on Alpaca's pre-adjusted prices via the cache. Live trading will need explicit handling.
- **Taxes.** Pre-tax returns only.
- **Cash drag accrual.** v1 ignores risk-free interest on idle cash. Adding it is a 3-line change once we settle the day-count convention.
- **Anything not US equities.** No options, futures, FX, crypto.

Limitations get a comment in the engine source so we don't forget them.

## Design decisions resolved

For the historical record — discussed during spec review, decided as follows:

| Question | Decision | Rationale |
|---|---|---|
| Order types in v1 | Market-on-open + Limit + Stop | Strategies sensitive to slippage can use limits; stops accepted with documented gap-through optimism. |
| Long vs short | Both from v1 | Negative target weights = short. Surfaces shorting bugs early; matches eventual HRP/allocator assumptions. |
| Delisting policy | Liquidate at last available close | Realistic for sector swaps and mergers; optimistic for bankruptcies. Bias documented; revisit with corp-actions data. |

## Worked example: one day in the engine

Setup: Strategy holds AAPL only, target weight 1.0. On Friday, signal says go to 0.5 (cut by half).

```
Equity at Thursday close: $1,000,000
AAPL position:            5,500 shares @ $185.00 close = $1,017,500
                          → weight = 1.0175 (slight drift)
Cash: -$17,500

After strategy.on_bar receives Friday's close snapshot:
  target_weights = {"AAPL": 0.5}

Engine computes target qty:
  Friday close = $186.00
  target_$ = 0.5 * $1,000,000 = $500,000
  target_qty = $500,000 / $186.00 ≈ 2,688 shares
  delta = 2,688 - 5,500 = -2,812 (SELL 2,812)

Queue order: SELL 2,812 AAPL at next open (Monday).

Monday open: $187.50
  total_bp = (2 / 2) + 3 = 4 bp slippage
  sell_fill_price = 187.50 * (1 - 4/10000) = $187.43
  proceeds = 2,812 * 187.43 = $526,853
  commission = 0
  costs_paid = 2,812 * (187.50 - 187.43) = $197

Position becomes 2,688 AAPL; cash becomes -$17,500 + $526,853 = $509,353.
```

After Monday's close, the engine marks to market, computes new equity, and `on_bar` runs again for Monday's decision.
