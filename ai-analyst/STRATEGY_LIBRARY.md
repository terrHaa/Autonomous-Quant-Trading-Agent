# STRATEGY_LIBRARY.md — Canonical Catalog of Active Strategies

> The authoritative record of every strategy currently in the ensemble.
> Loaded into the AI analyst's prompt each month so it knows what edges
> are already covered.
>
> **Append-only.** When a new AI strategy passes the sandbox gates, the
> monthly review appends its full specification here. Manually edited
> only when a strategy is decommissioned (rare).

---

## Risk-Management Components (system-level, not in build_strategies)

These components apply uniformly to every position, regardless of which
strategy generated the entry signal. They are tunable via fields on
`EnsembleState` and you (the analyst) may propose adjustments to them
via `proposed_state_changes` in your response.

### Per-entry initial stop loss — `STOP_LOSS_PCT = 0.05` (operator hard rule)

Every fresh entry has an atomic stop attached at
`entry_signal_price * (1 - 0.05)`. **This is the operator's hard floor
and cannot be changed by the analyst.** It exists to cap the loss on
any single new position at 5%. Quoted here so you know it's there.

### Trailing stop ratchet — `EnsembleState.trail_high` + `trail_pct` (TUNABLE)

Once a position is open, the system tracks its highest signal price
since entry (`trail_high[sym]`) and computes the daily stop as
`trail_high[sym] * (1 - trail_pct)`. The stop only ratchets UP, never
down — locking in gains on winners. On a flat or down day, the stop
stays at the prior high; if a retrace pushes the trailing stop above
the current signal price, the next rebalance refuses the re-entry and
the position stays flat (the trailing stop has effectively fired).

**Tunable**: `state.trail_pct` ∈ (0, 0.05]. Default 0.05 = behaves like
a static 5% stop. A tighter value (e.g. 0.03) locks in more gains but
exits earlier on retracements; a value near 0.05 lets winners breathe
more. Cannot exceed `STOP_LOSS_PCT` — a trail wider than the entry
stop would violate the operator's per-trade loss floor.

**When to propose a change**:
- If multiple recent months show winners running >20% then giving back
  >10% before the next rebalance flatted them organically, propose a
  tighter trail (e.g. 0.03).
- If many positions are getting stopped out by intraday noise then
  re-bought next day at similar prices (visible as same symbol in
  consecutive runs with sells-rebuys), propose a wider trail.
- If the trail behavior is fine and no clear signal exists, propose
  nothing — the static 5% default is sensible.

### Per-position max weight — `MAX_POSITION_WEIGHT = 0.20` (operator hard rule)

No single position may exceed 20% of equity. Hard floor; cannot be
changed by the analyst. Quoted so you know to size proposals so they
don't hit the cap in normal conditions.

---

## Active Strategies (loaded by ensemble.build_strategies)

### 1. `sma_crossover_50_200` — base strategy

**Type**: Time-series trend-following
**Edge thesis**: Sustained price trends are autocorrelated over multi-month
horizons due to slow information diffusion and behavioral underreaction
(Hong & Stein 1999). The 50/200 cross is a discretisation of this
persistence.

**Mathematical specification**:
```
fast_SMA_i,t = mean(close_i, t-50:t)
slow_SMA_i,t = mean(close_i, t-200:t)
signal_i,t = 1 if fast_SMA_i,t > slow_SMA_i,t else 0
weight_i,t = signal_i,t / sum_j(signal_j,t)   # equal weight across longs
```

**Regime behavior**:
- Trending: ✅ profits from sustained moves
- Range-bound: ❌ whipsaw losses
- High-vol: mixed — depends on whether vol is trending or mean-reverting
- Low-vol: positive carry, low turnover

**Blind spots**: no volatility awareness; no cross-sectional ranking;
late entries (200-day lookback).

---

### 2. `mean_reversion_5_200bp` — base strategy

**Type**: Time-series short-horizon mean reversion
**Edge thesis**: Single-name overreactions to short-term news create
mispricings that revert within a week (Lehmann 1990, Jegadeesh 1990).
Liquidity providers earn a premium for absorbing this flow.

**Mathematical specification**:
```
SMA5_i,t = mean(close_i, t-5:t)
threshold = 0.02   # 2%
signal_i,t = 1 if (close_i,t - SMA5_i,t) / SMA5_i,t < -threshold else 0
weight_i,t = signal_i,t / sum_j(signal_j,t)
```

**Regime behavior**:
- Trending: ❌ falling knife problem (buys things that keep falling)
- Range-bound: ✅ classic mean-reversion edge
- High-vol: ⚠️ more signals but also more drift continuation
- Low-vol: low signal count

**Blind spots**: doesn't distinguish fundamental drops from noise; no
volume confirmation; no regime filter.

---

### 3. `xsec_momo_60_5_10` — base strategy

**Type**: Cross-sectional momentum (Jegadeesh-Titman, skip-month)
**Edge thesis**: Stocks that outperformed peers over the last 12 months
(approximated by 60 days here) continue to outperform over the next 1-12
months (Jegadeesh-Titman 1993). The skip avoids the short-term reversal
documented by Lehmann.

**Mathematical specification**:
```
r_i,t = (close_i,t-5 - close_i,t-65) / close_i,t-65   # 60-day return ending 5 days ago
rank_i,t = rank(r_i,t) across i in universe
selected_t = top-10 by rank
weight_i,t = 0.1 if i in selected_t else 0   # equal-weight top decile
```

**Regime behavior**:
- Trending: ✅ rides relative winners
- Range-bound: ⚠️ reversal risk
- High-vol: ❌ momentum crashes (2009, 2020) — known failure mode
- Low-vol: ✅ steady performance

**Blind spots**: equal weights ignore risk; no momentum strength weighting;
exposed to known "momentum crash" tail risk.

---

## Accepted AI Strategies

_None yet — accepted AI proposals will be appended below by the monthly review._

<!-- AI_STRATEGIES_INSERTION_POINT -->

---

## Decommissioned Strategies

_None yet — manually move a strategy here if you remove it from `ai_strategy_names`._

<!-- DECOMMISSIONED_INSERTION_POINT -->

---

_End of STRATEGY_LIBRARY.md_
