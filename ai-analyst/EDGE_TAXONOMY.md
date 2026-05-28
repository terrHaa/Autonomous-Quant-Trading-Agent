# EDGE_TAXONOMY.md — A Map of the Quant Strategy Space

> A structured atlas of factor categories and inefficiency types you (the AI
> analyst) can mine for new strategy proposals. Loaded into your prompt
> each month so you start from a coverage map instead of a blank page.
>
> **Purpose**: when you ask yourself *"what's the gap in the current ensemble?"*,
> this is the answer key. Cross-reference active strategies (in STRATEGY_LIBRARY.md)
> against the categories here to find genuine gaps — not duplicates dressed up
> in new clothing.
>
> **Format**: each category has (1) what inefficiency it exploits, (2) when
> it works, (3) what universe-properties it needs, and (4) the ensemble's
> current coverage of it.

---

## The big-picture organization

Quant edges fall into five families. Most "new" strategies are rearrangements
within these families; few are genuinely outside them:

1. **Risk premia** — compensation for bearing risk no one else wants
2. **Behavioral** — exploiting persistent biases in how humans process info
3. **Time-series patterns** — autocorrelation in a single asset's price/return
4. **Microstructure** — compensation for liquidity provision or info asymmetry
5. **Structural / quality** — companies' intrinsic properties priced inefficiently

A single strategy can blend families (e.g. *quality-momentum* = quality
filter + momentum signal). Combinations often diversify better than
single-family strategies.

---

## Family 1 — Risk Premia (compensation for bearing risk)

### 1.1 Value (HML)
- **Inefficiency**: cheap stocks (low P/B, P/E, P/CF) outperform expensive ones over multi-year horizons because investors over-pay for growth narratives.
- **Canonical reference**: Fama & French (1992, 1993)
- **Universe fit**: ✅ works in cross-sections with dispersion of valuation multiples. Mega-caps have less dispersion → weaker effect. Better in mid/small caps.
- **Signal proxies**: book-to-market, earnings yield, free-cash-flow yield, EV/EBITDA
- **Failure modes**: secular shifts (post-2010 value drawdown), value traps (companies cheap for fundamental reasons)
- **Current ensemble coverage**: ❌ none

### 1.2 Size (SMB)
- **Inefficiency**: small-caps earn a premium for illiquidity and limited analyst coverage.
- **Canonical reference**: Banz (1981); Fama-French (1993)
- **Universe fit**: ⚠️ largely decayed in mega-caps (no size dispersion). Don't propose size-tilt strategies in a top-50 universe — it's mathematically near-zero.
- **Current ensemble coverage**: ❌ none, and probably shouldn't be added unless universe expands.

### 1.3 Profitability (RMW) and Investment (CMA)
- **Inefficiency**: high-profitability firms with conservative investment policies outperform.
- **Canonical reference**: Fama-French (2015) 5-factor model; Novy-Marx (2013)
- **Universe fit**: ✅ works in any universe with quarterly fundamentals data. Subtle in mega-caps but present.
- **Signal proxies**: gross profitability (GP/A), ROE, asset growth (lower = better)
- **Limitation in this stack**: would require fundamentals data — not currently in the daily-bars pipeline.
- **Current ensemble coverage**: ❌ none

### 1.4 Carry
- **Inefficiency**: instruments with higher yield outperform on average; market doesn't fully compensate for expected appreciation of the lower-carry side.
- **Universe fit**: classic in FX and bond markets. In equities: dividend yield is the rough analog. Mega-caps have a real spread of yields.
- **Current ensemble coverage**: ❌ none

---

## Family 2 — Behavioral (exploiting bias)

### 2.1 Cross-sectional momentum (12-1 month)
- **Inefficiency**: stocks that outperformed peers over the last 12 months (skipping the most recent month) continue outperforming for 1–12 more months. Driven by anchoring and slow information diffusion.
- **Canonical reference**: Jegadeesh-Titman (1993); Carhart (1997)
- **Universe fit**: ✅ works in any liquid cross-section. Mega-caps included.
- **Failure modes**: momentum crashes (post-crisis rebounds in 2009, 2020) — known fat left tail.
- **Current ensemble coverage**: ✅ `xsec_momo_60_5_10` (note: 60-day lookback is shorter than canonical 12-1; that's a TUNABLE choice).

### 2.2 Time-series momentum (TSMOM)
- **Inefficiency**: a single asset's own past return predicts its own future return over 1–12 months. Behavioral underreaction + trend persistence.
- **Canonical reference**: Moskowitz, Ooi & Pedersen (2012)
- **Universe fit**: ✅ universal — works on individual stocks AND on the market index. Different lookbacks suit different horizons.
- **Difference from xsec momo**: TSMOM is binary per asset (long/flat); xsec is relative ranking.
- **Current ensemble coverage**: 🟡 PARTIAL — `sma_crossover_50_200` is a discretised TSMOM (50/200 day crossover). A continuous-signal version, or a multi-horizon TSMOM, would be a new edge.

### 2.3 Short-horizon reversal
- **Inefficiency**: a 1–5 day overreaction reverts. Liquidity providers absorb the imbalance and earn a premium for it.
- **Canonical reference**: Lehmann (1990); Jegadeesh (1990)
- **Universe fit**: ✅ universal. Stronger in less-liquid names but exists in mega-caps too.
- **Current ensemble coverage**: ✅ `mean_reversion_5_200bp` covers this. ⚠️ 5-day lookback + 2% threshold are CHOICES — different parameters (e.g. 3-day, looser threshold, OR 1-day with very loose threshold + volume confirmation) would be a different strategy.

### 2.4 Long-term reversal (3–5 year)
- **Inefficiency**: stocks that have done very poorly over 3–5 years tend to revert.
- **Canonical reference**: De Bondt & Thaler (1985)
- **Universe fit**: ⚠️ decayed since publication; effect is small and noisy in modern markets.
- **Current ensemble coverage**: ❌ none, but probably not worth pursuing.

### 2.5 Post-earnings announcement drift (PEAD)
- **Inefficiency**: stocks that beat earnings continue drifting up for 30–60 days; stocks that miss continue drifting down. Slow analyst revision.
- **Canonical reference**: Bernard & Thomas (1989); Chordia et al. (2009)
- **Universe fit**: ✅ works in any universe with earnings dates. Stronger in less-analyst-covered names.
- **Limitation in this stack**: would need earnings dates — not currently in the daily-bars pipeline.
- **Current ensemble coverage**: ❌ none

### 2.6 Disposition effect (behavioral selling reluctance)
- **Inefficiency**: investors hold losers too long and sell winners too soon. Creates predictable supply/demand imbalance around past entry prices.
- **Universe fit**: ✅ universal but hard to operationalize without trade-level data.
- **Current ensemble coverage**: ❌ none; hard to capture with daily OHLCV alone.

---

## Family 3 — Time-Series Patterns

### 3.1 Volatility regime
- **Inefficiency**: volatility is autocorrelated (high vol begets high vol). Trading on the vol state itself — not just the price — captures a real signal.
- **Signal proxies**: rolling realized vol; Parkinson estimator; Garman-Klass
- **Application**: as a STANDALONE strategy (long-vol or short-vol exposure via SPY proxies) OR as a REGIME FILTER (turn off momentum in high-vol regimes — known to fail there).
- **Current ensemble coverage**: ❌ no vol-aware strategy or regime filter. **THIS IS A REAL GAP.**

### 3.2 Volatility carry / risk-parity
- **Inefficiency**: low-vol stocks earn higher RISK-ADJUSTED returns than high-vol stocks (the "low-volatility anomaly").
- **Canonical reference**: Baker, Bradley & Wurgler (2011); Frazzini & Pedersen (2014) "Betting Against Beta"
- **Universe fit**: ✅ universal. Mega-cap universe has measurable spread of betas/vols.
- **Application**: weight portfolio by 1/vol rather than equal-weight; OR explicitly long low-beta short high-beta.
- **Current ensemble coverage**: ❌ none

### 3.3 Breakout strategies
- **Inefficiency**: prices break out of consolidation ranges and continue. Mix of behavioral (anchoring at round numbers) + microstructure (stop-loss cascades).
- **Universe fit**: ✅ universal but signal-to-noise varies.
- **Current ensemble coverage**: ❌ none (SMA crossover is related but not the same — breakout uses price level, crossover uses moving average).

---

## Family 4 — Microstructure

### 4.1 Volume-confirmed signals
- **Inefficiency**: price moves on high volume are more likely to persist; price moves on low volume revert.
- **Application**: filter ANY entry signal by requiring volume > N-day average.
- **Universe fit**: ✅ universal but noisier in extremely liquid names (mega-caps trade big absolute volumes daily).
- **Current ensemble coverage**: ❌ no volume confirmation on any existing signal. **A SIMPLE BUT REAL ADDITION.**

### 4.2 Order-flow / kyle's lambda
- **Inefficiency**: aggregate order flow predicts short-term price direction.
- **Universe fit**: requires intraday or trade-level data. Not in this stack.
- **Current ensemble coverage**: ❌ N/A — out of scope without intraday data.

### 4.3 Bid-ask spread / illiquidity premium
- **Inefficiency**: less-liquid names earn a premium for illiquidity bearing.
- **Universe fit**: ❌ mega-caps are all liquid; no dispersion. Move to small-caps if pursuing.

---

## Family 5 — Structural / Quality

### 5.1 Quality (Asness-Frazzini-Pedersen)
- **Inefficiency**: high-quality firms (profitable, growing, safe, well-managed) outperform low-quality firms RISK-ADJUSTED. Investors don't sufficiently price the safety dividend.
- **Canonical reference**: Asness, Frazzini & Pedersen (2019)
- **Universe fit**: ✅ works in any universe with fundamentals. Mega-caps have wide quality dispersion.
- **Limitation in this stack**: needs fundamentals data — not in daily-bars pipeline.
- **Current ensemble coverage**: ❌ none

### 5.2 Defensive / low-beta
- **Inefficiency**: low-beta stocks earn higher risk-adjusted returns (the BAB effect). Captured via rolling-window beta calculation against the universe mean return.
- **Canonical reference**: Frazzini & Pedersen (2014)
- **Universe fit**: ✅ works in any universe with beta dispersion.
- **Current ensemble coverage**: ❌ none. **CAN BE BUILT FROM PURE OHLCV.**

### 5.3 Idiosyncratic-volatility filter
- **Inefficiency**: stocks with high idiosyncratic vol underperform on a risk-adjusted basis.
- **Application**: filter out high-idio-vol names from any cross-section ranking.
- **Current ensemble coverage**: ❌ none

---

## Combination patterns (multi-family edges)

The best modern strategies often combine factors:

- **Quality × momentum** (Asness): use a quality filter to gate momentum signals — avoid junk rallies.
- **Value × momentum** (AQR): combine into a single rank — long names that are BOTH cheap AND showing relative strength.
- **Momentum × low-vol**: dampen the momentum tail risk by tilting toward low-vol momentum winners.
- **Time-series filter on cross-sectional signal**: only trade cross-sectional momentum when TSMOM of the market is positive — avoids momentum crashes.

A combination strategy with each component drawing from a DIFFERENT family is more likely to provide genuine diversification than two strategies from the same family with different parameters.

---

## Coverage Map — What's Already Active vs. Gaps

| Family | Sub-category | Active? | Notes |
|---|---|---|---|
| Risk premia | Value | ❌ | Needs fundamentals data |
| Risk premia | Size | ❌ | Decayed in mega-caps |
| Risk premia | Profitability/Investment | ❌ | Needs fundamentals data |
| Risk premia | Carry (yield) | ❌ | Could build from dividends |
| Behavioral | Cross-sectional momentum (12-1) | 🟡 | `xsec_momo_60_5_10` covers a shorter horizon |
| Behavioral | Time-series momentum | 🟡 | `sma_crossover_50_200` is a discretised version |
| Behavioral | Short-horizon reversal | ✅ | `mean_reversion_5_200bp` |
| Behavioral | Long-term reversal | ❌ | Largely decayed |
| Behavioral | PEAD | ❌ | Needs earnings dates |
| Behavioral | Disposition effect | ❌ | Needs trade data |
| Time series | Vol regime / filter | ❌ | **Real gap** — buildable from OHLCV |
| Time series | Low-vol anomaly / BAB | ❌ | **Real gap** — buildable from OHLCV |
| Time series | Breakout | ❌ | Buildable from OHLCV |
| Microstructure | Volume confirmation | ❌ | **Easy + real gap** — buildable from OHLCV |
| Microstructure | Order flow | ❌ | Out of scope |
| Microstructure | Illiquidity | ❌ | Not in mega-caps |
| Quality | Quality factor | ❌ | Needs fundamentals |
| Quality | Defensive / BAB | ❌ | **Gap** — buildable from OHLCV |
| Quality | Idio-vol filter | ❌ | Buildable from OHLCV |

**Boxes marked with ✅ or 🟡 are filled.** Boxes marked ❌ are gaps; the
ones flagged in **bold** are buildable from the daily OHLCV pipeline
currently in place (no fundamentals or intraday data needed) and are
the natural candidates for proposals.

---

## Decay watch — anomalies that worked once but don't now

These are well-publicised in the literature but have largely disappeared
since publication. DON'T propose strategies based on them without
explicit evidence of present-day persistence:

- January effect (Rozeff & Kinney 1976) — gone since the 1990s
- Day-of-week effects (Monday/Friday) — gone since the 1990s
- Calendar/seasonality effects — mostly noise after multiple-testing correction
- Net-net Graham value (P/B < tangible book) — universe too small to be meaningful in mega-caps
- Dogs of the Dow — barely beats buy-and-hold after costs
- Simple P/E rankings without quality controls — value-trap risk has grown

If you find yourself proposing one of these, ask: *what's different
about the data NOW that would make it work, when it's failed for 30+
years?* If you can't answer crisply, don't propose it.

---

_End of EDGE_TAXONOMY.md_
