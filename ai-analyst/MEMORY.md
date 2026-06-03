# MEMORY.md — Analyst's Append-Only Log

> Every monthly review appends one entry to this file. Loaded into the
> AI analyst's prompt so it can see its own history, recognise patterns,
> and avoid repeating rejected ideas without conscious reasoning.
>
> **Format**: newest entries at the BOTTOM. The analyst reads top-to-bottom,
> so the most recent context arrives last (recency bias works in our favor here).

---

## Pattern Notes (manually edited or summarised periodically)

_This section is for the operator (you) to record patterns that survive
multiple months — e.g., "RSI-based strategies tend to fail in low-dispersion
regimes" — so the analyst sees the high-signal takeaways even after old
monthly entries get truncated._

- (none yet)

---

## Monthly Entries


### 2026-06-01 — Monthly Review

**Grid-search step**: Evaluated 8 candidates. No improvement found: no candidate beat current on BOTH Sharpe and max drawdown

**Outcome**: `no_proposal`

**Sandbox result**: (no proposal made)

**Analysis**:

This is the inaugural monthly review with only 3 trading days of data (2026-05-27 through 2026-05-29), making statistical inference premature. The +1.04% return over this stub period is encouraging but statistically meaningless—2 daily returns cannot establish any reliable pattern. The annualized Sharpe of 136.5 and 100% win rate are artifacts of the tiny sample and will normalize rapidly. With n=2 daily returns, the 95% confidence interval on the true Sharpe spans roughly ±50, rendering the point estimate uninformative.

Triangulating the three data sources reveals consistency rather than disagreement this week: (1) The weekly narrative correctly identifies LLY (+5.8%) and ORCL (+5.5%) as top contributors with energy names (XOM -1.9%, CAT -2.3%, LIN -2.5%) as drags. (2) The raw daily table confirms steady gains on both Thursday (+0.56%) and Friday (+0.47%) with stable top-3 positions (XOM, COST, PM/WMT/NVDA rotating). (3) The monthly statistical view's top-10 gainers/losers align perfectly with the weekly narrative. No disagreement to surface—the sources are mutually consistent, but this is trivially true with only 3 days of data. The day-of-week breakdown (Thu: +0.56%, Fri: +0.47%) shows no calendar effect, but with n=1 per day this is noise.

The weekly analyst flagged one item 'WORTH ESCALATING TO MONTHLY REVIEW': the mean reversion strategy's negative Sharpe (-0.50 over lookback). However, this is premature to act on. The mean reversion strategy is DESIGNED to underperform in trending regimes (see STRATEGY_LIBRARY.md §2 regime behavior), and the current environment appears to be mildly trending. Its 3.8% HRP weight appropriately limits exposure. The strategy has not failed—it's behaving as documented. I will not propose removing or modifying it based on 3 days of data in an unfavorable regime. If negative performance persists for 4+ weeks with evidence that the regime has shifted to range-bound (where it should excel), that would warrant escalation.

Regarding ensemble gaps: per EDGE_TAXONOMY.md, the current three strategies cover cross-sectional momentum (partial), time-series momentum (partial via SMA crossover), and short-horizon reversal. The documented gaps buildable from OHLCV are: (1) volatility regime filter/strategy, (2) low-volatility anomaly/BAB, (3) volume confirmation, (4) breakout strategies, (5) defensive/low-beta. Given only 3 days of live data, I cannot yet identify which gap would most improve the ensemble's risk-adjusted returns. Proposing a new strategy now would be pure speculation without the backtest signal that comes from observing strategy behavior across multiple regimes. I will wait for at least 20 trading days (one full calendar month) before proposing additions.

For trail_pct: the weekly analyst correctly noted XOM appears in top-3 positions despite being a top-5 loser. However, examining the trail_high table shows XOM's trail_high is 146.96—the stop would fire at 139.61 (5% below). Without intraday price data, I cannot determine how close XOM came to this stop. There is no evidence of give-back (positions running up then retracing significantly before organic exit) or whipsaw (same symbol appearing as sell→re-buy in consecutive days) in this 3-day window. The position persistence metric of 94.6% indicates stable holdings with minimal forced exits. No trail_pct change is warranted.

**Proposal**:

_No strategy proposed this month._

---

### 2026-06-01 — Monthly Review

**Grid-search step**: Evaluated 8 candidates. No improvement found: no candidate beat current on BOTH Sharpe and max drawdown

**Outcome**: `no_proposal`

**Sandbox result**: (no proposal made)

**Analysis**:

This is the second monthly review, now with 3 trading days of data (2026-05-27 through 2026-05-29). The +1.04% return and 100% win rate (2/2 daily returns) remain statistically meaningless—with n=2 daily returns, we cannot draw any inference about strategy efficacy. The annualized Sharpe of 136.5 is a mathematical artifact that will collapse toward realistic values as the sample grows. I estimate we need at least 20 trading days before proposing any new strategy with confidence.

Triangulating the three data sources shows complete consistency this month: (1) The weekly narrative identifies LLY (+5.8%) and ORCL (+5.5%) as top contributors with energy/industrials (LIN -2.5%, CAT -2.3%, XOM -1.9%) as drags. (2) The raw daily table confirms steady gains Thursday (+0.56%) and Friday (+0.47%) with stable positions (XOM, COST, PM/WMT/NVDA rotating through top-3). (3) The monthly statistical view's top-10 gainers/losers align exactly with the weekly narrative. No disagreement exists—this is trivially true with only 3 days of data where the weekly IS the monthly. The day-of-week breakdown (Thu: +0.56%, Fri: +0.47%) shows no calendar effect, but with n=1 observation per day, this is pure noise.

The weekly analyst flagged one item 'WORTH ESCALATING TO MONTHLY REVIEW': the mean reversion strategy's negative Sharpe (-0.50 over lookback). I explicitly addressed this in the prior monthly review and my assessment is unchanged. The mean reversion strategy is DESIGNED to underperform in trending regimes (per STRATEGY_LIBRARY.md §2 regime behavior). Its 3.8% HRP weight appropriately limits exposure. The strategy is behaving as documented—this is not a failure, it's regime-conditional performance. I will not propose modifications based on 3 days of data in a regime where underperformance is expected. If negative performance persists for 4+ weeks AND the market shifts to range-bound conditions (where mean reversion should excel) without improvement, that would warrant action.

Regarding trail_pct: examining the trail_high table shows synthetic test values (SYM0-SYM9), not live positions—this appears to be test data. The actual positions (XOM, COST, PM, etc.) don't appear in trail_high, suggesting either the trail tracking isn't populating correctly or this is a display artifact. I cannot assess give-back or whipsaw patterns without accurate trail_high data for the actual held symbols. Position persistence of 94.6% indicates stable holdings, and I see no evidence of same-symbol sell→re-buy patterns in consecutive daily runs. No trail_pct change is warranted given insufficient evidence.

Per EDGE_TAXONOMY.md, the genuine gaps buildable from OHLCV are: (1) volatility regime filter, (2) low-volatility anomaly/BAB, (3) volume confirmation, (4) breakout strategies. With only 3 days of live trading data, I cannot identify which gap would most improve risk-adjusted returns. A 20-day minimum is needed to observe strategy behavior across varied market conditions. Proposing now would be speculation. I return no strategy proposal this month.

**Proposal**:

_No strategy proposed this month._

---
<!-- ENTRIES_INSERTION_POINT — new entries appended above this comment by monthly_review.py -->

_End of MEMORY.md_
