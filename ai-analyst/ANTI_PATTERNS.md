# ANTI_PATTERNS.md — Failure Modes to Avoid

> A curated catalog of recurring mistakes in quantitative strategy research,
> loaded into your prompt each month. You (the AI analyst) know most of these
> from your training; the purpose of this file is to make them top-of-mind
> when drafting proposals, not buried in long-tail knowledge.
>
> **Rule of use**: before submitting any proposal, do a final pass and verify
> none of these apply. If one might — surface it explicitly in your reasoning
> and explain why your proposal is robust against it.

---

## Category 1 — Data-mining sins (the classics)

### 1.1 Parameter sweeps without theory
**The trap**: "I tested lookbacks of 20, 40, 60, 80, 100 days and 60 was best." This is over-fitting masquerading as research.

**The rule**: every parameter must be justified theoretically OR by an external source (a paper that found the same value, an established practitioner convention). "I tried five values and one won" is a rejection ground per ANALYST.md §4.

**Concrete fix**: cite the source. E.g. "60-day lookback per Jegadeesh-Titman's 6-month original (~126 trading days), shortened here to balance signal freshness with sample stability." Now the parameter has a defense.

### 1.2 Backtest selection bias
**The trap**: you mentally try N variants, only present the best one, and the apparent Sharpe is now biased upward. Even if you didn't explicitly grid-search, your iterative refinement IS a search.

**The rule**: declare your candidate space up front. If you considered 3 alternatives before this one, mention them and say why this one beat them — show your work.

### 1.3 Survivorship bias
**The trap**: the top-50 universe is RE-COMPUTED — it doesn't carry historical delisted names. A strategy that depends on this composition implicitly benefits from survivorship.

**The rule**: prefer signals that depend on a name's RELATIVE position (rank, z-score) or its OWN time-series properties, not its absolute identity. A strategy that says "if X is in the universe and shows pattern P, do Y" is much more robust than "if X is in the universe, do Y."

### 1.4 Look-ahead bias
**The trap**: using data that wouldn't have been available at decision time. Subtle versions: normalizing by the full-sample mean (uses future data); computing a "rolling" stat that includes the current bar.

**The rule**: `snapshot.bars` is correctly truncated. But anything YOU compute from it must use `.shift(1)` if you're predicting day t's return from day t's features. When in doubt, lag by one extra day.

### 1.5 Cherry-picked backtest window
**The trap**: a strategy that looks great over 2018-2024 but would have died in 2008. Or one tuned specifically to the post-COVID regime.

**The rule**: the sandbox runs a 2-year backtest. Note in your reasoning whether the 2-year window is REGIME-DIVERSE (multiple vol regimes, both trending and mean-reverting periods). If not, lower your conviction.

---

## Category 2 — Statistical fallacies

### 2.1 Treating Sharpe like a constant
**The trap**: "this strategy has a Sharpe of 1.2." Sharpe is a sample statistic with a confidence interval. A point estimate of 1.2 from a 2-year backtest with daily returns has a 95% CI of roughly ±0.6 — meaning the true Sharpe might be anywhere from 0.6 to 1.8.

**The rule**: report a RANGE, not a number. The sandbox already enforces this through DSR (Deflated Sharpe), but your reasoning should also acknowledge confidence intervals.

### 2.2 Ignoring correlation with the existing ensemble
**The trap**: proposing a strategy with a high Sharpe but a correlation of 0.85 with `sma_crossover_50_200`. The marginal addition to the ensemble is small even if its standalone Sharpe is excellent.

**The rule**: the correlation hypothesis (ANALYST.md §2.4) requires |ρ| < 0.5 with each existing strategy. If you predict higher correlation, propose a different edge. A new strategy must DIVERSIFY, not just amplify.

### 2.3 Multiple-testing problem (this is what DSR is FOR)
**The trap**: the more strategies you try, the more likely you are to find one with apparent alpha by chance. The 2-year sandbox backtest's Sharpe is BIASED upward by the implicit selection.

**The rule**: this is exactly why the sandbox enforces a DSR (Deflated Sharpe Ratio) gate at 0.95, accounting for the number of historical trials. Don't argue around it. If DSR fails, the edge probably isn't real.

### 2.4 Confusing in-sample fit with out-of-sample performance
**The trap**: a strategy that fits beautifully on 2022-2024 is not necessarily going to work in 2025. Markets evolve.

**The rule**: prefer simple, theoretically-motivated strategies over complex parameter-rich ones. Complexity ≠ alpha. The simpler your edge, the less you'll lose to regime changes.

### 2.5 "Trading the noise"
**The trap**: a tiny daily signal that, after transaction costs, has zero expected return. The backtest may not model the friction realistically.

**The rule**: signals that require sub-daily rebalancing OR that depend on small price differences (e.g. micro-mean-reversion within 0.1%) are suspicious. The current system rebalances daily with close-to-close prices — slippage is real on top of the modeled commissions.

---

## Category 3 — Implementation traps

### 3.1 Strategies that don't scale
**The trap**: works on a $1M backtest but degrades on $100M because the trades move the market.

**The rule for THIS portfolio**: $100k paper, top-N mega-caps with daily turnover. Slippage is low. But if you propose a strategy that requires aggressive intraday execution OR concentration in one name beyond the 20% cap, flag it.

### 3.2 Strategies that depend on perfect data
**The trap**: a signal that breaks when one symbol is missing a bar. Production data is messy.

**The rule**: defensively code for `KeyError` on missing symbols, short histories, NaN values. ANALYST.md §3 already says this; here we reinforce: failure-mode robustness matters more than peak Sharpe.

### 3.3 Strategies that assume cost-free execution
**The trap**: paper trading doesn't model commissions or slippage. A strategy that produces 5-bp daily alpha in backtest might net out to zero after costs in live trading.

**The rule**: signals should produce 50+ bp of expected daily alpha to leave a margin of safety after costs. Anything tighter is noise.

### 3.4 Strategies that require borrowing or shorting
**The trap**: this system is structurally long-only. Proposing a long/short strategy means proposing infrastructure changes the operator may not want.

**The rule**: stay long-only. If the inefficiency you're chasing only works long/short (e.g. classic BAB), propose only the long leg ("long low-beta") and note the limitation.

---

## Category 4 — Behavioral traps for the analyst itself

### 4.1 Proposing complexity to look impressive
**The trap**: a 5-parameter strategy with feature engineering and conditional logic feels more "research-y" than a simple z-score. It's not. Complexity adds overfitting risk.

**The rule**: a strategy with 2-3 parameters that captures a real inefficiency is BETTER than one with 7 parameters that captures the same inefficiency more "thoroughly." Occam's razor applies double in quant.

### 4.2 Repeating rejected ideas without addressing the rejection
**The trap**: month 1 you propose strategy X with parameter P; it fails. Month 2 you propose strategy X with parameter P' (similar). Either propose a genuinely different edge or change the math meaningfully and explain why this version is different.

**The rule**: read MEMORY.md every month. If you're tempted to re-propose something rejected, write down what's CHANGED about the world or your understanding that would make it work now.

### 4.3 Confirmation bias from successful months
**The trap**: a strategy worked great last month → you propose more of its family. Maybe the strategy worked because of the regime, not the edge.

**The rule**: a single month is not statistically meaningful at the strategy level. Don't propose anything based on "last month's results" alone; require multi-month evidence or theory.

### 4.4 Anchoring to existing parameters
**The trap**: `xsec_momo_60_5_10` has lookback=60. You think "I'll propose lookback=80 to differentiate." Why 80? Because it's near 60. That's anchoring, not research.

**The rule**: if a parameter choice has a real reason (e.g. matches a research paper, has economic meaning like "1 quarter = 63 trading days"), choose accordingly. Don't tweak existing parameters by small amounts just to appear different.

### 4.5 Proposing to look productive when there's nothing to add
**The trap**: ANALYST.md §7 says it's acceptable to return null. But the API call cost feels wasted if you don't propose something. Wrong instinct.

**The rule**: a null month with good analysis is a productive month. The cost ($0.50) is paid for the ANALYSIS, not the proposal. Many months should be null.

---

## Category 5 — Famous dead anomalies

These appeared in early academic literature and showed strong backtests at the time. Most have decayed or disappeared. Don't propose them unless you have specific evidence of present-day persistence:

| Anomaly | Source | Status |
|---|---|---|
| January effect | Rozeff & Kinney (1976) | Gone since ~1990 |
| Day-of-week (Monday) | French (1980) | Gone since ~1990 |
| Holiday effect | Lakonishok & Smidt (1988) | Marginal, possibly noise |
| Turn-of-the-month | Ariel (1987) | Heavily decayed |
| Halloween indicator ("sell in May") | Bouman & Jacobsen (2002) | Statistically marginal, transaction costs eat alpha |
| Weather / sentiment | Hirshleifer & Shumway (2003) | Sample-specific, doesn't replicate well |
| Lunar cycle | Yuan, Zheng & Zhu (2006) | Almost certainly noise after MTB correction |
| Net-net Graham (P/B < tangible book) | Graham (1934) | Universe too small in modern mega-caps |
| Pre-FOMC drift | Lucca & Moench (2015) | Possibly real but tactically irrelevant for daily-rebalance equity |

If you ever feel pulled to propose one of these, ask: *what evidence in
the LAST 5 YEARS suggests this still works?* If you can't cite specific
evidence, don't propose it.

---

## Category 6 — Real-world traps unique to this stack

### 6.1 The close-and-reopen execution cost
The executor closes-and-reopens every position every day to refresh stops. This is correct for invariants but adds ~52 small orders/day. Strategies that turn over the entire portfolio every day add zero on top of this; strategies with longer holds get NO commission benefit because the close-and-reopen happens anyway. **Implication**: in this stack, low-turnover and high-turnover strategies have roughly the same friction. Don't propose a strategy with a "low turnover" advantage — it doesn't exist here.

### 6.2 The 5% per-trade stop floor
Every position has a hard 5% stop attached at entry. This is an operator
hard rule and the AI analyst CANNOT change it. **Implication**: strategies
that need a wider stop (e.g. long-vol breakout that needs 10% room) will
get stopped out prematurely. Either propose a tighter-signal version OR
explicitly acknowledge this will happen and explain why the strategy is
still worth it.

### 6.3 The HRP refit dynamic
The weekly HRP refit re-weights strategies by risk. A strategy that's
correct but high-volatility will get DOWN-WEIGHTED, even if its
risk-adjusted returns are good. **Implication**: when proposing a new
strategy, anticipate that HRP will allocate to it proportional to its
INVERSE VOLATILITY. A low-vol strategy with mediocre Sharpe will get
more weight than a high-vol strategy with the same Sharpe.

### 6.4 The dust threshold
`compute_ensemble_targets` drops positions below 0.5% weight. **Implication**:
a strategy that produces many tiny positions (e.g. equal-weight top-100)
will see most of them dust-filtered out. Concentrate signals into fewer,
larger positions.

### 6.5 The shadow period
New AI strategies trade in shadow mode for 10 days — zero allocation,
just record what they WOULD have traded. **Implication**: don't expect
your strategy to influence P&L for ~2 weeks post-acceptance. Plan your
proposal's evaluation window accordingly.

---

_End of ANTI_PATTERNS.md_
