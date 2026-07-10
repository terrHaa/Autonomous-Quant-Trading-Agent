# CLAUDE.md — operating rules for this repo

Autonomous quant trading platform. Trades US equities via Alpaca paper.
Operator is in China (CST = UTC+8). Runs unattended under launchd.

## ⚠️ Trade-window freeze rule (READ FIRST)

**launchd runs the code straight from the working tree — there is no build
step.** Any change saved to `src/` takes effect on the *next* scheduled run
of that job. A commit is not required; saving the file is enough. The
process loads its code at start, so what matters is the state of the tree
when the job *starts*.

Therefore: **do not modify the live execution path within one hour before a
scheduled trade.** The live path is `src/quant/execution/alpaca_executor.py`
and `src/quant/agent/daily_runner.py`.

- **Trade fires: Mon–Fri 21:35 CST.** Freeze window: 20:35–21:35 CST (and
  until the run completes, ~21:45).
- Changes to the execution path should land with a **full day of buffer**
  before the next trade — ideally on a weekend (no trades Sat/Sun).
- Before editing those files, run `date` and check you are clear of the
  window. If in doubt, wait for the next day.
- 2026-07-10 incident: a stop-repair fix was committed 21:28, seven minutes
  before the 21:35 trade, and ran live that night. It worked, but it should
  never have landed that close to the window.

Reporting/review jobs (daily report/audit ~06:30–07:00 CST, weekly Sat,
monthly day 2) are read-mostly and lower-risk, but still prefer a buffer.

## Operator hard rules

Live in `src/quant/agent/daily_runner.py` as constants (deliberately NOT in
YAML), and mirrored in `configs/default.yaml`. Do not change these without
explicit operator confirmation:

- `STOP_LOSS_PCT = 0.05`
- `MAX_POSITION_WEIGHT = 0.20` (concentrated-bet policy, not the
  institutional 3–5%)
- `MAX_DRAWDOWN_KILL = 0.15`
- `MAX_SECTOR_WEIGHT = 0.30`
- Long-only. No wash trading. IEX data feed pin.

## Change discipline

- Every incident becomes a **regression test + a guard/audit check** before
  it is considered closed. No failure class should be able to recur silently.
- Changes to the trading path ship with tests. Run the full suite
  (`.venv/bin/python -m pytest tests/ -q`) and `ruff check` before commit.
- Inputs are not trusted: guards exist for stale bars (T3.18), bar
  integrity, and broker position-feed desync (`position_gate`). Add to this
  set rather than assuming a feed is correct.
- Fail-safe direction: on a genuine ambiguity in the live path, prefer the
  no-op (skip a trade) over acting on suspect data.
