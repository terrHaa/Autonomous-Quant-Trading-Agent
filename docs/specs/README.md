# Design specs

Each major concept in the platform gets its own markdown spec here. A spec
should answer: *what is this, why this design, what are the tradeoffs, what
are the alternatives we rejected*.

Planned specs (write as we build):

- `dsr.md` — Deflated Sharpe Ratio: math, implementation, how we feed it the
  true trial count from the registry.
- `hrp.md` — Hierarchical Risk Parity allocator: clustering, recursive
  bisection, why it beats mean-variance for our use case.
- `backtest-engine.md` — No-leak guarantees, fill model, cost model.
- `walk-forward.md` — Fold construction, anchored vs rolling, how it interacts
  with strategy parameter selection.
- `registry-and-promotion.md` — Stage gates, what evidence is required for
  each promotion.
- `risk-overlay.md` — The hard-limit philosophy and why these checks don't
  live in strategies.

Specs are reference material; code is the source of truth. When they disagree,
update one to match the other (and pick which deliberately).
