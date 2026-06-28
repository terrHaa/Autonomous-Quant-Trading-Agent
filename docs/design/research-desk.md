# Design: Continuous Research Pipeline + Agent Desk

**Status:** APPROVED to build (2026-06-27) — build now, substrate-first,
falsifier-first within the agent desk. Phase gates in §7 govern when each
later phase starts.
**Author:** drafted 2026-06-27 with the operator.
**Supersedes (conceptually):** the fixed monthly-interval deploy cadence.

---

## 1. Problem

Two things are bundled today that shouldn't be:

- **Research** (generate + backtest candidate strategies/factors) runs once
  a month, in the monthly review. There's no reason thinking should be
  monthly — it's cheap and could run continuously.
- **Deployment** (push a new strategy into the live ensemble) is gated to
  the same monthly tick.

The operator's instinct: decouple them — research continuously, deploy when
a candidate is *ready*, not on an arbitrary date.

This is correct **only if** the deployment discipline is preserved. The
naive version — "deploy whenever a backtest looks good" — is a
false-positive harvesting machine: the more you test, the more likely you
deploy something that cleared the bar by luck (the look-elsewhere effect).
So the design's whole job is: **continuous search, paced and
statistically-honest deployment.**

Non-goal: more idea *generators*. Generation is not the bottleneck (one LLM
call yields fifty plausible ideas). The bottleneck is rigorous *rejection*.

---

## 2. Architecture: the assembly line

```
 continuous research            shadow queue                deployment
 ┌──────────────┐   sandbox   ┌──────────────┐  promotion  ┌──────────┐
 │  generators  │──gate+DSR──▶│ paper-shadow │──criteria──▶│  live    │
 │ (background) │             │  (accruing   │   met       │ ensemble │
 └──────────────┘             │  OOS record) │             └──────────┘
        │                     └──────────────┘                  ▲
        ▼                            ▲                           │
 ┌──────────────┐                    │                    risk authority
 │  falsifier   │── kills bad ───────┘                    (human-gated for
 │ (adversary)  │   candidates                             risk-layer changes)
 └──────────────┘
        ▲
        │ every candidate scored against
 ┌─────────────────────────────────────────┐
 │  GLOBAL TRIAL LEDGER  (multiple-testing) │  ← the non-negotiable
 └─────────────────────────────────────────┘
```

**Stages a candidate passes through** (extends today's single-shot path):

1. **Proposed** — a generator emits a strategy/factor + thesis.
2. **Sandbox** — existing gates (`strategy_sandbox`, `_MIN_SHARPE`,
   structural validity, no look-ahead). Backtested over the holdout.
3. **Falsified-or-survives** — the adversary agent attacks it (§4). A
   candidate that survives proceeds; one that's killed is logged with the
   kill reason (feeds the trial ledger either way).
4. **Shadow** — enters paper-shadow with **zero** live allocation, accruing
   an out-of-sample live track record. (Mechanism exists today:
   `EnsembleState.ai_strategy_shadow_until`, shadow handling in
   `compute_ensemble_targets`.)
5. **Promotion** — event-driven, when promotion criteria (§5) are met. Not
   on a calendar.
6. **Live** — added to the ensemble; HRP gives it real weight.

The monthly review does **not** disappear — it becomes the **risk &
allocation review** (the 5-pillar mandate already built): factor
attribution, signal health, regime policy, pipeline reliability. It stops
being the deploy gate; promotion is continuous and event-driven.

---

## 3. The global trial ledger (the non-negotiable)

Continuous + parallel search makes multiple-testing **worse**, not better.
The defense is a single, append-only ledger of *every* hypothesis tested —
across all agents, all time:

- Each entry: candidate id, generating agent, family/domain, backtest
  Sharpe, the DSR inputs, outcome (killed / shadow / promoted / retired),
  timestamp.
- The **Deflated Sharpe Ratio** for any candidate is computed with
  `n_trials` = the *cumulative* count of related trials in the ledger, not 1
  and not reset per run. (We already have `evaluation.dsr`; today it's fed a
  local trial count — this globalizes it.)
- Promotion is **impossible** without a ledger entry and a DSR that clears
  the bar *after* deflation. This is what stops parallelism from harvesting
  noise.

Without this, everything else in this doc is a faster way to lose money.
Build it first.

---

## 4. The agent desk (specialize by function + domain, not clones)

A **small** desk (~5 roles), all feeding ONE validation gate and ONE risk
authority. Headcount is not the edge; process is.

| Role | Job | Why it exists |
|---|---|---|
| **Falsifier / red-team** | Attack every candidate: look-ahead, overfit, regime-dependence, capacity, correlation to existing book. Default verdict = reject. | The scarce, high-value function. **First hire.** |
| **Generator — cross-sectional** | Propose equity factor / cross-sectional strategies. | Domain split prevents convergence on the same overfit. |
| **Generator — time-series/trend** | Propose trend / TS-momentum / breakout / vol strategies. | Different part of the edge space. |
| **Risk / allocation** | Portfolio construction, regime policy, correlation de-gross. | = today's monthly analyst role (already built). |
| **Orchestrator / PM** | Owns the shadow queue, promotion decisions, the deployment-rate budget, and the trial ledger. Human-in-the-loop for risk-layer changes. | Single deployment authority; prevents N agents racing to deploy. |

Principles:
- **Falsifier before generators.** Adding generators without a strong
  adversary strictly worsens the false-positive rate.
- **One gate, one risk authority.** Parallel proposers, centralized
  validation and capital allocation — exactly how a real desk is shaped.
- **Domain-split generators**, not cloned ones, so they explore different
  regions of strategy space.

---

## 5. Promotion criteria (event-driven, replaces the calendar)

A shadow candidate promotes to live the moment **all** hold:

1. **Out-of-sample shadow length** ≥ N trading days (start N≈20; the paper
   track record must be *live*, not backtest).
2. **Shadow performance** consistent with its backtest thesis (no large
   live-vs-backtest divergence — reuses the implementation-shortfall idea).
3. **Deflated Sharpe** clears the institutional floor (≥ ~0.7) using the
   **cumulative** trial count from the ledger.
4. **Diversification** — correlation to the current book below a ceiling
   (a 0.6-Sharpe uncorrelated sleeve beats a 1.0-Sharpe near-clone).
5. **Deployment-rate budget** — not exceeding the rolling cap (e.g. ≤1 new
   strategy live per ~30 days, regardless of how many qualify), so
   attribution stays clean and blast radius stays small.

Anything touching the **risk/allocation layer** (sizing, regime policy,
sector caps) keeps the **human-in-the-loop** gate — it re-risks the whole
book, not just adds a sleeve.

---

## 6. What already exists to build on

- **Shadow mechanism:** `EnsembleState.ai_strategy_shadow_until`,
  `compute_ensemble_targets(shadow_strategies=...)` — 10-day paper period.
- **Registry + stages:** `quant.registry.registry.Registry.promote(...)`.
- **Sandbox gates:** `strategy_sandbox`, `_MIN_SHARPE`.
- **DSR:** `quant.evaluation.dsr` (needs globalizing per §3).
- **The just-built engines:** factor attribution, signal-health, regime
  tools, reliability — these are the falsifier's and risk agent's
  instruments.
- **Regime-policy auto-apply gate:** `quant.agent.regime_gate` — the
  template for "auto-apply behind a backtest+DSR gate."

The gap is: continuous (not monthly) generation, a *multi-candidate*
shadow queue, the *global* trial ledger, and the falsifier/orchestrator
agents.

---

## 7. Phased build + explicit phase gates

Decision (2026-06-27): **build now**, substrate-first. This is a paper
system in its early, high-iteration phase — capital risk is nil, and fast
disciplined iteration compounds. The substrate is *itself* the measurement
backbone that keeps fast iteration from becoming thrash, so it's built
first not as caution but as the enabler.

Each phase has an explicit **gate**: a short list of *checkable*
conditions that must hold before starting the next phase. The gates are
evaluated automatically — see "Readiness reporter" below — so the operator
doesn't have to remember; the monthly review surfaces gate status.

**Phase A — Substrate** *(build now)*
Global trial ledger + DSR globalization + multi-candidate shadow queue +
event-driven promotion criteria (§5) + a backtest A/B harness. No new
agents. Lets the existing analyst's ideas flow continuously, and makes
every *other* structural experiment independently measurable.

→ **Gate A→B (build the falsifier when ALL hold):**
- substrate operational: ledger recording, ≥3 candidates have flowed
  through the shadow queue to a promotion *decision* (promote or reject);
- the A/B harness has scored ≥1 structural change end-to-end;
- evidence the adversary is needed: ≥1 candidate that passed backtest
  later failed in shadow (backtest-good / live-bad) — i.e. naive
  backtest-passing is demonstrably insufficient.

**Phase B — Falsifier agent.** The adversary (§4). Highest marginal value.

→ **Gate B→C (build generators when ALL hold):**
- falsifier operational with a *sane* kill rate (kills some, not all/none
  — neither rubber-stamp nor reject-everything);
- the shadow queue is *starving* (candidate inflow is the bottleneck, not
  validation throughput);
- factor attribution shows a genuine **alpha gap** (book is mostly beta /
  small residual alpha over a meaningful `n_obs`) — i.e. we actually need
  new alpha *sources*, not just better allocation of existing ones.

**Phase C — Domain-split generators.** Two specialized proposers.

→ **Gate C→D (build orchestrator when ALL hold):**
- multiple candidates flowing concurrently;
- manual promotion decisions have become real toil/bottleneck;
- promotion criteria (§5) have been stable for ≥2 cycles (safe to automate).

**Phase D — Orchestrator/PM.** Promotion automation + budget enforcement.

Each phase is independently useful and independently revertible. Some gate
conditions are judgment calls (e.g. "genuine alpha gap"); the readiness
reporter shows the *evidence* and flags "operator call", it does not
pretend to auto-decide the go/no-go.

### Readiness reporter (how we remember we're ready)

Built as part of Phase A. A small function evaluates each open gate's
conditions against live artifacts (ledger, shadow queue, attribution,
reliability scorecard) and emits, e.g.:

```
Research desk — next gate: A→B (build falsifier)
  [x] substrate operational (ledger + queue live)
  [ ] ≥3 candidates reached a promotion decision  (now: 0)
  [x] A/B harness scored ≥1 structural change
  [ ] ≥1 backtest-good / shadow-bad candidate observed  (now: 0)
  → 2/4 met — not yet. (operator call on the last one.)
```

This block is appended to the **monthly review email** (and runnable on
demand). Phase-readiness becomes a number that lands in the inbox every
month, not a thing anyone has to track by hand.

---

## 8. Reality check / open questions

- **Scale match.** This is a paper account, one operator, ~500-name
  universe, 3 strategies. Capital risk is nil, so the build-now decision
  (§7) stands; the discipline is enforced by the gates and the trial
  ledger, not by waiting. The *agents* are still gated on evidence (§7
  gates), but the *substrate* is built now because it's the measurement
  backbone that makes fast iteration honest rather than thrash.
- **Cost.** Each agent is recurring LLM spend + operational surface. Every
  role must earn its place against the falsifier-first principle.
- **Open:** promotion thresholds (N days, DSR floor, correlation ceiling)
  need calibration on live data. Start conservative, loosen with evidence.
- **Open:** does generation stay LLM-proposed code, or move toward a
  parameterized factor search over the taxonomy? The latter is cheaper and
  more auditable but less creative.
- **Open:** capacity/turnover modeling once multiple sleeves compete for the
  same names.
