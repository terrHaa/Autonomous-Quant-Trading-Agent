# quant

A quantitative equity research and trading platform.

The goal of this project is to take a strategy idea from research → backtest →
evaluation → portfolio construction → paper trading → (eventually) live capital,
with **research-grade rigor** at every step. That means:

- No look-ahead leakage in backtests.
- Deflated Sharpe Ratio (DSR) and walk-forward analysis to fight overfitting.
- Hierarchical Risk Parity (HRP) for portfolio construction, with volatility
  targeting and Kelly sizing overlays.
- Risk overlays (drawdown limits, position caps, circuit breakers) sitting
  *outside* the strategy code, so a bad strategy cannot bypass them.
- A strategy registry with explicit promotion gates — nothing trades real money
  until it has earned the right.

## Layout

```
quant/
├── configs/        YAML configs: universe, date ranges, vol targets, etc.
├── docs/specs/     Design specs (one markdown file per concept: DSR, HRP, ...)
├── src/quant/      The Python package itself.
│   ├── data/         Ingestion, on-disk storage, integrity checks.
│   ├── backtest/     Event-driven backtest engine with no-leak guarantees.
│   ├── strategies/   Individual strategy implementations.
│   ├── evaluation/   Metrics, DSR, walk-forward, probabilistic Sharpe.
│   ├── allocator/    HRP, volatility targeting, Kelly fraction.
│   ├── risk/         Overlays: drawdown kill switch, exposure caps, etc.
│   ├── registry/     Strategy registry + promotion gates.
│   ├── execution/    Broker interface (Alpaca).
│   ├── agent/        LLM agent layer (placeholder for now).
│   └── reports/      Markdown report generation for runs/research.
└── tests/          Unit + integration tests.
```

## Getting started

This project is managed with [`uv`](https://docs.astral.sh/uv/), a fast Python
package manager. From the `quant/` directory:

```bash
# 1. Install Python + project dependencies into a local virtual environment.
#    uv reads pyproject.toml and creates `.venv/` automatically.
uv sync

# 2. Copy the env template and fill in your Alpaca keys.
cp .env.example .env

# 3. Run the test suite to confirm everything is wired up.
uv run pytest
```

You don't need to `pip install` anything by hand — `uv sync` handles it.

## Workflow (for the trader-new-to-code)

Think of this repo like a quant fund's internal codebase, scaled down:

1. **Research a strategy** as a notebook or script that calls the backtest
   engine in `src/quant/backtest/`.
2. **Evaluate** it with `src/quant/evaluation/` — Sharpe is not enough; we use
   DSR and walk-forward to make sure the result isn't just lucky.
3. **Register** the strategy in `src/quant/registry/`. The registry tracks every
   variant you've ever tried, so we can correct for multiple-testing bias.
4. **Allocate** capital across multiple registered strategies via
   `src/quant/allocator/` (HRP + vol targeting).
5. **Risk overlay** wraps the allocator output. This is the last line of defense.
6. **Execute** via `src/quant/execution/` — Alpaca paper account first, always.

See `docs/specs/` for the design rationale behind each major piece.
