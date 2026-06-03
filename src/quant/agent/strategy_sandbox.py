"""strategy_sandbox.py — safe validation and backtesting of AI-generated strategies.

Three layers of protection run in sequence before a proposed strategy
is accepted into the live ensemble:

Layer 1 — AST safety scan
    Parse the code as an AST and reject any import that isn't
    numpy / pandas / quant.backtest.types. Also block dangerous built-ins
    (open, eval, exec, __import__). This prevents the AI from writing code
    that reads files, hits the network, or exfiltrates data.

Layer 2 — Protocol check + smoke test
    Exec the code in an isolated namespace, instantiate the class, and call
    ``on_bar()`` with a real (tiny) snapshot. Confirms the class follows the
    Strategy Protocol before running an expensive backtest.

Layer 3 — Backtest gates
    Run a full 2-year backtest. Accept only if ALL three pass:
      • Sharpe  > _MIN_SHARPE  (absolute floor — not relative to existing strategies)
      • Max drawdown ≤ _MAX_DRAWDOWN_ABS  (absolute cap on capital risk)
      • DSR     ≥ _DSR_THRESHOLD  (multi-testing-corrected statistical significance)

Persistence helpers
    ``save_generated_strategy`` writes accepted strategy code + a sidecar
    JSON metadata file to ``src/quant/strategies/generated/``.
    ``load_generated_strategy`` reads them back for ``build_strategies``.
"""

from __future__ import annotations

import ast
import json
import logging
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config import Config

logger = logging.getLogger(__name__)


# Wall-clock timeout for the full backtest in Layer 3. An infinite loop or
# O(n²) implementation in AI-generated code could otherwise hang the
# monthly review for hours. 60 seconds is generous (2-year backtests for
# the human-written strategies finish in 2-10 seconds).
_BACKTEST_TIMEOUT_SECONDS: int = 60


class _SandboxTimeout(Exception):
    """Raised when an AI-generated strategy's backtest exceeds the timeout."""


@contextmanager
def _enforce_timeout(seconds: int):
    """Raise _SandboxTimeout if the block runs longer than `seconds`.

    Uses SIGALRM, which works on macOS + Linux (the platforms the agent
    actually runs on). Falls through silently on Windows.
    """
    try:
        # Save the previous handler so nested timeouts behave sanely.
        prev = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(seconds)
    except (AttributeError, ValueError):
        # SIGALRM unavailable (Windows, or running inside a non-main thread).
        # Skip enforcement rather than crash — the gates still apply.
        prev = None
    try:
        yield
    finally:
        try:
            signal.alarm(0)  # cancel any pending alarm
            if prev is not None:
                signal.signal(signal.SIGALRM, prev)
        except (AttributeError, ValueError):
            pass


def _raise_timeout(_signum, _frame) -> None:  # noqa: ANN001
    raise _SandboxTimeout(
        f"AI-generated strategy backtest exceeded {_BACKTEST_TIMEOUT_SECONDS}s — "
        "likely infinite loop or O(n²) implementation. Rejected."
    )

# ---- Gate thresholds (absolute, not relative to current strategies) --------

# A new strategy must clear these bars to join the ensemble. They are
# intentionally conservative: the AI can propose as many strategies as it
# wants, but only genuinely positive-expectation, statistically valid ones
# make it through.

# Institutional-grade gates. The earlier values (Sharpe 0.30, DD 35%)
# were retail-research floors that would let mediocre strategies into a
# real-capital ensemble; tightened to thresholds a professional shop
# would accept.
#
#   _MIN_SHARPE:          0.30 → 0.70 — retail floor is ~0.7;
#                         institutional shops typically want 1.0+. We
#                         compromise at 0.7 so the analyst can still
#                         propose strategies periodically, but
#                         mediocrity no longer ships to real capital.
#   _MAX_DRAWDOWN_ABS:    0.35 → 0.20 — 35% drawdown on a single
#                         strategy is a career-ending blowup; no real
#                         shop accepts it. 20% matches institutional norms.
#   _DSR_THRESHOLD:       0.95 unchanged — multi-testing correction is
#                         already at the right level.
_MIN_SHARPE: float = 0.70          # annualised, after costs
_MAX_DRAWDOWN_ABS: float = 0.20    # 20% max peak-to-trough, absolute
_DSR_THRESHOLD: float = 0.95       # 95th-percentile confidence after multi-testing

# ---- Allowed top-level modules in AI-generated code -----------------------

_ALLOWED_MODULES: frozenset[str] = frozenset({"numpy", "pandas", "quant", "__future__", "typing"})

# ---- Directory for persisted generated strategies -------------------------

_GENERATED_DIR: Path = (
    Path(__file__).resolve().parent.parent / "strategies" / "generated"
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    passed_gates: bool
    sharpe: float           # annualised Sharpe from full backtest
    max_drawdown: float     # absolute (positive) max drawdown, e.g. 0.18 = 18%
    dsr: float              # Deflated Sharpe Ratio (0–1)
    rejection_reason: str   # empty string when passed_gates is True


# ---------------------------------------------------------------------------
# Layer 1 — AST safety scan
# ---------------------------------------------------------------------------


def _check_imports(code: str) -> list[str]:
    """Return a list of forbidden-import error strings. Empty = clean code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        # Block any import whose top-level module isn't on the allow-list.
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    errors.append(f"Forbidden import: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _ALLOWED_MODULES:
                errors.append(f"Forbidden 'from' import: '{node.module}'")
        # Block calls to dangerous built-ins.
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name in {"open", "eval", "exec", "__import__", "compile",
                        "breakpoint", "input", "getattr", "setattr", "delattr"}:
                errors.append(f"Forbidden built-in call: {name}()")
    return errors


# ---------------------------------------------------------------------------
# Layer 2 — Protocol check + smoke test
# ---------------------------------------------------------------------------


def _load_strategy_class(
    code: str,
    class_name: str,
    symbols: list[str],
) -> Any | None:
    """Exec code in an isolated namespace and return an instantiated strategy.

    Returns ``None`` (and logs a warning) on any error so the caller can
    record a clean rejection rather than raising.
    """
    # Give the exec'd code its own __builtins__ so standard built-ins work
    # but the namespace doesn't share state with the host process.
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(code, "<ai_strategy>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        logger.warning("strategy_sandbox: exec failed: %s", e)
        return None

    cls = namespace.get(class_name)
    if cls is None:
        logger.warning(
            "strategy_sandbox: class '%s' not found in exec'd code. "
            "Classes found: %s",
            class_name,
            [k for k, v in namespace.items() if isinstance(v, type)],
        )
        return None

    try:
        instance = cls(symbols)
    except Exception as e:
        logger.warning("strategy_sandbox: could not instantiate %s: %s", class_name, e)
        return None

    if not isinstance(getattr(instance, "name", None), str):
        logger.warning("strategy_sandbox: %s.name is not a str", class_name)
        return None
    if not callable(getattr(instance, "on_bar", None)):
        logger.warning("strategy_sandbox: %s missing callable on_bar()", class_name)
        return None

    return instance


def _smoke_test(instance: Any, bars: pd.DataFrame) -> str | None:
    """Call on_bar with a tiny real snapshot. Returns an error string or None."""
    from quant.backtest.types import Snapshot  # local import avoids circular dep

    try:
        # Use the last 20 rows — enough for most strategies to warm up.
        recent = bars.iloc[-20:] if len(bars) >= 20 else bars
        as_of = recent.index.get_level_values("timestamp").max()
        snap = Snapshot(as_of=as_of, bars=recent)
        result = instance.on_bar(snap)
        if not isinstance(result, dict):
            return f"on_bar returned {type(result).__name__}, expected dict"
        for sym, w in result.items():
            if not isinstance(w, (int, float)):
                return (
                    f"on_bar value for '{sym}' is {type(w).__name__}, expected float"
                )
            if float(w) < 0:
                return f"on_bar returned negative weight for '{sym}': {w}"
    except Exception as e:
        return f"on_bar raised during smoke test: {type(e).__name__}: {e}"
    return None  # success


# ---------------------------------------------------------------------------
# Layer 3 — Full backtest gates
# ---------------------------------------------------------------------------


def validate_and_test_strategy(
    *,
    code: str,
    class_name: str,
    strategy_name: str,
    universe: list[str],
    bars: pd.DataFrame,
    config: Config,
    dsr_threshold: float = _DSR_THRESHOLD,
    min_sharpe: float = _MIN_SHARPE,
    max_drawdown_abs: float = _MAX_DRAWDOWN_ABS,
) -> SandboxResult:
    """Full validation pipeline: AST → load → smoke → backtest → gates.

    This is the single function monthly_review.py calls. It returns a
    ``SandboxResult`` describing what happened; it never raises (all errors
    become rejection reasons so the review email always sends).
    """
    from quant.backtest.engine import run_backtest
    from quant.evaluation.dsr import dsr_for
    from quant.evaluation.metrics import metrics_for
    from quant.registry.registry import Registry

    # ---- Layer 1: AST safety ------------------------------------------------
    import_errors = _check_imports(code)
    if import_errors:
        return SandboxResult(
            passed_gates=False,
            sharpe=0.0,
            max_drawdown=1.0,
            dsr=0.0,
            rejection_reason=f"Unsafe imports: {'; '.join(import_errors)}",
        )

    # ---- Layer 2a: Load class -----------------------------------------------
    instance = _load_strategy_class(code, class_name, universe)
    if instance is None:
        return SandboxResult(
            passed_gates=False,
            sharpe=0.0,
            max_drawdown=1.0,
            dsr=0.0,
            rejection_reason=(
                f"Could not load class '{class_name}' from generated code. "
                "Check class name and constructor signature."
            ),
        )

    # ---- Layer 2b: Smoke test -----------------------------------------------
    smoke_error = _smoke_test(instance, bars)
    if smoke_error:
        return SandboxResult(
            passed_gates=False,
            sharpe=0.0,
            max_drawdown=1.0,
            dsr=0.0,
            rejection_reason=f"Smoke test failed: {smoke_error}",
        )

    # ---- Layer 3: Full backtest (with wall-clock timeout) -------------------
    try:
        with _enforce_timeout(_BACKTEST_TIMEOUT_SECONDS):
            backtest_result = run_backtest(config=config, strategy=instance, bars=bars)
    except _SandboxTimeout as e:
        return SandboxResult(
            passed_gates=False,
            sharpe=0.0,
            max_drawdown=1.0,
            dsr=0.0,
            rejection_reason=str(e),
        )
    except Exception as e:
        return SandboxResult(
            passed_gates=False,
            sharpe=0.0,
            max_drawdown=1.0,
            dsr=0.0,
            rejection_reason=f"Backtest raised {type(e).__name__}: {e}",
        )

    m = metrics_for(backtest_result)
    sharpe = float(m.sharpe)
    # m.max_drawdown is negative (e.g. -0.18); we want the absolute value.
    max_dd_abs = abs(float(m.max_drawdown))

    # Register in the DSR registry so n_trials stays accurate.
    registry_path = Path("data/agent/registry.db")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(path=registry_path)
    registry.record(
        backtest_result,
        parameters={"name": strategy_name, "class_name": class_name},
        stage="research",
        notes="AI-generated strategy candidate (strategy_sandbox)",
    )
    n_trials = registry.n_trials()
    trial_sharpes = registry.trial_sharpes()
    var_sr = (
        float(pd.Series(trial_sharpes).var(ddof=1))
        if len(trial_sharpes) > 1
        else 0.10   # conservative default when very few trials exist
    )

    try:
        dsr = float(dsr_for(
            backtest_result,
            n_trials=n_trials,
            var_sr_trials_annual=var_sr,
        ))
    except Exception as e:
        logger.warning("strategy_sandbox: DSR computation failed: %s", e)
        dsr = 0.0

    # ---- Gate checks ---------------------------------------------------------
    failures: list[str] = []
    if sharpe < min_sharpe:
        failures.append(
            f"Sharpe {sharpe:.3f} < minimum {min_sharpe:.2f}"
        )
    if max_dd_abs > max_drawdown_abs:
        failures.append(
            f"Max drawdown {max_dd_abs:.1%} > cap {max_drawdown_abs:.0%}"
        )
    if dsr < dsr_threshold:
        failures.append(
            f"DSR {dsr:.3f} < threshold {dsr_threshold:.2f} "
            f"(n_trials={n_trials}; multi-testing significance too low)"
        )

    if failures:
        return SandboxResult(
            passed_gates=False,
            sharpe=sharpe,
            max_drawdown=max_dd_abs,
            dsr=dsr,
            rejection_reason="Gate failures: " + "; ".join(failures),
        )

    return SandboxResult(
        passed_gates=True,
        sharpe=sharpe,
        max_drawdown=max_dd_abs,
        dsr=dsr,
        rejection_reason="",
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_generated_strategy(
    *,
    name: str,
    class_name: str,
    code: str,
) -> Path:
    """Write validated strategy code + metadata sidecar to the generated dir.

    Creates the directory and ``__init__.py`` if they don't exist.
    If a file for ``name`` already exists (a previous month's version of the
    same strategy), it is **overwritten** — we only keep the latest approved
    version, since it passed the gates more recently.

    Returns the path to the saved ``.py`` file.
    """
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    init = _GENERATED_DIR / "__init__.py"
    if not init.exists():
        init.write_text(
            "# Auto-generated strategy modules — managed by strategy_sandbox.py.\n"
        )

    # Write the Python source.
    py_path = _GENERATED_DIR / f"{name}.py"
    py_path.write_text(code)

    # Write sidecar metadata so load_generated_strategy knows the class name.
    meta: dict[str, str] = {"name": name, "class_name": class_name}
    meta_path = _GENERATED_DIR / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("strategy_sandbox: saved generated strategy → %s", py_path)
    return py_path


def load_generated_strategy(
    name: str,
    symbols: list[str],
) -> Any | None:
    """Load a previously saved generated strategy from disk.

    Reads the sidecar ``.json`` for the class name, then loads the ``.py``.
    Returns ``None`` (and logs a warning) if either file is missing or the
    class can't be instantiated.
    """
    meta_path = _GENERATED_DIR / f"{name}.json"
    py_path = _GENERATED_DIR / f"{name}.py"

    if not meta_path.exists() or not py_path.exists():
        logger.warning(
            "load_generated_strategy: missing files for '%s' in %s",
            name, _GENERATED_DIR,
        )
        return None

    meta = json.loads(meta_path.read_text())
    class_name = meta.get("class_name", "")
    if not class_name:
        logger.warning("load_generated_strategy: no class_name in metadata for '%s'", name)
        return None

    code = py_path.read_text()
    instance = _load_strategy_class(code, class_name, symbols)
    if instance is None:
        logger.warning("load_generated_strategy: failed to load '%s' from disk", name)
    return instance
