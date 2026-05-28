"""Tests for the AI-strategy sandbox: AST safety, protocol check, timeout.

These tests cover the three protection layers in
quant.agent.strategy_sandbox that gate AI-generated strategy code
before it touches the ensemble. The full backtest gates (Sharpe/DSR/DD)
are exercised in integration tests; here we focus on the cheap layers
and the timeout mechanism.
"""

from __future__ import annotations

import pytest

from quant.agent.strategy_sandbox import (
    _check_imports,
    _enforce_timeout,
    _load_strategy_class,
    _SandboxTimeout,
)


# ---------------------------------------------------------------------------
# Layer 1: AST import scan
# ---------------------------------------------------------------------------


def test_clean_imports_pass() -> None:
    """Allowed imports (numpy, pandas, quant.backtest.types) return no errors."""
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from quant.backtest.types import Snapshot\n"
    )
    assert _check_imports(code) == []


def test_forbidden_top_level_imports_blocked() -> None:
    """os, socket, subprocess, etc. must be rejected before exec."""
    for forbidden in ["os", "socket", "subprocess", "requests", "urllib"]:
        errors = _check_imports(f"import {forbidden}")
        assert any(forbidden in e for e in errors), f"{forbidden} was not blocked"


def test_forbidden_from_imports_blocked() -> None:
    """`from os import ...` style must also be rejected."""
    errors = _check_imports("from os import path")
    assert errors  # at least one error
    assert any("os" in e for e in errors)


def test_dangerous_builtins_blocked() -> None:
    """open(), eval(), exec(), __import__() in the AST must be rejected."""
    for snippet in [
        "x = open('/etc/passwd')",
        "x = eval('1+1')",
        "x = exec('print(1)')",
        "x = __import__('os')",
        "x = compile('1', '<x>', 'exec')",
    ]:
        errors = _check_imports(snippet)
        assert errors, f"Snippet was not flagged: {snippet}"


def test_syntax_error_returned_as_import_error() -> None:
    """Malformed code returns a SyntaxError string rather than raising."""
    errors = _check_imports("class Foo(:\n    pass")
    assert errors
    assert "SyntaxError" in errors[0]


# ---------------------------------------------------------------------------
# Layer 2: load + protocol check
# ---------------------------------------------------------------------------


def test_valid_strategy_loads_successfully() -> None:
    """A class with .name + .on_bar instantiates cleanly."""
    code = (
        "class MyStrat:\n"
        "    def __init__(self, symbols):\n"
        "        self.name = 'my_strat'\n"
        "        self._symbols = symbols\n"
        "    def on_bar(self, snapshot):\n"
        "        return {}\n"
    )
    instance = _load_strategy_class(code, "MyStrat", ["AAPL"])
    assert instance is not None
    assert instance.name == "my_strat"


def test_missing_class_returns_none() -> None:
    """class_name not defined in the code → None, no crash."""
    code = "class Foo:\n    pass\n"
    instance = _load_strategy_class(code, "DoesNotExist", ["AAPL"])
    assert instance is None


def test_strategy_without_on_bar_returns_none() -> None:
    """Class has .name but no on_bar method → rejected."""
    code = (
        "class Bad:\n"
        "    def __init__(self, symbols):\n"
        "        self.name = 'bad'\n"
    )
    instance = _load_strategy_class(code, "Bad", ["AAPL"])
    assert instance is None


def test_strategy_without_name_returns_none() -> None:
    """Class has on_bar but no .name attribute → rejected."""
    code = (
        "class Bad:\n"
        "    def __init__(self, symbols): pass\n"
        "    def on_bar(self, snapshot): return {}\n"
    )
    instance = _load_strategy_class(code, "Bad", ["AAPL"])
    assert instance is None


def test_constructor_failure_returns_none() -> None:
    """If the constructor raises, we get None — no exception propagates."""
    code = (
        "class Boom:\n"
        "    def __init__(self, symbols):\n"
        "        raise RuntimeError('cannot init')\n"
        "    def on_bar(self, s): return {}\n"
        "    name = 'boom'\n"
    )
    instance = _load_strategy_class(code, "Boom", ["AAPL"])
    assert instance is None


# ---------------------------------------------------------------------------
# Layer 3: wall-clock timeout
# ---------------------------------------------------------------------------


def test_timeout_raises_on_long_running_block() -> None:
    """_enforce_timeout aborts a block that exceeds the limit."""
    import time

    with pytest.raises(_SandboxTimeout):
        with _enforce_timeout(1):
            time.sleep(3)  # exceeds 1s limit


def test_timeout_does_not_fire_on_fast_block() -> None:
    """A block that finishes in time runs without exception."""
    with _enforce_timeout(5):
        x = sum(range(1000))   # trivially fast
    assert x == 499500


def test_timeout_cleans_up_after_normal_exit() -> None:
    """After the context manager exits normally, no alarm is left armed."""
    import signal
    with _enforce_timeout(2):
        pass
    # If alarm wasn't cleaned up, this sleep would be interrupted.
    # We assert by checking pending alarm is 0 (works on macOS/Linux).
    pending = signal.alarm(0)
    assert pending == 0
