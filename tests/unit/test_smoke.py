"""Smoke test — confirms the package is installed and importable.

This is the cheapest possible test: it doesn't exercise any logic, it just
proves the project is wired up correctly. If this fails, nothing else will
work, so it's the first thing to run after `uv sync`.
"""

import quant


def test_package_imports() -> None:
    """The top-level package should import and expose a version string."""
    assert quant.__version__ == "0.1.0"


def test_all_submodules_import() -> None:
    """Every declared submodule should import without error.

    Catches the classic "I renamed a file but forgot to update the import"
    failure mode early, before it hides inside a backtest run.
    """
    from quant import (  # noqa: F401  (imports are the test)
        agent,
        allocator,
        backtest,
        data,
        evaluation,
        execution,
        registry,
        reports,
        risk,
        strategies,
    )
