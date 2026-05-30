"""Tests for the AI analyst's parser — JSON shape handling, parsing, defaults.

We do NOT exercise the live Anthropic API here (that would cost money and
need a key). Instead we test the response-parsing logic directly by
constructing the raw text the API would return.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quant.agent.ai_analyst import (
    AIAnalyst,
    AnalysisReport,
    StateChangeProposal,
    StrategyProposal,
    WeeklyAnalysisReport,
    _build_system_prompt,
    _build_weekly_system_prompt,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def test_parse_json_response_plain_json() -> None:
    raw = '{"analysis": "ok", "proposed_strategy": null}'
    data = _parse_json_response(raw)
    assert data["analysis"] == "ok"
    assert data["proposed_strategy"] is None


def test_parse_json_response_strips_fenced_markdown() -> None:
    """Some models wrap JSON in ```json fences — parser should still recover."""
    raw = '```json\n{"analysis": "ok"}\n```'
    data = _parse_json_response(raw)
    assert data["analysis"] == "ok"


# ---------------------------------------------------------------------------
# State-change proposal parsing — the new pathway
# ---------------------------------------------------------------------------


def _fake_analyst(raw_json: str) -> AIAnalyst:
    """Construct an AIAnalyst whose API client returns the given raw text.

    Bypasses the constructor's API-key check by directly setting attributes.
    """
    a = object.__new__(AIAnalyst)  # don't run __init__
    a._model = "test-model"

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    content=[SimpleNamespace(text=raw_json)],
                )

    a._client = _FakeClient()
    return a


def test_analyze_returns_state_change_when_present() -> None:
    """Response with proposed_state_changes → StateChangeProposal in report."""
    raw = json.dumps({
        "analysis": "trailing stops are too loose; tighten.",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "trail_pct": 0.03,
            "reasoning": "AVGO ran +30% then retraced -15%; -3% trail would have saved 6%.",
        },
    })
    analyst = _fake_analyst(raw)
    report = analyst.analyze(daily_runs=[], current_state={})
    assert report.proposed_strategy is None
    assert report.proposed_state_changes is not None
    assert report.proposed_state_changes.trail_pct == 0.03
    assert "AVGO" in report.proposed_state_changes.reasoning


def test_analyze_handles_missing_state_change_field() -> None:
    """Old-format response (no proposed_state_changes) → None, no crash."""
    raw = json.dumps({
        "analysis": "nothing this month",
        "proposed_strategy": None,
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is None


def test_analyze_handles_null_state_change() -> None:
    """Explicit null → None (same as missing field)."""
    raw = json.dumps({
        "analysis": "nothing this month",
        "proposed_strategy": None,
        "proposed_state_changes": None,
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is None


def test_analyze_handles_state_change_with_null_trail_pct() -> None:
    """proposed_state_changes present but trail_pct null → trail_pct = None."""
    raw = json.dumps({
        "analysis": "...",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "trail_pct": None,
            "reasoning": "no change",
        },
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is not None
    assert report.proposed_state_changes.trail_pct is None


def test_analyze_gracefully_handles_unparseable_trail_pct() -> None:
    """If trail_pct isn't a number, the field is dropped (logged warning, no raise)."""
    raw = json.dumps({
        "analysis": "...",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "trail_pct": "not-a-number",
            "reasoning": "...",
        },
    })
    # Should NOT raise — bad field is logged and the change is discarded.
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is None


def test_analyze_can_return_both_strategy_and_state_change() -> None:
    """A single response may include both a new strategy AND a tuning change."""
    raw = json.dumps({
        "analysis": "...",
        "proposed_strategy": {
            "name": "test_strat",
            "class_name": "TestStrat",
            "reasoning": "...",
            "code": "class TestStrat:\n    name='test_strat'\n    def __init__(self, syms): pass\n    def on_bar(self, s): return {}",
        },
        "proposed_state_changes": {
            "trail_pct": 0.04,
            "reasoning": "loosen slightly given the new strategy's profile",
        },
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert isinstance(report.proposed_strategy, StrategyProposal)
    assert isinstance(report.proposed_state_changes, StateChangeProposal)
    assert report.proposed_state_changes.trail_pct == 0.04


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_state_change_proposal_defaults() -> None:
    sc = StateChangeProposal()
    assert sc.trail_pct is None
    assert sc.reasoning == ""


def test_analysis_report_state_change_optional() -> None:
    """AnalysisReport's new field defaults to None for backwards-compat."""
    r = AnalysisReport(analysis="x", proposed_strategy=None)
    assert r.proposed_state_changes is None


# ---------------------------------------------------------------------------
# System-prompt assembly — verifies all 5 reference files are loaded
# ---------------------------------------------------------------------------


def test_system_prompt_loads_all_five_reference_files() -> None:
    """The analyst's brain = ANALYST.md + STRATEGY_LIBRARY.md + EDGE_TAXONOMY.md
    + ANTI_PATTERNS.md + MEMORY.md + the response shape. If any file is missing
    or its header isn't in the prompt, the analyst is operating with reduced
    context and proposals will be lower quality."""
    prompt = _build_system_prompt()
    # Headers from each section (the `# === ... ===` lines we wrap them with).
    assert "ANALYST.md (your constitution)" in prompt
    assert "STRATEGY_LIBRARY.md" in prompt
    assert "EDGE_TAXONOMY.md" in prompt
    assert "ANTI_PATTERNS.md" in prompt
    assert "MEMORY.md" in prompt
    assert "RESPONSE PROTOCOL" in prompt


def test_system_prompt_includes_taxonomy_signal_terms() -> None:
    """Spot-check that EDGE_TAXONOMY's content actually arrived in the prompt
    (catches the case where the file is renamed or empty)."""
    prompt = _build_system_prompt()
    # Distinctive phrases from EDGE_TAXONOMY.md
    assert "Family 1" in prompt           # the 5-family organisation
    assert "Coverage Map" in prompt        # the gaps table
    assert "Decay watch" in prompt         # the dead-anomalies section


def test_system_prompt_includes_anti_pattern_signal_terms() -> None:
    """Spot-check that ANTI_PATTERNS's content actually arrived in the prompt."""
    prompt = _build_system_prompt()
    # Distinctive phrases from ANTI_PATTERNS.md
    assert "Parameter sweeps without theory" in prompt
    assert "Famous dead anomalies" in prompt


# ---------------------------------------------------------------------------
# Weekly analyst — narrower prompt, narrative-only response
# ---------------------------------------------------------------------------


def test_weekly_system_prompt_loads_weekly_role_not_monthly() -> None:
    """Weekly prompt should NOT include EDGE_TAXONOMY or ANTI_PATTERNS
    (saves ~7k tokens; those files are about proposing new edges and the
    weekly analyst is explicitly forbidden from doing that)."""
    prompt = _build_weekly_system_prompt()
    assert "WEEKLY_ANALYST.md" in prompt
    assert "STRATEGY_LIBRARY.md" in prompt
    assert "MEMORY.md" in prompt
    # These should NOT be present in weekly mode:
    assert "EDGE_TAXONOMY" not in prompt
    assert "ANTI_PATTERNS" not in prompt
    # Distinctive content from WEEKLY_ANALYST.md
    assert "performance analyst" in prompt.lower()
    assert "Forbidden behaviors" in prompt or "forbidden" in prompt.lower()


def test_weekly_system_prompt_demands_markdown_not_json() -> None:
    """The weekly response shape says raw markdown, not JSON — operators
    embed the narrative directly in email."""
    prompt = _build_weekly_system_prompt()
    assert "markdown ONLY" in prompt or "markdown only" in prompt.lower()
    assert "no JSON" in prompt or "no json" in prompt.lower()


def test_analyze_weekly_returns_report_with_narrative() -> None:
    """analyze_weekly takes daily runs + metrics and returns markdown narrative."""
    raw_markdown = (
        "## Headline\n\n"
        "The week returned +1.2% with 18/25 positions positive...\n\n"
        "## Attribution\n\n"
        "NVDA contributed +0.4%, AVGO +0.3%..."
    )
    analyst = _fake_analyst(raw_markdown)
    report = analyst.analyze_weekly(
        daily_runs=[{"date": "2026-05-29", "execution_report": {}}],
        weekly_metrics={"total_return_pct": 1.2},
        hrp_diagnostic={},
    )
    assert isinstance(report, WeeklyAnalysisReport)
    assert "+1.2%" in report.narrative
    assert "NVDA" in report.narrative


def test_analyze_weekly_strips_markdown_fences() -> None:
    """If the model wraps the response in ```markdown … ``` fences, we strip them."""
    raw = "```markdown\n## Headline\n\nText here.\n```"
    analyst = _fake_analyst(raw)
    report = analyst.analyze_weekly(
        daily_runs=[],
        weekly_metrics={},
        hrp_diagnostic={},
    )
    assert report.narrative.startswith("## Headline")
    assert "```" not in report.narrative
