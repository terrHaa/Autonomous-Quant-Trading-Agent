"""Tests for the AI analyst's parser — JSON shape handling, parsing, defaults.

We do NOT exercise the live Anthropic API here (that would cost money and
need a key). Instead we test the response-parsing logic directly by
constructing the raw text the API would return.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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
    assert sc.sma_fast is None
    assert sc.sma_slow is None
    assert sc.mr_lookback is None
    assert sc.mr_threshold_pct is None
    assert sc.reasoning == ""


# ---------------------------------------------------------------------------
# T4.22 — parser handles SMA/MR knobs in addition to trail_pct
# ---------------------------------------------------------------------------


def test_analyze_parses_sma_fields() -> None:
    """SMA crossover knobs come through as ints."""
    raw = json.dumps({
        "analysis": "fast SMA too short for current regime",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "sma_fast": 60,
            "sma_slow": 220,
            "reasoning": "shift to longer windows; 50/200 whipsawed in May",
        },
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is not None
    assert report.proposed_state_changes.sma_fast == 60
    assert report.proposed_state_changes.sma_slow == 220
    # Un-set knobs stay None.
    assert report.proposed_state_changes.trail_pct is None
    assert report.proposed_state_changes.mr_lookback is None
    assert report.proposed_state_changes.mr_threshold_pct is None


def test_analyze_parses_mr_fields() -> None:
    """Mean-reversion knobs: lookback (int) + threshold_pct (float)."""
    raw = json.dumps({
        "analysis": "MR threshold too tight given recent vol regime",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "mr_lookback": 7,
            "mr_threshold_pct": 0.025,
            "reasoning": "loosen the baseline; vol-normalize will still scale per name",
        },
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is not None
    assert report.proposed_state_changes.mr_lookback == 7
    assert report.proposed_state_changes.mr_threshold_pct == 0.025


def test_analyze_parses_mixed_knobs() -> None:
    """All five knobs in one proposal."""
    raw = json.dumps({
        "analysis": "comprehensive retune",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "trail_pct": 0.04,
            "sma_fast": 30,
            "sma_slow": 150,
            "mr_lookback": 5,
            "mr_threshold_pct": 0.03,
            "reasoning": "regime shift evidence below",
        },
    })
    sc = _fake_analyst(raw).analyze(daily_runs=[], current_state={}).proposed_state_changes
    assert sc is not None
    assert sc.trail_pct == 0.04
    assert sc.sma_fast == 30
    assert sc.sma_slow == 150
    assert sc.mr_lookback == 5
    assert sc.mr_threshold_pct == 0.03


def test_analyze_gracefully_handles_unparseable_sma_field() -> None:
    """Bad SMA value → whole proposal discarded (matches trail_pct behavior)."""
    raw = json.dumps({
        "analysis": "...",
        "proposed_strategy": None,
        "proposed_state_changes": {
            "sma_fast": "fifty",  # not an int
            "reasoning": "...",
        },
    })
    report = _fake_analyst(raw).analyze(daily_runs=[], current_state={})
    assert report.proposed_state_changes is None


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
    weekly analyst is explicitly forbidden from doing that).

    We check for DISTINCTIVE CONTENT from each file, not the file names
    themselves — MEMORY.md gets monthly entries that may reference the
    other files by name, and that's fine; we just don't want the full
    contents loaded into weekly's prompt.
    """
    prompt = _build_weekly_system_prompt()
    # Loaded files (header markers from _build_weekly_system_prompt):
    assert "WEEKLY_ANALYST.md (your role for this call)" in prompt
    assert "STRATEGY_LIBRARY.md (the ensemble you are analyzing)" in prompt
    assert "MEMORY.md (history from the monthly analyst" in prompt
    # NOT loaded — check via header marker that only appears when the
    # file is actually concatenated in:
    assert "EDGE_TAXONOMY.md (the space of possible edges" not in prompt
    assert "ANTI_PATTERNS.md (failure modes" not in prompt
    # Distinctive content unique to those files (would only appear if
    # the full file body were loaded, not from a passing mention in MEMORY):
    assert "Coverage Map" not in prompt           # EDGE_TAXONOMY heading
    assert "Famous dead anomalies" not in prompt   # ANTI_PATTERNS heading
    # Distinctive content from WEEKLY_ANALYST.md (must be present):
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


# ---------------------------------------------------------------------------
# Cross-week / cross-month context — the self-improvement plumbing
# ---------------------------------------------------------------------------


class _CapturingClient:
    """Fake Anthropic client that captures the user message for inspection."""

    def _make_messages(self):
        outer = self

        class _M:
            @staticmethod
            def create(*, system, messages, **kw):
                outer.last_user_msg = messages[0]["content"]
                return SimpleNamespace(
                    content=[SimpleNamespace(text=outer._raw)]
                )

        return _M

    def __init__(self, raw_response: str = "ok"):
        self.last_user_msg: str | None = None
        self._raw = raw_response
        self.messages = self._make_messages()


def _analyst_with_capturing_client(raw: str = "ok") -> tuple[AIAnalyst, _CapturingClient]:
    a = object.__new__(AIAnalyst)
    a._model = "test-model"
    fake = _CapturingClient(raw)
    a._client = fake
    return a, fake


def test_analyze_weekly_includes_past_reports_in_user_message() -> None:
    """When past_weekly_reports is passed, the analyst sees them in the prompt."""
    analyst, client = _analyst_with_capturing_client("## ok\n\nbody")
    past = [
        {"week_ending": "2026-05-15", "narrative": "## Headline\n\nNVDA flagged elevated trail."},
        {"week_ending": "2026-05-22", "narrative": "## Headline\n\nNVDA stopped out at $X."},
    ]
    analyst.analyze_weekly(
        daily_runs=[], weekly_metrics={}, hrp_diagnostic={},
        past_weekly_reports=past,
    )
    msg = client.last_user_msg
    assert msg is not None
    # Both past report dates appear in the user message
    assert "2026-05-15" in msg
    assert "2026-05-22" in msg
    # Both narratives are embedded verbatim
    assert "NVDA flagged elevated trail" in msg
    assert "NVDA stopped out" in msg
    # Oldest first ordering (5-15 before 5-22)
    assert msg.index("2026-05-15") < msg.index("2026-05-22")


def test_analyze_weekly_handles_no_past_reports_explicitly() -> None:
    """First-call case: no past reports → user message says so plainly so
    the analyst doesn't pretend it has history it doesn't have."""
    analyst, client = _analyst_with_capturing_client("ok")
    analyst.analyze_weekly(
        daily_runs=[], weekly_metrics={}, hrp_diagnostic={},
        past_weekly_reports=None,
    )
    msg = client.last_user_msg
    assert "No past weekly reports" in msg


def test_analyze_monthly_includes_weekly_reports_in_user_message() -> None:
    """The monthly analyst's user message embeds the weekly narratives."""
    analyst, client = _analyst_with_capturing_client(
        '{"analysis": "x", "proposed_strategy": null}'
    )
    weekly = [
        {"week_ending": "2026-05-22", "narrative": "Watch items: tighten trail. WORTH ESCALATING TO MONTHLY REVIEW."},
        {"week_ending": "2026-05-29", "narrative": "Watch items: same as last week. WORTH ESCALATING TO MONTHLY REVIEW."},
    ]
    analyst.analyze(
        daily_runs=[], current_state={"hrp_weights": {}},
        recent_weekly_reports=weekly,
    )
    msg = client.last_user_msg
    assert msg is not None
    assert "2026-05-22" in msg
    assert "2026-05-29" in msg
    # The escalation marker survives — that's the key cross-system signal
    assert "WORTH ESCALATING TO MONTHLY REVIEW" in msg
    # And the user message prompts the analyst to address the escalation
    assert "WORTH ESCALATING TO MONTHLY REVIEW" in msg


def test_weekly_analyst_md_documents_self_improvement_loop() -> None:
    """The constitution must explicitly tell the analyst to use past reports."""
    prompt = _build_weekly_system_prompt()
    assert "Self-improvement" in prompt
    assert "WORTH ESCALATING TO MONTHLY REVIEW" in prompt


def test_analyst_md_documents_weekly_cross_reference() -> None:
    """The monthly analyst's constitution must explicitly direct it to
    read the weekly reports (workflow step 5a)."""
    prompt = _build_system_prompt()
    assert "Recent weekly reports" in prompt
    assert "WORTH ESCALATING TO MONTHLY REVIEW" in prompt


# ---------------------------------------------------------------------------
# Pipeline self-audit — the institutional-grade infra-review pathway
# ---------------------------------------------------------------------------


def test_analyst_md_directs_pipeline_self_audit() -> None:
    """ANALYST.md §6 step 5c must direct the analyst to run the monthly
    pipeline self-audit and emit findings for drift/dead-code/etc."""
    prompt = _build_system_prompt()
    assert "Pipeline self-audit" in prompt
    assert "pipeline_findings" in prompt
    # The four finding categories must be documented:
    assert "Drift" in prompt
    assert "Dead code" in prompt
    assert "Below industry norm" in prompt
    assert "Missing safeguard" in prompt


def test_response_shape_documents_pipeline_findings() -> None:
    """The strict JSON response schema must show the pipeline_findings
    array — otherwise the model won't emit it."""
    prompt = _build_system_prompt()
    assert "pipeline_findings" in prompt
    assert "severity" in prompt
    assert "recommendation" in prompt


def test_analyze_passes_pipeline_snapshot_into_user_message() -> None:
    """When monthly_review passes a pipeline_snapshot, the user message
    must embed it AND tell the analyst to review it."""
    from quant.agent.ai_analyst import PipelineFinding  # noqa: F401
    analyst, client = _analyst_with_capturing_client(
        '{"analysis": "x", "proposed_strategy": null, "pipeline_findings": []}'
    )
    snapshot = {
        "operator_hard_rules_in_code": {"MAX_POSITION_WEIGHT": 0.20},
        "config_yaml_values": {"risk_max_position_weight": 0.05},
        "wiring_status": {"drawdown_kill_switch_active_in_daily_trade": False},
        "industry_norms_for_comparison": {"max_position_weight_institutional": "0.03 to 0.05"},
    }
    analyst.analyze(
        daily_runs=[], current_state={"hrp_weights": {}},
        pipeline_snapshot=snapshot,
    )
    msg = client.last_user_msg
    assert msg is not None
    # Snapshot embedded as JSON
    assert "Pipeline Self-Audit" in msg
    assert "MAX_POSITION_WEIGHT" in msg
    assert "drawdown_kill_switch_active_in_daily_trade" in msg
    # Directives to act on it
    assert "pipeline_findings" in msg
    assert "drift" in msg.lower() or "Drift" in msg
    assert "wiring_status" in msg


def test_analyze_parses_pipeline_findings_list() -> None:
    """A response with pipeline_findings must come back as a list of
    PipelineFinding dataclass instances."""
    raw = json.dumps({
        "analysis": "...",
        "proposed_strategy": None,
        "proposed_state_changes": None,
        "pipeline_findings": [
            {
                "severity": "critical",
                "category": "drift",
                "description": "MAX_POSITION_WEIGHT in code (0.20) != config (0.05).",
                "recommendation": "Lower the code constant to 0.05.",
            },
            {
                "severity": "high",
                "category": "dead_code",
                "description": "Drawdown kill switch is configured but not called.",
                "recommendation": "Wire it into run_daily_trade.",
            },
        ],
    })
    analyst, _client = _analyst_with_capturing_client(raw)
    report = analyst.analyze(daily_runs=[], current_state={})
    assert len(report.pipeline_findings) == 2
    assert report.pipeline_findings[0].severity == "critical"
    assert report.pipeline_findings[0].category == "drift"
    assert "MAX_POSITION_WEIGHT" in report.pipeline_findings[0].description
    assert report.pipeline_findings[1].category == "dead_code"


def test_analyze_handles_missing_pipeline_findings_as_empty_list() -> None:
    """Old-format responses without pipeline_findings → empty list."""
    raw = json.dumps({"analysis": "x", "proposed_strategy": None})
    analyst, _ = _analyst_with_capturing_client(raw)
    report = analyst.analyze(daily_runs=[], current_state={})
    assert report.pipeline_findings == []


def test_analyst_md_documents_triangulation() -> None:
    """ANALYST.md §6 step 5b must direct the monthly analyst to
    triangulate weekly narratives + raw daily table + monthly stats."""
    prompt = _build_system_prompt()
    # Step 5b heading + the explicit instruction
    assert "Triangulate" in prompt or "triangulate" in prompt
    assert "Monthly Statistical View" in prompt
    # The 3 data sources should all be named
    assert "weekly narratives" in prompt.lower()
    assert "daily" in prompt.lower()


def test_analyze_monthly_includes_monthly_metrics_in_user_message() -> None:
    """Monthly metrics get embedded as a JSON block in the user prompt."""
    analyst, client = _analyst_with_capturing_client(
        '{"analysis": "x", "proposed_strategy": null}'
    )
    metrics = {
        "n_days": 30,
        "ann_sharpe": 1.2,
        "lag1_autocorrelation": 0.18,
        "day_of_week_breakdown": {"Mon": {"mean_return_pct": -0.3, "n": 5}},
        "longest_losing_streak_days": 3,
    }
    analyst.analyze(
        daily_runs=[], current_state={"hrp_weights": {}},
        monthly_metrics=metrics,
    )
    msg = client.last_user_msg
    assert msg is not None
    # Section header + content
    assert "Monthly Statistical View" in msg
    assert "lag1_autocorrelation" in msg
    assert "0.18" in msg
    # Triangulation directive should be present
    assert "TRIANGULATE" in msg or "triangulate" in msg.lower()


def test_analyze_monthly_without_metrics_omits_section() -> None:
    """If monthly_metrics is None, the section is skipped — backwards-compat."""
    analyst, client = _analyst_with_capturing_client(
        '{"analysis": "x", "proposed_strategy": null}'
    )
    analyst.analyze(daily_runs=[], current_state={"hrp_weights": {}})
    msg = client.last_user_msg
    assert "Monthly Statistical View" not in msg


def test_summarise_runs_includes_daily_delta_column() -> None:
    """The daily-runs table now has a Daily Δ% column so the analyst can
    spot streaks without computing deltas itself."""
    from quant.agent.ai_analyst import _summarise_runs
    runs = [
        {"date": "2024-06-03", "execution_report": {"account_equity_before": 100_000.0, "submitted_orders": []}, "target_weights": {}},
        {"date": "2024-06-04", "execution_report": {"account_equity_before": 101_000.0, "submitted_orders": []}, "target_weights": {}},
    ]
    table = _summarise_runs(runs)
    assert "Daily Δ%" in table
    # +1% should appear in row 2 (formatted)
    assert "+1.00%" in table


# ---------------------------------------------------------------------------
# format_ai_error — the actionable failure-message helper (T-fix A)
# ---------------------------------------------------------------------------


class _Fake403(Exception):
    """Mimics anthropic.PermissionDeniedError just by class name."""
    pass


def test_format_ai_error_detects_403_by_class_name() -> None:
    from quant.agent.ai_analyst import format_ai_error
    # The real Anthropic class is named PermissionDeniedError; we match by
    # exception class name so we don't have to import anthropic here.
    exc = type("PermissionDeniedError", (Exception,), {})(
        "Error code: 403 - {'error': {'type': 'forbidden', "
        "'message': 'Request not allowed'}}"
    )
    msg = format_ai_error(exc)
    assert "403" in msg
    assert "VPN" in msg  # actionable hint
    assert "--ai-only" in msg  # recovery recipe
    assert "quant-weekly-review" in msg or "quant-monthly-review" in msg


def test_format_ai_error_detects_403_by_message_body() -> None:
    """Even an unrelated exception class lands on the 403 branch if the
    message contains 'forbidden' or '403' — defends against the SDK
    renaming the class in a future release."""
    from quant.agent.ai_analyst import format_ai_error
    exc = RuntimeError("API returned forbidden: Request not allowed")
    msg = format_ai_error(exc)
    assert "403" in msg or "VPN" in msg


def test_format_ai_error_detects_connection_error() -> None:
    from quant.agent.ai_analyst import format_ai_error
    exc = type("APIConnectionError", (Exception,), {})("Connection error.")
    msg = format_ai_error(exc)
    assert "connection" in msg.lower()
    assert "--ai-only" in msg
    # Should NOT mention "VPN exit IP flagged" (that's the 403 branch).
    assert "exit IP" not in msg


def test_format_ai_error_unknown_exception_still_includes_recovery_recipe() -> None:
    """Even an unclassified exception should hand the operator the
    --ai-only command — partial failure is better than no guidance."""
    from quant.agent.ai_analyst import format_ai_error
    msg = format_ai_error(RuntimeError("some weird bug"))
    assert "--ai-only" in msg
    assert "some weird bug" in msg
