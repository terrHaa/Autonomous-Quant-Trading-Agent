"""Tests for the HTTPS email transport (Resend / SendGrid)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant.agent.email_sender import EmailConfig, EmailSender, _build_http_request


def _http_config(provider: str) -> EmailConfig:
    return EmailConfig(
        smtp_host="", smtp_port=0, smtp_username="", smtp_password="",
        sender="alerts@quant.example", default_recipient="me@outlook.com",
        http_provider=provider, http_api_key="test-key-123",
    )


def test_resend_payload_shape() -> None:
    url, headers, payload = _build_http_request(
        "resend", "k", sender="a@x.com", to="b@y.com",
        subject="hi", body_text="plain", body_html="<b>rich</b>",
    )
    assert url == "https://api.resend.com/emails"
    assert headers["Authorization"] == "Bearer k"
    assert payload == {"from": "a@x.com", "to": ["b@y.com"], "subject": "hi",
                       "text": "plain", "html": "<b>rich</b>"}


def test_sendgrid_payload_shape() -> None:
    url, _, payload = _build_http_request(
        "sendgrid", "k", sender="a@x.com", to="b@y.com",
        subject="hi", body_text="plain", body_html=None,
    )
    assert url == "https://api.sendgrid.com/v3/mail/send"
    assert payload["personalizations"][0]["to"][0]["email"] == "b@y.com"
    assert payload["from"]["email"] == "a@x.com"
    assert [c["type"] for c in payload["content"]] == ["text/plain"]  # no html


def test_send_posts_to_provider_on_success() -> None:
    captured = {}

    def _post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(status_code=200, text='{"id":"abc"}')

    EmailSender(_http_config("resend"), http_post=_post).send(
        subject="daily report", body_text="body",
    )
    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["json"]["subject"] == "daily report"
    assert captured["json"]["to"] == ["me@outlook.com"]


def test_4xx_is_permanent_no_retry() -> None:
    calls = {"n": 0}

    def _post(url, *, headers, json, timeout):
        calls["n"] += 1
        return SimpleNamespace(status_code=401, text="invalid api key")

    with pytest.raises(RuntimeError, match="rejected"):
        EmailSender(_http_config("resend"), http_post=_post).send(
            subject="s", body_text="b",
        )
    assert calls["n"] == 1          # 4xx not retried


def test_5xx_retries_then_succeeds(monkeypatch) -> None:
    import quant.util.retry as retry_mod
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def _post(url, *, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(status_code=503, text="upstream busy")
        return SimpleNamespace(status_code=202, text="")

    EmailSender(_http_config("sendgrid"), http_post=_post).send(
        subject="s", body_text="b",
    )
    assert calls["n"] == 3          # retried through the 5xx


def test_from_env_selects_http_provider(monkeypatch) -> None:
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("REPORT_FROM", "a@x.com")
    monkeypatch.setenv("REPORT_TO", "b@y.com")
    cfg = EmailConfig.from_env()
    assert cfg.http_provider == "resend"
    assert cfg.http_api_key == "re_test"
    assert cfg.sender == "a@x.com"
