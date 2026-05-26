"""Tests for the email sender.

Pattern: inject a fake SMTP client via smtp_client_factory so we never
talk to a real SMTP server. The fake records every send_message call;
tests assert on the resulting MIMEMultipart's headers and parts.
"""

from __future__ import annotations

import pytest

from quant.agent.email_sender import EmailConfig, EmailSender

# ---------------------------------------------------------------------------
# Fake SMTP client
# ---------------------------------------------------------------------------


class _FakeSMTPClient:
    """Records messages instead of sending them. Auto-shareable via factory."""

    def __init__(self) -> None:
        self.messages: list = []
        self.quit_called = False
        self.login_called = False
        self.starttls_called = False

    def starttls(self, context=None) -> None:
        self.starttls_called = True

    def login(self, user: str, password: str) -> None:
        self.login_called = True

    def send_message(self, msg) -> None:
        self.messages.append(msg)

    def quit(self) -> None:
        self.quit_called = True


def _make_config() -> EmailConfig:
    """Hardcoded config so tests don't depend on .env."""
    return EmailConfig(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="agent@example.com",
        smtp_password="fake-app-password",
        sender="agent@example.com",
        default_recipient="ops@example.com",
    )


# ---------------------------------------------------------------------------
# EmailConfig — env loading
# ---------------------------------------------------------------------------


def test_env_loader_raises_when_required_vars_missing(monkeypatch) -> None:
    """Fail loudly at startup rather than silently in the daily loop."""
    # Make load_dotenv a no-op so it doesn't re-populate from .env.
    monkeypatch.setattr("quant.agent.email_sender.load_dotenv", lambda: None)
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD",
              "REPORT_FROM", "REPORT_TO"):
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(RuntimeError, match="missing email environment"):
        EmailConfig.from_env()


def test_env_loader_falls_back_to_smtp_username_for_from(monkeypatch) -> None:
    """REPORT_FROM defaults to SMTP_USERNAME if not explicitly set —
    convenient since most users send from their authenticating address."""
    monkeypatch.setattr("quant.agent.email_sender.load_dotenv", lambda: None)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "agent@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("REPORT_TO", "ops@example.com")
    monkeypatch.delenv("REPORT_FROM", raising=False)

    cfg = EmailConfig.from_env()
    assert cfg.sender == "agent@example.com"
    assert cfg.default_recipient == "ops@example.com"


def test_env_loader_rejects_non_integer_port(monkeypatch) -> None:
    monkeypatch.setattr("quant.agent.email_sender.load_dotenv", lambda: None)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "notaport")
    monkeypatch.setenv("SMTP_USERNAME", "u")
    monkeypatch.setenv("SMTP_PASSWORD", "p")
    monkeypatch.setenv("REPORT_FROM", "u@x.com")
    monkeypatch.setenv("REPORT_TO", "t@x.com")
    with pytest.raises(RuntimeError, match="SMTP_PORT"):
        EmailConfig.from_env()


# ---------------------------------------------------------------------------
# EmailSender.send
# ---------------------------------------------------------------------------


def test_send_constructs_message_with_correct_headers() -> None:
    fake = _FakeSMTPClient()
    sender = EmailSender(_make_config(), smtp_client_factory=lambda: fake)
    sender.send(
        subject="daily report",
        body_text="hello world",
    )
    assert len(fake.messages) == 1
    msg = fake.messages[0]
    assert msg["Subject"] == "daily report"
    assert msg["From"] == "agent@example.com"
    assert msg["To"] == "ops@example.com"


def test_send_plain_only_attaches_one_part() -> None:
    """No HTML body → multipart has just the plain alternative."""
    fake = _FakeSMTPClient()
    sender = EmailSender(_make_config(), smtp_client_factory=lambda: fake)
    sender.send(subject="x", body_text="plain only")
    msg = fake.messages[0]
    parts = msg.get_payload()
    assert len(parts) == 1
    assert parts[0].get_content_type() == "text/plain"
    assert "plain only" in parts[0].get_payload()


def test_send_includes_html_when_provided() -> None:
    """multipart/alternative with plain FIRST and html LAST — mail clients
    pick the LAST part they can render."""
    fake = _FakeSMTPClient()
    sender = EmailSender(_make_config(), smtp_client_factory=lambda: fake)
    sender.send(
        subject="x",
        body_text="plain",
        body_html="<p>html</p>",
    )
    msg = fake.messages[0]
    parts = msg.get_payload()
    assert len(parts) == 2
    assert parts[0].get_content_type() == "text/plain"
    assert parts[1].get_content_type() == "text/html"


def test_send_overrides_recipient() -> None:
    """`recipient=` kwarg overrides the default REPORT_TO."""
    fake = _FakeSMTPClient()
    sender = EmailSender(_make_config(), smtp_client_factory=lambda: fake)
    sender.send(
        subject="x",
        body_text="x",
        recipient="someone-else@example.com",
    )
    assert fake.messages[0]["To"] == "someone-else@example.com"


def test_send_calls_quit_on_client() -> None:
    """Always close the SMTP connection cleanly — tests catch leaks."""
    fake = _FakeSMTPClient()
    sender = EmailSender(_make_config(), smtp_client_factory=lambda: fake)
    sender.send(subject="x", body_text="x")
    assert fake.quit_called
