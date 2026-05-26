"""email_sender.py — SMTP wrapper for the agent's daily/weekly/monthly reports.

Why a custom wrapper at all?
----------------------------
``smtplib`` is in the stdlib and works fine, but its API is verbose and
trips on TLS subtleties. This wrapper:

- Loads SMTP credentials from ``.env`` via ``EmailConfig.from_env()`` and
  refuses to send if any required variable is missing (so the agent
  fails fast at startup, not deep in the daily loop).
- Sends plain-text + optional HTML alternative parts so reports look
  good in Outlook / Gmail webmail (which prefer HTML) without losing
  the plain-text fallback for terminal mail clients.
- Catches and re-raises auth errors with a clear-message: Gmail's
  "Application-specific password required" is the #1 first-time stumble,
  so we surface that explicitly.

Why Gmail SMTP specifically
---------------------------
Gmail's SMTP is the most reliable free option for low-volume programmatic
sending. STARTTLS on port 587 with an app password (after enabling 2FA)
is the supported path. Sending to Outlook/Hotmail addresses works
without issues — they're recipients, not the sender.

What this is NOT
----------------
- Not a queue / retry mechanism. If SMTP is down, we raise and let the
  caller decide whether to retry, log, or skip. A daily job missing one
  day's email isn't catastrophic; aggressive retry could cause spam.
- Not bulk-send. One recipient per call. Operators are humans.
- Not async. SMTP is slow but a few seconds per send is fine for the
  daily cadence; threading adds bugs without benefit.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class EmailConfig:
    """SMTP + From/To configuration. Frozen because reload-mid-send would
    be a mess; rebuild a new instance instead."""

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    sender: str
    default_recipient: str

    @classmethod
    def from_env(cls) -> EmailConfig:
        """Build from environment variables (loaded from ``.env`` if present).

        Required env vars (any missing → RuntimeError with a helpful message):
            SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
            REPORT_FROM, REPORT_TO
        """
        load_dotenv()
        required = {
            "SMTP_HOST": os.environ.get("SMTP_HOST"),
            "SMTP_PORT": os.environ.get("SMTP_PORT"),
            "SMTP_USERNAME": os.environ.get("SMTP_USERNAME"),
            "SMTP_PASSWORD": os.environ.get("SMTP_PASSWORD"),
            "REPORT_FROM": os.environ.get("REPORT_FROM") or os.environ.get("SMTP_USERNAME"),
            "REPORT_TO": os.environ.get("REPORT_TO"),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"missing email environment variables: {missing}. "
                f"Fill them in .env per the comments in .env.example. "
                f"(Gmail SMTP requires 2FA + a 16-char app password.)"
            )
        try:
            port_int = int(required["SMTP_PORT"])
        except ValueError as e:
            raise RuntimeError(
                f"SMTP_PORT must be an integer; got {required['SMTP_PORT']!r}"
            ) from e
        return cls(
            smtp_host=required["SMTP_HOST"],
            smtp_port=port_int,
            smtp_username=required["SMTP_USERNAME"],
            smtp_password=required["SMTP_PASSWORD"],
            sender=required["REPORT_FROM"],
            default_recipient=required["REPORT_TO"],
        )


class EmailSender:
    """Send plain-text-with-optional-HTML email via SMTP.

    Usage:

        sender = EmailSender()              # loads .env automatically
        sender.send(
            subject="quant agent — daily report 2026-05-26",
            body_text="...markdown rendered as plain text...",
            body_html="...markdown rendered as html (optional)...",
        )

    Or inject a custom ``EmailConfig`` for tests / multi-account setups.
    """

    def __init__(
        self,
        config: EmailConfig | None = None,
        *,
        smtp_client_factory: Any | None = None,
    ) -> None:
        """Initialize with a config (loaded from env if None).

        ``smtp_client_factory`` is a callable that returns an object with
        ``.starttls / .login / .send_message / .quit`` methods —
        injected by tests so we don't need a real SMTP server.
        Production code leaves it None and we use ``smtplib.SMTP``.
        """
        self._config = config or EmailConfig.from_env()
        self._smtp_factory = smtp_client_factory

    @property
    def config(self) -> EmailConfig:
        return self._config

    def send(
        self,
        *,
        subject: str,
        body_text: str,
        body_html: str | None = None,
        recipient: str | None = None,
    ) -> None:
        """Send one email. Raises on SMTP failure with a useful message.

        Parameters
        ----------
        subject
            The Subject: header. Keep short — daily reports may pile up.
        body_text
            Plain-text body. Always sent (even when HTML is provided)
            for terminal-mail-client fallback.
        body_html
            Optional HTML alternative. If provided, modern mail clients
            (Outlook, Gmail web) will prefer it. Markdown-to-HTML
            conversion is the caller's job.
        recipient
            Override the default REPORT_TO. None → use the configured
            default.
        """
        to = recipient or self._config.default_recipient

        # Use 'alternative' multipart so the mail client picks the
        # richest version it supports.
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._config.sender
        msg["To"] = to
        msg.attach(MIMEText(body_text, "plain"))
        if body_html is not None:
            # IMPORTANT: HTML must be attached AFTER plain so it's
            # preferred per the multipart/alternative spec ("last part
            # is the richest").
            msg.attach(MIMEText(body_html, "html"))

        if self._smtp_factory is not None:
            client = self._smtp_factory()
            try:
                client.send_message(msg)
            finally:
                close = getattr(client, "quit", None)
                if close is not None:
                    close()
            return

        # Real send: STARTTLS on the configured host:port, login, send, quit.
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(
                self._config.smtp_host, self._config.smtp_port, timeout=30
            ) as client:
                client.starttls(context=context)
                client.login(
                    self._config.smtp_username, self._config.smtp_password
                )
                client.send_message(msg)
        except smtplib.SMTPAuthenticationError as e:
            # The single most common first-time problem with Gmail.
            raise RuntimeError(
                "SMTP auth failed. For Gmail you need a 16-character "
                "*app password*, not your normal Gmail password. "
                "Enable 2FA on the Gmail account, then generate one at "
                "myaccount.google.com/apppasswords. Original error: "
                f"{e}"
            ) from e
        except smtplib.SMTPException as e:
            raise RuntimeError(f"SMTP send failed: {e}") from e
