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

# HTTPS email providers. These POST to a REST API on port 443 — which
# demonstrably works from the operator's China network (Anthropic + Alpaca
# both reach 443) — where outbound SMTP (587/465) is chronically reset.
# T-incident 2026-07-15: an SMTP-only alert path meant a days-long silent
# trading halt because every "refused to trade" / audit-failure email died
# at the SMTP layer and never reached the operator. HTTPS delivery is the
# fix. See tools + .env.example for activation.
_HTTP_PROVIDERS = ("resend", "sendgrid")


@dataclass(frozen=True)
class EmailConfig:
    """Email transport + From/To configuration. Frozen because reload-mid-
    send would be a mess; rebuild a new instance instead.

    ``http_provider`` selects the transport: one of ``_HTTP_PROVIDERS`` for
    HTTPS delivery (443, reliable from China), or "" for SMTP (legacy
    fallback). When an HTTPS provider is set, the SMTP fields may be blank.
    """

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    sender: str
    default_recipient: str
    http_provider: str = ""      # "resend" | "sendgrid" | "" (=SMTP)
    http_api_key: str = ""

    @classmethod
    def from_env(cls) -> EmailConfig:
        """Build from environment variables (loaded from ``.env`` if present).

        Transport chosen by ``EMAIL_PROVIDER`` (default "smtp"):
          - ``resend`` / ``sendgrid``: needs ``<PROVIDER>_API_KEY`` +
            REPORT_FROM + REPORT_TO. SMTP vars optional.
          - ``smtp`` (default): needs SMTP_HOST/PORT/USERNAME/PASSWORD +
            REPORT_FROM + REPORT_TO (the legacy path).
        """
        load_dotenv()
        provider = (os.environ.get("EMAIL_PROVIDER") or "smtp").strip().lower()
        sender = os.environ.get("REPORT_FROM") or os.environ.get("SMTP_USERNAME")
        recipient = os.environ.get("REPORT_TO")

        if provider in _HTTP_PROVIDERS:
            api_key = os.environ.get(f"{provider.upper()}_API_KEY")
            missing = [
                name for name, val in [
                    (f"{provider.upper()}_API_KEY", api_key),
                    ("REPORT_FROM", sender),
                    ("REPORT_TO", recipient),
                ] if not val
            ]
            if missing:
                raise RuntimeError(
                    f"EMAIL_PROVIDER={provider} but missing: {missing}. "
                    "See .env.example for setup."
                )
            return cls(
                smtp_host="", smtp_port=0, smtp_username="", smtp_password="",
                sender=sender, default_recipient=recipient,
                http_provider=provider, http_api_key=api_key,
            )

        # Legacy SMTP path.
        required = {
            "SMTP_HOST": os.environ.get("SMTP_HOST"),
            "SMTP_PORT": os.environ.get("SMTP_PORT"),
            "SMTP_USERNAME": os.environ.get("SMTP_USERNAME"),
            "SMTP_PASSWORD": os.environ.get("SMTP_PASSWORD"),
            "REPORT_FROM": sender,
            "REPORT_TO": recipient,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"missing email environment variables: {missing}. "
                f"Fill them in .env per the comments in .env.example. "
                f"(Gmail SMTP requires 2FA + a 16-char app password. Or set "
                f"EMAIL_PROVIDER=resend for HTTPS delivery.)"
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
        http_post: Any | None = None,
    ) -> None:
        """Initialize with a config (loaded from env if None).

        ``smtp_client_factory`` is a callable that returns an object with
        ``.starttls / .login / .send_message / .quit`` methods —
        injected by tests so we don't need a real SMTP server.
        ``http_post`` is an injectable ``requests.post``-shaped callable
        for testing the HTTPS transport without network.
        Production code leaves both None.
        """
        self._config = config or EmailConfig.from_env()
        self._smtp_factory = smtp_client_factory
        self._http_post = http_post

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

        # HTTPS transport (port 443, reliable from China) takes precedence
        # when configured. This is the primary path post-2026-07-15.
        if self._config.http_provider in _HTTP_PROVIDERS:
            self._send_http(
                subject=subject, body_text=body_text, body_html=body_html, to=to,
            )
            return

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
        # The whole connect/login/send dance is retried as a unit on
        # transient socket errors — Gmail occasionally resets connections
        # through the VPN/proxy and a fresh handshake recovers cleanly.
        # SMTPAuthenticationError is checked FIRST so we don't retry a
        # permanent credential problem (just makes Gmail rate-limit you).
        from quant.util.retry import retry_on_transient

        context = ssl.create_default_context()

        def _connect_and_send() -> None:
            with smtplib.SMTP(
                self._config.smtp_host, self._config.smtp_port, timeout=30
            ) as client:
                client.starttls(context=context)
                client.login(
                    self._config.smtp_username, self._config.smtp_password
                )
                client.send_message(msg)

        try:
            retry_on_transient(
                _connect_and_send,
                transient=(
                    smtplib.SMTPServerDisconnected,
                    smtplib.SMTPConnectError,
                    ConnectionError,   # builtin — covers socket-level resets
                    OSError,           # parent of socket.error; e.g. ECONNRESET
                ),
                description="SMTP send",
            )
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

    # ---- HTTPS transport (Resend / SendGrid, port 443) ------------------

    def _send_http(
        self, *, subject: str, body_text: str, body_html: str | None, to: str,
    ) -> None:
        """POST the email to the configured HTTPS provider's REST API.

        Retries the POST as a unit on connection-level errors (same policy
        as SMTP), raises RuntimeError on a non-2xx response or exhausted
        retries. Runs over 443, which is reachable where SMTP is not.
        """
        import requests

        from quant.util.retry import retry_on_transient

        url, headers, payload = _build_http_request(
            self._config.http_provider, self._config.http_api_key,
            sender=self._config.sender, to=to,
            subject=subject, body_text=body_text, body_html=body_html,
        )
        post = self._http_post or requests.post

        def _do_post():
            resp = post(url, headers=headers, json=payload, timeout=30)
            code = getattr(resp, "status_code", 0)
            if not (200 <= code < 300):
                body = getattr(resp, "text", "")
                # 4xx = permanent (bad key, unverified sender): don't retry.
                if 400 <= code < 500:
                    raise RuntimeError(
                        f"{self._config.http_provider} rejected the email "
                        f"(HTTP {code}): {body[:300]}"
                    )
                raise ConnectionError(f"HTTP {code} from provider: {body[:200]}")
            return resp

        try:
            retry_on_transient(
                _do_post,
                transient=(ConnectionError, OSError),
                description=f"{self._config.http_provider} send",
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"{self._config.http_provider} send failed: {e}"
            ) from e


def _build_http_request(
    provider: str, api_key: str, *,
    sender: str, to: str, subject: str, body_text: str, body_html: str | None,
) -> tuple[str, dict, dict]:
    """Return (url, headers, json_payload) for the provider's send endpoint."""
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    if provider == "resend":
        payload: dict = {
            "from": sender, "to": [to], "subject": subject, "text": body_text,
        }
        if body_html is not None:
            payload["html"] = body_html
        return "https://api.resend.com/emails", headers, payload
    if provider == "sendgrid":
        content = [{"type": "text/plain", "value": body_text}]
        if body_html is not None:
            content.append({"type": "text/html", "value": body_html})
        payload = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {"email": sender},
            "subject": subject,
            "content": content,
        }
        return "https://api.sendgrid.com/v3/mail/send", headers, payload
    raise RuntimeError(f"unknown HTTPS email provider: {provider!r}")
