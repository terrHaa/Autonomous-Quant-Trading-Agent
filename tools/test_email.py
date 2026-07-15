"""test_email.py — verify the configured email transport actually delivers.

Run:  .venv/bin/python tools/test_email.py

Reads the live .env, builds the EmailSender exactly as the agent does, and
sends ONE real test email to REPORT_TO. Use this after switching
EMAIL_PROVIDER (e.g. to resend) to confirm delivery works from this
network — the whole point of the 2026-07-15 HTTPS migration was that SMTP
(587/465) is dead from China while HTTPS (443) is not.

Prints the resolved transport so you can see which path is active. Exits
non-zero on failure so it's usable in a checklist.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone  # timezone.utc: runs on py<3.11 too

from quant.agent.email_sender import EmailConfig, EmailSender

UTC = timezone.utc


def main() -> int:
    cfg = EmailConfig.from_env()
    transport = cfg.http_provider or f"smtp ({cfg.smtp_host}:{cfg.smtp_port})"
    print(f"transport: {transport}")
    print(f"from: {cfg.sender}  ->  to: {cfg.default_recipient}")
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        EmailSender(cfg).send(
            subject=f"quant agent — email transport test {stamp}",
            body_text=(
                "This is a test of the quant agent's email transport.\n\n"
                f"Transport: {transport}\n"
                f"Sent: {stamp}\n\n"
                "If you received this, alerts (daily reports, audit "
                "failures, gate refusals) will reach you."
            ),
            body_html=(
                f"<p>Email transport test — <b>{transport}</b>.</p>"
                f"<p>Sent {stamp}. If you got this, alerts will reach you.</p>"
            ),
        )
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1
    print("SENT — check your inbox (and spam).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
