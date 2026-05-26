# launchd jobs — the agent's schedule

Four macOS launchd User Agent jobs that run the autonomous trader.

| Job | When (local time) | What |
|---|---|---|
| `com.terrancehan.quant-daily-trade` | 09:35 Mon-Fri | Compute targets, submit market entries with stop-losses |
| `com.terrancehan.quant-daily-report` | 16:05 Mon-Fri | Email a summary of today's trades |
| `com.terrancehan.quant-weekly-review` | 16:30 Fri | Email the week's aggregate |
| `com.terrancehan.quant-monthly-review` | 16:30 on the 1st of each month | Run improver, possibly auto-apply, email |

**All four jobs assume your Mac's timezone is `America/New_York`.** If it
isn't, either set it to ET (System Settings → General → Date & Time) or
edit the `Hour` fields in each plist to match your local zone.

## First-time install

```bash
cd "/Users/terrancehan/Claude trader/quant"
bash launchd/install.sh
```

That copies the four `.plist` files to `~/Library/LaunchAgents/` and
loads them with `launchctl`. From then on macOS keeps them registered
across logins / reboots until you uninstall.

## Before the first scheduled run

Verify `.env` is fully filled in:

- `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_API_SECRET` — already populated.
- `SMTP_USERNAME` — a Gmail address you control.
- `SMTP_PASSWORD` — a 16-character app password (from
  [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords);
  requires 2FA on the Gmail account first).
- `REPORT_FROM` — typically same as `SMTP_USERNAME`.
- `REPORT_TO` — defaults to `terrancehan@outlook.com`.

Then run a dry-run to confirm everything's wired up:

```bash
uv run quant-daily-trade --dry-run
```

If that completes without errors, the live jobs are good to go.

## Verifying the jobs are loaded

```bash
launchctl list | grep terrancehan
```

You should see four entries, all with status `0` (last exit code).
Status `78` is the most common error code — it means launchctl found the
plist but `bash -lc` couldn't run the command. Usually a missing
`uv` on PATH or a wrong project path; check the `.err` log file.

## Logs

Each job writes stdout and stderr to:

```
data/agent/launchd-logs/{daily-trade,daily-report,weekly-review,monthly-review}.{out,err}
```

These files grow indefinitely — rotate manually every few months
(`mv daily-trade.out daily-trade.out.old`) if you keep the agent running
for the long haul.

## Uninstall

```bash
bash launchd/uninstall.sh
```

Removes all four plists from `~/Library/LaunchAgents` and unloads them
from launchd. Doesn't touch the project's own files (the agent's daily
JSON logs, the strategy_params.json, etc.).

## Manual one-off runs

You don't need the schedule to invoke any of the jobs — each is also a
console script:

```bash
uv run quant-daily-trade        # rebalance now
uv run quant-daily-trade --dry-run     # preview what it WOULD do
uv run quant-daily-report       # email today's report
uv run quant-daily-report --for-date 2026-05-24    # backfill
uv run quant-weekly-review
uv run quant-monthly-review
uv run quant-monthly-review --no-apply    # run improver without applying
```

## Troubleshooting

**The morning trade fired but no email arrived.**
Check `data/agent/launchd-logs/daily-trade.err` and
`daily-report.err`. The most common cause is SMTP auth failure
(Gmail app password not set). The agent emails its own failures too,
but if SMTP itself is broken, those failure emails can't go out — fall
back to the launchd error logs.

**A specific job didn't run on its scheduled day.**
launchd skips runs when the Mac is asleep at the scheduled time. If
you want missed schedules to fire on wake, the docs suggest adding
`<key>StartCalendarIntervalInterval</key>` keys; we don't currently
do that to avoid double-trades. If your Mac sleeps overnight, consider
keeping it awake at trade time via Energy Saver settings.

**I want to change the schedule.**
Edit the plist files in `launchd/`, then re-run `bash launchd/install.sh`
— the install script unloads the old version before re-loading.
