#!/bin/bash
# Idempotent catch-up wrapper for the weekly review.
#
# WHY THIS EXISTS
# ---------------
# The old schedule fired the weekly review at a single instant (Sat 06:30
# CST). That instant is fragile: if the Mac is asleep, powered off, or
# online-but-not-yet-networked right then, the week's report is silently
# lost. It happened twice in a row — weeks ending 2026-06-26 (job never
# fired) and 2026-07-03 (fired, but the Mac had no network for the whole
# run, so every host — Alpaca, Anthropic, SMTP — failed DNS resolution).
#
# This wrapper makes the job survive a sleeping/offline Mac by being:
#
#   1. IDEMPOTENT  — it only does work if THIS week's report hasn't been
#      sent yet, so it is safe to fire every day. The plist now runs it
#      daily; six days out of seven it finds the report already present
#      and exits in milliseconds.
#
#   2. CATCH-UP    — because it runs daily and self-skips, the first day
#      the Mac is awake AND online after a Friday close, the report goes
#      out. A Mac that was off all weekend still sends on Monday.
#
#   3. NETWORK-AWARE — it waits for DNS to come up before running, so a
#      just-woken Mac whose Wi-Fi isn't ready yet doesn't fail the way
#      2026-07-04 did; it simply waits, or defers to the next day's fire.
#
# The report for a given week is considered "already sent" iff its saved
# JSON exists at data/agent/weekly_reports/<friday>.json. The review writes
# that file only after the AI deep-dive succeeds, immediately before it
# emails — so its presence is a reliable "we got through the hard part"
# marker. (Residual gap: if the JSON saves but the SMTP send then fails,
# this wrapper will consider the week done and not retry the email. The
# in-app SMTP retry plus the up-front network wait below make that window
# small; both real failures observed so far were BEFORE the JSON save.)

set -uo pipefail

PROJECT_ROOT="/Users/terrancehan/Claude trader/quant"
UV="/opt/homebrew/bin/uv"
TAG="[weekly-catchup]"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

cd "${PROJECT_ROOT}" || { echo "$(ts) ${TAG} cannot cd to project root; abort."; exit 1; }

# Target the most recent COMPLETED trading week. Subtract a day first, THEN
# snap to the most recent Friday: this guarantees that even when the wrapper
# runs ON a Friday it targets the *previous* week (whose close has passed),
# never the current in-progress week.
TARGET_FRIDAY="$(date -v-1d -v-fri +%Y-%m-%d)"
REPORT_JSON="${PROJECT_ROOT}/data/agent/weekly_reports/${TARGET_FRIDAY}.json"

echo "$(ts) ${TAG} target week ending ${TARGET_FRIDAY}"

# 1. IDEMPOTENCY — already sent this week? Nothing to do.
if [[ -f "${REPORT_JSON}" ]]; then
    echo "$(ts) ${TAG} already sent (${REPORT_JSON} exists); nothing to do."
    exit 0
fi

# 2. NETWORK-WAIT — poll DNS for up to ~10 min (20 x 30s). DNS resolving is
#    the exact signal that was absent on 2026-07-04 ("nodename nor servname
#    provided"). If it never comes up we exit 0 (not an error) so launchd
#    doesn't thrash; the next daily fire retries.
NET_OK=0
for attempt in $(seq 1 20); do
    if nslookup api.anthropic.com >/dev/null 2>&1; then
        echo "$(ts) ${TAG} network up (DNS resolved on attempt ${attempt})."
        NET_OK=1
        break
    fi
    echo "$(ts) ${TAG} network down (attempt ${attempt}/20); waiting 30s..."
    sleep 30
done

if [[ "${NET_OK}" -ne 1 ]]; then
    echo "$(ts) ${TAG} network still down after 10 min; deferring to next daily fire."
    exit 0
fi

# 3. RUN — idempotent send. quant-weekly-review refits HRP, runs the AI
#    deep-dive, saves the JSON, then emails. On success the JSON now exists,
#    so tomorrow's fire self-skips at step 1.
echo "$(ts) ${TAG} running weekly review for week ending ${TARGET_FRIDAY}..."
"${UV}" run quant-weekly-review --for-date "${TARGET_FRIDAY}"
rc=$?
echo "$(ts) ${TAG} quant-weekly-review exited ${rc}."
exit "${rc}"
