#!/bin/bash
# Install the four launchd jobs that drive the autonomous agent.
#
# Copies the .plist files to ~/Library/LaunchAgents/ and loads them with
# launchctl. Idempotent — safe to re-run after editing a plist (it will
# unload the old version first).
#
# Run from anywhere; the script resolves its own directory to find the
# plists. Adjust the AGENT_PROJECT_DIR variable if the project moves.

set -euo pipefail

# Resolve the directory of this script (the launchd/ directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure the LaunchAgents dir exists.
mkdir -p "${LAUNCH_AGENTS_DIR}"

# Ensure the log directory exists.
mkdir -p "${PROJECT_ROOT}/data/agent/launchd-logs"

JOBS=(
    "com.terrancehan.quant-daily-trade"
    "com.terrancehan.quant-daily-report"
    "com.terrancehan.quant-weekly-review"
    "com.terrancehan.quant-monthly-review"
)

for job in "${JOBS[@]}"; do
    src="${SCRIPT_DIR}/${job}.plist"
    dst="${LAUNCH_AGENTS_DIR}/${job}.plist"
    if [[ ! -f "${src}" ]]; then
        echo "ERROR: plist not found at ${src}" >&2
        exit 1
    fi

    # Unload any existing version first (no-op if absent).
    launchctl unload "${dst}" 2>/dev/null || true

    cp "${src}" "${dst}"
    launchctl load "${dst}"
    echo "loaded: ${job}"
done

echo ""
echo "All four jobs installed and loaded."
echo ""
echo "To verify:  launchctl list | grep terrancehan"
echo "To check a specific job:  launchctl print gui/\$(id -u)/com.terrancehan.quant-daily-trade"
echo "Logs:       ${PROJECT_ROOT}/data/agent/launchd-logs/"
echo ""
echo "REMINDER: before the first scheduled run, ensure .env has:"
echo "  - ALPACA_PAPER_API_KEY / SECRET (already filled in this project)"
echo "  - SMTP_USERNAME / SMTP_PASSWORD (Gmail app password — see .env.example)"
echo "  - REPORT_FROM, REPORT_TO"
