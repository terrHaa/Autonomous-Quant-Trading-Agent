#!/bin/bash
# Unload and remove all five agent launchd jobs.
#
# Safe to run even if nothing is loaded — launchctl unload is tolerant of
# missing/unloaded plists when given || true.

set -euo pipefail

LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
JOBS=(
    "com.terrancehan.quant-daily-trade"
    "com.terrancehan.quant-daily-report"
    "com.terrancehan.quant-daily-audit"
    "com.terrancehan.quant-weekly-review"
    "com.terrancehan.quant-monthly-review"
)

for job in "${JOBS[@]}"; do
    plist="${LAUNCH_AGENTS_DIR}/${job}.plist"
    if [[ -f "${plist}" ]]; then
        launchctl unload "${plist}" 2>/dev/null || true
        rm -f "${plist}"
        echo "removed: ${job}"
    fi
done

echo ""
echo "All four jobs uninstalled. To re-install: bash launchd/install.sh"
