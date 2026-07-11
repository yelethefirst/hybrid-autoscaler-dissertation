#!/usr/bin/env bash
# watch-progress.sh — live view of a running forecasting grid (FULL_GRID runs
# take hours; this answers "is it working or hung?" at a glance).
#
# Usage:
#   bash bin/watch-progress.sh                    # default heartbeat location
#   bash bin/watch-progress.sh path/to/progress.json
#
# The heartbeat is written by forecasting/selection.py after every
# (service, fold, model) unit. A stale updated_at_utc (> ~15 min during
# non-LSTM models, longer for LSTM) plus a SlowFitAlarm warning in the run log
# is the signature of a hang (DEV-010 class).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HB="${1:-$REPO_ROOT/experiments/results/forecasting_progress.json}"

if ! command -v watch >/dev/null 2>&1; then
  # macOS has no watch(1) by default — poor man's loop
  while true; do
    clear
    date -u +"%Y-%m-%dT%H:%M:%SZ  (poll)"
    [[ -f "$HB" ]] && cat "$HB" || echo "no heartbeat yet at: $HB"
    sleep 10
  done
else
  exec watch -n 10 "date -u +%Y-%m-%dT%H:%M:%SZ; echo; cat '$HB' 2>/dev/null || echo 'no heartbeat yet at: $HB'"
fi
