#!/usr/bin/env bash
# watch-ab-trial.sh — live terminal monitor for A/B trials and telemetry campaigns.
#
# Queries Prometheus directly; no browser or port-forward to Grafana required.
# Useful when SSH'd into the AWS instance.  Pairs with the Grafana dashboard at
# infra/observability/dashboards/ab-trial.json for a full visual view.
#
# Usage:
#   bash bin/watch-ab-trial.sh                          # start time = now
#   bash bin/watch-ab-trial.sh 2026-07-10T09:00:00Z    # explicit start (ISO-8601)
#
# Environment overrides:
#   PROM_PORT=9090          Prometheus port (default 9090)
#   DURATION_HOURS=5        Expected campaign length for progress bar (default 5)
#   INTERVAL=15             Refresh interval in seconds (default 15)

set -euo pipefail

PROM_PORT="${PROM_PORT:-9090}"
PROM_URL="http://localhost:${PROM_PORT}"
DURATION_HOURS="${DURATION_HOURS:-5}"
DURATION_SECS=$(( DURATION_HOURS * 3600 ))
INTERVAL="${INTERVAL:-15}"

START_ARG="${1:-}"
if [[ -n "$START_ARG" ]]; then
  START_EPOCH=$(python3 -c "
from datetime import datetime, timezone
import sys
s = sys.argv[1].replace('Z', '+00:00')
print(int(datetime.fromisoformat(s).timestamp()))
" "$START_ARG")
else
  START_EPOCH=$(date +%s)
fi

# ─── Prometheus helpers ───────────────────────────────────────────────────────

_prom_scalar() {
  # Query Prometheus and return the first scalar value as a float string, or "—".
  local query="$1"
  local fmt="${2:-%.1f}"
  local encoded
  encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$query")
  local json
  json=$(curl -fsS --max-time 3 "${PROM_URL}/api/v1/query?query=${encoded}" 2>/dev/null) || { echo "—"; return; }
  python3 -c "
import json, sys
fmt = sys.argv[2]
try:
    d = json.loads(sys.argv[1])
    results = d.get('data', {}).get('result', [])
    if results:
        v = float(results[0]['value'][1])
        print(fmt % v)
    else:
        print('—')
except:
    print('—')
" "$json" "$fmt"
}

_prom_table() {
  # Query Prometheus and print one row per series as "  label   value".
  local query="$1"
  local label_key="$2"
  local unit="${3:-}"
  local encoded
  encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$query")
  local json
  json=$(curl -fsS --max-time 3 "${PROM_URL}/api/v1/query?query=${encoded}" 2>/dev/null) || { echo "  (prometheus unreachable)"; return; }
  python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    label = sys.argv[2]
    unit  = sys.argv[3]
    results = d.get('data', {}).get('result', [])
    pairs = [(r['metric'].get(label, '?'), float(r['value'][1])) for r in results]
    pairs.sort(key=lambda x: x[0])
    for name, val in pairs:
        suffix = (' ' + unit) if unit else ''
        print(f'  {name:<35} {val:>6.0f}{suffix}')
except Exception as e:
    print(f'  (error: {e})')
" "$json" "$label_key" "$unit"
}

_check_prom() {
  curl -fsS --max-time 2 "${PROM_URL}/-/healthy" >/dev/null 2>&1
}

# ─── Pre-flight ───────────────────────────────────────────────────────────────
if ! _check_prom; then
  echo ""
  echo "  Prometheus not reachable at ${PROM_URL}"
  echo ""
  echo "  Start a port-forward first (if running inside the cluster):"
  echo "    kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090 &"
  echo ""
  echo "  Or if running directly on the AWS instance:"
  echo "    export PROM_PORT=9090   # NodePort or direct address"
  echo ""
  exit 1
fi

echo ""
echo "  Prometheus OK.  Starting monitor (Ctrl-C to exit)..."
sleep 1

# ─── Main render loop ─────────────────────────────────────────────────────────
while true; do
  clear

  NOW_EPOCH=$(date +%s)
  ELAPSED=$(( NOW_EPOCH - START_EPOCH ))
  REMAIN=$(( DURATION_SECS - ELAPSED ))
  [[ $REMAIN -lt 0 ]] && REMAIN=0
  E_H=$(( ELAPSED / 3600 ))
  E_M=$(( (ELAPSED % 3600) / 60 ))
  E_S=$(( ELAPSED % 60 ))
  R_H=$(( REMAIN / 3600 ))
  R_M=$(( (REMAIN % 3600) / 60 ))
  R_S=$(( REMAIN % 60 ))
  PCT=$(( ELAPSED * 100 / DURATION_SECS ))
  [[ $PCT -gt 100 ]] && PCT=100

  # ASCII progress bar (40 chars wide)
  FILL=$(( PCT * 40 / 100 ))
  EMPTY=$(( 40 - FILL ))
  BAR=""
  for (( i=0; i<FILL;  i++ )); do BAR+="█"; done
  for (( i=0; i<EMPTY; i++ )); do BAR+="░"; done

  echo "╔══════════════════════════════════════════════════════════════╗"
  printf  "║  Hybrid Autoscaler — Live Trial Monitor   %-17s ║\n" "$(date -u '+%H:%M:%S UTC')"
  echo "╠══════════════════════════════════════════════════════════════╣"
  printf  "║  [%s] %3d%%                              ║\n" "$BAR" "$PCT"
  printf  "║  Elapsed %02dh%02dm%02ds    Remaining %02dh%02dm%02ds    Target %dh  ║\n" \
          "$E_H" "$E_M" "$E_S" "$R_H" "$R_M" "$R_S" "$DURATION_HOURS"
  echo "╠══════════════════════════════════════════════════════════════╣"

  # ── Latency ────────────────────────────────────────────────────────────────
  P50=$(_prom_scalar 'histogram_quantile(0.50,sum by(le)(rate(http_request_duration_seconds_bucket{namespace="default",pod=~"frontend-.*"}[1m])))*1000' "%.0f")
  P95=$(_prom_scalar 'histogram_quantile(0.95,sum by(le)(rate(http_request_duration_seconds_bucket{namespace="default",pod=~"frontend-.*"}[1m])))*1000' "%.0f")
  P99=$(_prom_scalar 'histogram_quantile(0.99,sum by(le)(rate(http_request_duration_seconds_bucket{namespace="default",pod=~"frontend-.*"}[1m])))*1000' "%.0f")
  printf "║  Latency (frontend)   p50 %-6s ms   p95 %-6s ms   p99 %-4s ms ║\n" \
         "$P50" "$P95" "$P99"

  # ── Throughput ─────────────────────────────────────────────────────────────
  RPS=$(_prom_scalar     'sum(rate(http_requests_total{namespace="default"}[1m]))'           "%.1f")
  ERR5=$(_prom_scalar    'sum(rate(http_requests_total{namespace="default",code=~"5.."}[1m]))' "%.2f")
  printf "║  Throughput           RPS  %-8s        5xx/s %-22s ║\n" "$RPS" "$ERR5"

  # ── CPU ────────────────────────────────────────────────────────────────────
  CPU=$(_prom_scalar 'sum(rate(node_cpu_seconds_total{mode!="idle"}[1m]))' "%.1f")
  printf "║  Node CPU (active)    %-6s cores                                  ║\n" "$CPU"

  echo "╠══════════════════════════════════════════════════════════════╣"
  echo "║  Replicas (available)                                        ║"
  _prom_table 'kube_deployment_status_replicas_available{namespace="default"}' "deployment" ""
  echo "╠══════════════════════════════════════════════════════════════╣"
  printf "║  Prometheus: %-47s ║\n" "$PROM_URL"
  printf "║  Grafana:    kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80\n" || true
  printf "║  Refresh every %ds   Ctrl-C to exit                        ║\n" "$INTERVAL"
  echo "╚══════════════════════════════════════════════════════════════╝"

  sleep "$INTERVAL"
done
