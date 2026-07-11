#!/usr/bin/env bash
#
# run-collection.sh — Phase 2 telemetry collection campaign.
#
# Modes:
#   bash bin/run-collection.sh synthetic       # generate synthetic telemetry, write Parquet
#   bash bin/run-collection.sh live <minutes>  # pull live Prometheus data
#
# Phase 2 deliverable: §3.5 long-format Parquet under data/parquet/.
#
set -euo pipefail

MODE="${1:-synthetic}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/parquet}"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

if [[ "$MODE" == "synthetic" ]]; then
  WORKLOAD="${WORKLOAD:-periodic}"
  DURATION="${DURATION:-1800}"          # 30 min synthetic = 120 samples
  SEED="${SEED:-0}"
  echo "▶  Generating synthetic telemetry: workload=$WORKLOAD duration=${DURATION}s seed=$SEED"
  uv run python -c "
from pathlib import Path
from datetime import datetime, timezone
from data.synthetic import generate
df = generate(workload='$WORKLOAD', duration_seconds=$DURATION, seed=$SEED)
out = Path('$OUT_DIR') / f'synthetic_{datetime.now(timezone.utc).strftime(\"%Y%m%dT%H%M%SZ\")}_{\"$WORKLOAD\"}.parquet'
df.to_parquet(out, index=False)
print(f'✅  wrote {len(df):,} rows → {out}')
"

elif [[ "$MODE" == "live" ]]; then
  MINUTES="${2:-30}"
  PROM_LOCAL_PORT="${PROM_LOCAL_PORT:-9090}"
  PROM_NS="${PROM_NS:-monitoring}"
  PROM_SVC="${PROM_SVC:-kube-prometheus-stack-prometheus}"

  # Ensure port-forward (same dance as run-controller.sh).
  if ! curl -fsS --max-time 2 "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1; then
    echo "▶  starting Prometheus port-forward on localhost:${PROM_LOCAL_PORT}…"
    kubectl -n "$PROM_NS" port-forward "svc/${PROM_SVC}" "${PROM_LOCAL_PORT}:9090" >/dev/null 2>&1 &
    PF_PID=$!
    trap "kill $PF_PID 2>/dev/null || true" EXIT
    for _ in {1..10}; do
      sleep 1
      curl -fsS --max-time 1 "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1 && break
    done
  fi

  echo "▶  Live collection: last ${MINUTES} minutes against http://localhost:${PROM_LOCAL_PORT}"
  uv run python -c "
from pathlib import Path
from datetime import datetime, timedelta, timezone
from prometheus_api_client import PrometheusConnect
from data.collect import PrometheusExporter, run_campaign
from data.collect.exporter import CampaignConfig

SERVICES = ['emailservice','recommendationservice','cartservice','checkoutservice',
            'currencyservice','productcatalogservice','shippingservice','adservice',
            'paymentservice','frontend']
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=$MINUTES)
client = PrometheusConnect(url='http://localhost:${PROM_LOCAL_PORT}', disable_ssl=True)
exp = PrometheusExporter(client)
cfg = CampaignConfig(services=SERVICES, output_dir=Path('$OUT_DIR'))
written = run_campaign(exp, cfg, start=start, end=end)
print(f'✅  wrote {len(written)} Parquet batches → {cfg.output_dir}')
"
else
  echo "❌  Usage: $0 {synthetic|live [minutes]}"
  exit 1
fi
