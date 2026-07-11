#!/usr/bin/env bash
# run-campaign.sh — 5-hour telemetry collection campaign (§3.5).
#
# Generates continuous traffic against the Online Boutique cluster using Locust,
# keeps macOS awake, auto-restarts port-forwards, then exports all Prometheus
# telemetry to Parquet.
#
# Usage:
#   bash bin/run-campaign.sh burst            # looping burst (default)
#   bash bin/run-campaign.sh ramp             # looping ramp
#   bash bin/run-campaign.sh periodic         # extended sinusoidal
#   DURATION_HOURS=1 bash bin/run-campaign.sh burst   # short test run
#
# Requirements:
#   - kind cluster running + Online Boutique ready  (bash infra/verify.sh)
#   - uv environment installed  (uv sync)
#
set -euo pipefail

WORKLOAD="${1:-burst}"
if [[ -n "${DURATION_MINUTES:-}" ]]; then
  DURATION_SECONDS=$(( DURATION_MINUTES * 60 ))
  DURATION_HOURS=$(( DURATION_MINUTES / 60 ))
  [[ $DURATION_HOURS -eq 0 ]] && DURATION_HOURS=1   # minimum label for Locust
else
  DURATION_HOURS="${DURATION_HOURS:-5}"
  DURATION_SECONDS=$(( DURATION_HOURS * 3600 ))
  DURATION_MINUTES=$(( DURATION_HOURS * 60 ))
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROM_PORT="${PROM_PORT:-9090}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
PROM_NS="monitoring"
PROM_SVC="kube-prometheus-stack-prometheus"

# Map workload name → Locust shape file
case "$WORKLOAD" in
  burst)    SHAPE_FILE="$REPO_ROOT/experiments/workloads/campaign_burst.py" ;;
  ramp)     SHAPE_FILE="$REPO_ROOT/experiments/workloads/campaign_ramp.py" ;;
  periodic) SHAPE_FILE="$REPO_ROOT/experiments/workloads/periodic.py" ;;
  trace)    SHAPE_FILE="$REPO_ROOT/experiments/workloads/trace_replay.py" ;;
  *)
    echo "❌  Unknown workload '$WORKLOAD'. Use: burst | ramp | periodic | trace"
    exit 1
    ;;
esac

USER_FILE="$REPO_ROOT/experiments/workloads/user.py"

# trace replay needs its CSV; default to the checked-in Alibaba trace
if [[ "$WORKLOAD" == "trace" ]]; then
  export TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/experiments/traces/alibaba_rps_30m.csv}"
fi

# For periodic: set NUM_PERIODS large enough to cover the whole campaign
if [[ "$WORKLOAD" == "periodic" ]]; then
  export NUM_PERIODS=$(( DURATION_HOURS * 6 ))   # 6 periods/hour at 10 min/period
  export PERIOD_SECONDS=600
fi

START_TS=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_DIR="$REPO_ROOT/experiments/results/logs"
OUT_DIR="$REPO_ROOT/data/parquet/${WORKLOAD}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LOCUST_PID=""
PROM_PF_PID=""
FRONTEND_PF_PID=""
CAFFEINE_PID=""

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "▶  Shutting down..."
  [[ -n "$LOCUST_PID" ]]      && kill "$LOCUST_PID"      2>/dev/null || true
  [[ -n "$PROM_PF_PID" ]]     && kill "$PROM_PF_PID"     2>/dev/null || true
  [[ -n "$FRONTEND_PF_PID" ]] && kill "$FRONTEND_PF_PID" 2>/dev/null || true
  [[ -n "$CAFFEINE_PID" ]]    && kill "$CAFFEINE_PID"    2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── Sleep prevention (macOS dev host only; cloud VMs do not sleep) ─────────
if command -v caffeinate >/dev/null 2>&1; then
  CAFFEINE_SECS=$(( DURATION_SECONDS + 900 ))   # campaign + 15 min buffer
  caffeinate -dims -t "$CAFFEINE_SECS" &
  CAFFEINE_PID=$!
  echo "▶  caffeinate active for ${CAFFEINE_SECS}s (pid=$CAFFEINE_PID)"
fi

# ─── Port-forward helpers (auto-restart loops) ───────────────────────────────
_prom_pf_loop() {
  while true; do
    kubectl -n "$PROM_NS" port-forward "svc/$PROM_SVC" "${PROM_PORT}:9090" \
      >>"$LOG_DIR/prom-pf-${START_TS}.log" 2>&1 || true
    sleep 2
  done
}

_frontend_pf_loop() {
  while true; do
    kubectl -n default port-forward svc/frontend "${FRONTEND_PORT}:80" \
      >>"$LOG_DIR/frontend-pf-${START_TS}.log" 2>&1 || true
    sleep 2
  done
}

# Kill any existing port-forwards on these ports
lsof -ti :"$PROM_PORT"     | xargs kill -9 2>/dev/null || true
lsof -ti :"$FRONTEND_PORT" | xargs kill -9 2>/dev/null || true
sleep 1

echo "▶  Starting Prometheus port-forward → localhost:${PROM_PORT}"
_prom_pf_loop &
PROM_PF_PID=$!

echo "▶  Starting frontend port-forward → localhost:${FRONTEND_PORT}"
_frontend_pf_loop &
FRONTEND_PF_PID=$!

# Wait for both to be ready
echo "▶  Waiting for port-forwards to stabilise..."
for i in {1..20}; do
  sleep 1
  PROM_OK=false; FRONT_OK=false
  curl -fsS --max-time 1 "http://localhost:${PROM_PORT}/-/ready"  >/dev/null 2>&1 && PROM_OK=true
  curl -fsS --max-time 1 "http://localhost:${FRONTEND_PORT}/"     >/dev/null 2>&1 && FRONT_OK=true
  [[ "$PROM_OK" == true && "$FRONT_OK" == true ]] && break
done

if ! curl -fsS --max-time 1 "http://localhost:${PROM_PORT}/-/ready" >/dev/null 2>&1; then
  echo "❌  Prometheus not reachable on :${PROM_PORT}. Is the cluster running?"
  exit 1
fi
echo "✓  Prometheus ready at http://localhost:${PROM_PORT}"

if ! curl -fsS --max-time 2 "http://localhost:${FRONTEND_PORT}/" >/dev/null 2>&1; then
  echo "⚠   Frontend not responding yet — Locust will retry"
else
  echo "✓  Frontend ready at http://localhost:${FRONTEND_PORT}"
fi

# ─── Locust ───────────────────────────────────────────────────────────────────
echo ""
echo "▶  Starting Locust: workload=${WORKLOAD} duration=${DURATION_HOURS}h"
echo "   Shape file : $SHAPE_FILE"
echo "   User file  : $USER_FILE"
echo "   Target     : http://localhost:${FRONTEND_PORT}"
echo "   CSV log    : $LOG_DIR/locust-${WORKLOAD}-${START_TS}"
echo ""

cd "$REPO_ROOT"
uv run locust \
  -f "${SHAPE_FILE},${USER_FILE}" \
  --headless \
  --host "http://localhost:${FRONTEND_PORT}" \
  --users 500 \
  --spawn-rate 30 \
  --run-time "${DURATION_SECONDS}s" \
  --csv "$LOG_DIR/locust-${WORKLOAD}-${START_TS}" \
  --html "$LOG_DIR/locust-${WORKLOAD}-${START_TS}.html" \
  >>"$LOG_DIR/locust-${WORKLOAD}-${START_TS}.log" 2>&1 &
LOCUST_PID=$!

# ─── Progress monitor ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Campaign running.  DO NOT close this terminal."
echo ""
echo "  Workload  : ${WORKLOAD}"
echo "  Duration  : ${DURATION_HOURS} hours"
echo "  Started   : $(date '+%H:%M:%S %Z')"
echo "  Ends ~    : $(date -d "+${DURATION_HOURS} hours" '+%H:%M:%S %Z' 2>/dev/null || date -v "+${DURATION_HOURS}H" '+%H:%M:%S %Z')"
echo ""
echo "  Tail logs with:"
echo "    tail -f $LOG_DIR/locust-${WORKLOAD}-${START_TS}.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ELAPSED=0
CHECK_INTERVAL=300  # report every 5 minutes
while kill -0 "$LOCUST_PID" 2>/dev/null && [[ $ELAPSED -lt $DURATION_SECONDS ]]; do
  sleep "$CHECK_INTERVAL"
  ELAPSED=$(( ELAPSED + CHECK_INTERVAL ))
  REMAINING=$(( DURATION_SECONDS - ELAPSED ))
  MINS=$(( REMAINING / 60 ))
  echo "[$(date '+%H:%M:%S')]  elapsed=$(( ELAPSED/60 ))min  remaining=${MINS}min  locust=running"
done

# ─── Data extraction ─────────────────────────────────────────────────────────
echo ""
echo "▶  Locust finished. Extracting ${DURATION_HOURS}h of telemetry from Prometheus..."

# Ensure Prometheus port-forward is still alive
curl -fsS --max-time 2 "http://localhost:${PROM_PORT}/-/ready" >/dev/null 2>&1 || {
  echo "⚠  Prometheus port-forward lost — restarting..."
  kill "$PROM_PF_PID" 2>/dev/null || true
  _prom_pf_loop &
  PROM_PF_PID=$!
  sleep 5
}

OUT_DIR="$OUT_DIR" PROM_LOCAL_PORT="$PROM_PORT" \
  bash "$REPO_ROOT/bin/run-collection.sh" live "$DURATION_MINUTES"

echo ""
echo "✅  Campaign complete!"
echo "    Workload   : ${WORKLOAD}"
echo "    Duration   : ${DURATION_HOURS}h"
echo "    Parquet dir: ${OUT_DIR}"
echo "    Locust HTML: $LOG_DIR/locust-${WORKLOAD}-${START_TS}.html"
echo ""
echo "  Next step (when all 4 workloads collected):"
echo "    FULL_GRID=1 N_SPLITS=5 bash bin/run-forecasting.sh"

# ─── Auto-stop reminder (credit protection) ───────────────────────────────────
# Set AUTO_STOP=true to emit a delete reminder at campaign end.
#
# Hetzner bills per hour of server *existence* — powering off saves nothing.
# Only `terraform destroy` (or `hcloud server delete`) stops billing.
# This section detects the platform and prints the right command.
#
if [[ "${AUTO_STOP:-false}" == "true" ]]; then
  echo ""
  # Detect Hetzner via its metadata endpoint
  if curl -fsS --max-time 1 "http://169.254.169.254/hetzner/v1/metadata" >/dev/null 2>&1; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  AUTO_STOP: Campaign complete on Hetzner."
    echo "  Hetzner bills per hour of existence — stopping saves nothing."
    echo ""
    echo "  To stop ALL billing, run from your laptop:"
    echo "    cd infra/terraform && terraform destroy"
    echo "  or:"
    echo "    hcloud server delete hybrid-autoscaler-measurement"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  else
    # Fallback: EC2 IMDSv2
    IMDS_TOKEN=$(curl -fsS --max-time 2 \
      -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
      "http://169.254.169.254/latest/api/token" 2>/dev/null || true)
    if [[ -n "$IMDS_TOKEN" ]]; then
      SELF_ID=$(curl -fsS --max-time 2 \
        -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
        "http://169.254.169.254/latest/meta-data/instance-id" 2>/dev/null || true)
      SELF_REGION=$(curl -fsS --max-time 2 \
        -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
        "http://169.254.169.254/latest/meta-data/placement/region" 2>/dev/null || true)
      if [[ -n "$SELF_ID" && -n "$SELF_REGION" ]]; then
        echo "  Instance : $SELF_ID  Region: $SELF_REGION"
        echo "  Stopping in 120 s — press Ctrl-C NOW to cancel."
        sleep 120
        aws ec2 stop-instances --region "$SELF_REGION" --instance-ids "$SELF_ID" \
          && echo "  Stop request sent." \
          || echo "  ⚠  aws cli stop failed — stop the instance manually."
      else
        echo "  ⚠  Could not read instance metadata — stop the instance manually."
      fi
    else
      echo "  ⚠  Platform not detected — stop or delete the server manually."
    fi
  fi
fi
