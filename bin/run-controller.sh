#!/usr/bin/env bash
#
# run-controller.sh — convenience launcher for the hybrid controller.
#
# Assumes:
#   1. Phase 0 is up (kind cluster + Online Boutique + observability stack).
#   2. Prometheus is port-forwarded to localhost:9090 (this script does that).
#   3. The current kube context is the kind cluster (kind switches it on create).
#
# Usage:
#   bash bin/run-controller.sh                               # default config, SeasonalNaive
#   bash bin/run-controller.sh path/to/config.yaml          # alternative config
#   MODEL_REGISTRY=experiments/results/phase3_*_model_registry.yaml \
#     bash bin/run-controller.sh                            # Phase 4: load selected model
#   DRY_RUN=1 bash bin/run-controller.sh                   # log decisions, skip kubectl scale
#   MAX_TICKS=10 bash bin/run-controller.sh                # bounded run
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/controller/configs/frontend-local.yaml}"
MAX_TICKS="${MAX_TICKS:-}"
MODEL_REGISTRY="${MODEL_REGISTRY:-}"
DRY_RUN="${DRY_RUN:-}"
PROM_NS="${PROM_NS:-monitoring}"
PROM_SVC="${PROM_SVC:-kube-prometheus-stack-prometheus}"
PROM_LOCAL_PORT="${PROM_LOCAL_PORT:-9090}"

echo "▶  config: $CONFIG"
[[ -n "$MODEL_REGISTRY" ]] && echo "▶  model-registry: $MODEL_REGISTRY"
[[ -n "$DRY_RUN" ]]        && echo "▶  dry-run: enabled"

# Ensure Prometheus is reachable via port-forward.
if ! curl -fsS --max-time 2 "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1; then
  echo "▶  starting Prometheus port-forward on localhost:${PROM_LOCAL_PORT}…"
  kubectl -n "$PROM_NS" port-forward "svc/${PROM_SVC}" "${PROM_LOCAL_PORT}:9090" >/dev/null 2>&1 &
  PF_PID=$!
  trap "kill $PF_PID 2>/dev/null || true" EXIT
  for _ in {1..10}; do
    sleep 1
    curl -fsS --max-time 1 "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1 && break
  done
else
  echo "✓  Prometheus reachable on localhost:${PROM_LOCAL_PORT}"
fi

cd "$REPO_ROOT"

EXTRA=()
[[ -n "$MAX_TICKS" ]]      && EXTRA+=("--max-ticks"      "$MAX_TICKS")
[[ -n "$MODEL_REGISTRY" ]] && EXTRA+=("--model-registry" "$MODEL_REGISTRY")
[[ -n "$DRY_RUN" ]]        && EXTRA+=("--dry-run")

uv run python -m controller.main \
  --config "$CONFIG" \
  --prometheus-url "http://localhost:${PROM_LOCAL_PORT}" \
  "${EXTRA[@]}"
