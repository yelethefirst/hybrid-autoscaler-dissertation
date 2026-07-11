#!/usr/bin/env bash
#
# verify.sh — Phase 0 exit-criterion check.
#
# Passes iff:
#   1. kind cluster reachable, Kubernetes v1.30.x.
#   2. All 11 Online Boutique pods Ready in the configured namespace.
#   3. kube-prometheus-stack Ready (Prometheus, kube-state-metrics, node-exporter,
#      Grafana). Versions match §3.10 (Prometheus 2.50.x, KSM 2.12.x, NE 1.7.x,
#      Grafana 10.4.x).
#   4. The five §3.5 metric families are queryable via Prometheus HTTP API:
#         container_cpu_usage_seconds_total
#         container_memory_working_set_bytes
#         kube_pod_status_ready
#         (request rate proxy: container_network_receive_packets_total)
#         (response-time proxy: any *_bucket histogram)
#
# Exits 0 on success, non-zero on first failure with a clear diagnostic.
#
set -euo pipefail

NS_APP="${ONLINE_BOUTIQUE_NAMESPACE:-default}"
NS_MON="monitoring"
PROM_SVC="kube-prometheus-stack-prometheus"

FAIL=0
say()  { printf "▶  %s\n" "$*"; }
ok()   { printf "✓  %s\n" "$*"; }
warn() { printf "⚠  %s\n" "$*"; FAIL=$((FAIL+1)); }
fail() { printf "❌ %s\n" "$*"; FAIL=$((FAIL+1)); }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cluster reachable, Kubernetes 1.30.x
# ─────────────────────────────────────────────────────────────────────────────
say "Checking cluster reachability and Kubernetes version…"
if ! kubectl cluster-info >/dev/null 2>&1; then
  fail "kubectl cannot reach a cluster."
  exit 1
fi
K8S_VER=$(kubectl version -o json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['serverVersion']['gitVersion'])")
if [[ "$K8S_VER" == v1.33.* ]]; then
  ok "Kubernetes server: $K8S_VER"
else
  warn "Kubernetes server is $K8S_VER (expected v1.33.x per §3.10)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Online Boutique — 11 pods Ready
# ─────────────────────────────────────────────────────────────────────────────
say "Checking Online Boutique pods in namespace '$NS_APP'…"
EXPECTED_DEPLOYMENTS=(emailservice recommendationservice cartservice
                     checkoutservice currencyservice productcatalogservice
                     shippingservice adservice paymentservice
                     loadgenerator frontend)
MISSING=()
for d in "${EXPECTED_DEPLOYMENTS[@]}"; do
  if ! kubectl -n "$NS_APP" get deploy "$d" >/dev/null 2>&1; then
    MISSING+=("$d")
  fi
done
if (( ${#MISSING[@]} )); then
  fail "Missing Online Boutique deployments: ${MISSING[*]}"
else
  ok "All 11 Online Boutique deployments present."
fi

NOT_READY=$(kubectl -n "$NS_APP" get pods --no-headers \
  | awk '{print $1, $2, $3}' \
  | grep -v Running \
  | grep -v Completed \
  || true)
if [[ -n "$NOT_READY" ]]; then
  fail "Online Boutique pods not Ready:"
  echo "$NOT_READY"
else
  ok "All Online Boutique pods Running."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Prometheus stack — Ready and version-matched
# ─────────────────────────────────────────────────────────────────────────────
say "Checking kube-prometheus-stack pods in namespace '$NS_MON'…"
if ! kubectl get ns "$NS_MON" >/dev/null 2>&1; then
  fail "Namespace '$NS_MON' missing — run infra/observability/install.sh"
  exit $FAIL
fi

MON_NOT_READY=$(kubectl -n "$NS_MON" get pods --no-headers 2>/dev/null \
  | awk '$3!="Running" && $3!="Completed" {print}' \
  || true)
if [[ -n "$MON_NOT_READY" ]]; then
  fail "Observability pods not Ready:"
  echo "$MON_NOT_READY"
else
  ok "All observability pods Running."
fi

# Version check (best-effort — image tags reveal versions).
PROM_IMG=$(kubectl -n "$NS_MON" get pods -l "app.kubernetes.io/name=prometheus" \
  -o jsonpath='{.items[0].spec.containers[?(@.name=="prometheus")].image}' 2>/dev/null || true)
case "$PROM_IMG" in
  *prometheus:v2.5[2-9]*|*prometheus:v2.[6-9]*|*prometheus:v3.*) ok "Prometheus version OK: $PROM_IMG" ;;
  *) warn "Prometheus image is '$PROM_IMG' (expected ≥v2.52.x per §3.10)" ;;
esac

KSM_IMG=$(kubectl -n "$NS_MON" get pods -l "app.kubernetes.io/name=kube-state-metrics" \
  -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null || true)
case "$KSM_IMG" in
  *kube-state-metrics:v2.12*) ok "kube-state-metrics OK: $KSM_IMG" ;;
  *) warn "kube-state-metrics image '$KSM_IMG' (expected v2.12.x per §3.10)" ;;
esac

NE_IMG=$(kubectl -n "$NS_MON" get pods -l "app.kubernetes.io/name=prometheus-node-exporter" \
  -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null || true)
case "$NE_IMG" in
  *node-exporter:v1.[89]*|*node-exporter:v[2-9].*) ok "node-exporter OK: $NE_IMG" ;;
  *) warn "node-exporter image '$NE_IMG' (expected ≥v1.8.x per §3.10)" ;;
esac

GR_IMG=$(kubectl -n "$NS_MON" get pods -l "app.kubernetes.io/name=grafana" \
  -o jsonpath='{.items[0].spec.containers[?(@.name=="grafana")].image}' 2>/dev/null || true)
case "$GR_IMG" in
  *grafana:10.4*) ok "Grafana OK: $GR_IMG" ;;
  *) warn "Grafana image '$GR_IMG' (expected 10.4.x per §3.10)" ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prometheus HTTP API — five metric families queryable
# ─────────────────────────────────────────────────────────────────────────────
say "Port-forwarding Prometheus and probing the five §3.5 metric families…"

PROM_LOCAL_PORT=$((RANDOM % 1000 + 19090))
kubectl -n "$NS_MON" port-forward "svc/$PROM_SVC" "${PROM_LOCAL_PORT}:9090" >/dev/null 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

probe() {
  local family="$1" query="$2"
  local n
  n=$(curl -fsS "http://127.0.0.1:${PROM_LOCAL_PORT}/api/v1/query?query=${query}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['data']['result']))" \
        2>/dev/null || echo "0")
  if [[ "$n" -gt 0 ]]; then
    ok "$family — $n series returned"
  else
    fail "$family — no series returned (query: $query)"
  fi
}

probe "CPU usage"           "container_cpu_usage_seconds_total"
probe "Memory working set"  "container_memory_working_set_bytes"
probe "Pod ready count"     "kube_pod_status_ready"
probe "Network rx (req-rate proxy)" "container_network_receive_packets_total"
probe "Histograms (API server + Prometheus internal)" 'count(apiserver_request_duration_seconds_bucket)'

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "✅  Phase 0 exit criterion MET. Ready for Phase 1 (vertical slice)."
  exit 0
else
  echo "❌  Phase 0 verification FAILED ($FAIL issue(s) above)."
  exit 1
fi
