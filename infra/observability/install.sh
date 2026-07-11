#!/usr/bin/env bash
#
# install.sh — install kube-prometheus-stack into the current cluster.
#
# Chart version pinned to 58.7.2 which ships:
#   - Prometheus 2.52.x        (§3.10 — DEV-002: shipped v2.52 not v2.50)
#   - kube-state-metrics 2.12.x (§3.10)
#   - node-exporter 1.8.x       (§3.10 — DEV-003: shipped v1.8 not v1.7)
#   - Grafana 10.4.x            (§3.10)
#
# Re-runnable: helm upgrade --install. Verify pinned versions with infra/verify.sh.
#
set -euo pipefail

CHART_VERSION="58.7.2"
RELEASE_NAME="kube-prometheus-stack"
NAMESPACE="monitoring"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VALUES="$SCRIPT_DIR/values.yaml"
SVCMON="$SCRIPT_DIR/servicemonitor.yaml"

if ! kubectl cluster-info >/dev/null 2>&1; then
  echo "❌  kubectl can't reach a cluster. Run 'bash infra/kind/up.sh local' first."
  exit 1
fi

# Add / update repo (idempotent).
if ! helm repo list 2>/dev/null | grep -q prometheus-community; then
  echo "▶  Adding prometheus-community helm repo…"
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
fi
helm repo update prometheus-community >/dev/null

# Ensure namespace.
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"

# Install or upgrade.
echo "▶  Installing/upgrading kube-prometheus-stack (chart $CHART_VERSION)…"
helm upgrade --install "$RELEASE_NAME" prometheus-community/kube-prometheus-stack \
  --namespace "$NAMESPACE" \
  --version "$CHART_VERSION" \
  --values "$VALUES" \
  --wait \
  --timeout 10m

# Apply the Online Boutique ServiceMonitor.
echo "▶  Applying ServiceMonitor for Online Boutique services…"
kubectl apply -f "$SVCMON"

# Wait for Prometheus to be Ready.
echo "▶  Waiting for Prometheus and Grafana pods to become Ready…"
kubectl wait --for=condition=Ready pods \
  -n "$NAMESPACE" \
  -l "app.kubernetes.io/name=prometheus" --timeout=300s
kubectl wait --for=condition=Ready pods \
  -n "$NAMESPACE" \
  -l "app.kubernetes.io/name=grafana" --timeout=300s

echo ""
echo "✅  Observability stack installed."
echo ""

# Apply the A/B trial dashboard ConfigMap.
# The Grafana sidecar auto-imports any ConfigMap labelled grafana_dashboard=1.
DASHBOARD_CM="$SCRIPT_DIR/dashboards/ab-trial-configmap.yaml"
if [[ -f "$DASHBOARD_CM" ]]; then
  echo "▶  Applying Grafana dashboard ConfigMap..."
  kubectl apply -f "$DASHBOARD_CM"
else
  echo "⚠  Dashboard ConfigMap not found at $DASHBOARD_CM — skipping"
fi

echo ""
kubectl -n "$NAMESPACE" get pods
echo ""
echo "Port-forward Prometheus:"
echo "  kubectl -n $NAMESPACE port-forward svc/$RELEASE_NAME-prometheus 9090:9090"
echo ""
echo "Port-forward Grafana (default admin/admin — change before cloud runs):"
echo "  kubectl -n $NAMESPACE port-forward svc/$RELEASE_NAME-grafana 3000:80"
echo ""
echo "Terminal monitor (no browser needed):"
echo "  bash bin/watch-ab-trial.sh"
echo ""
echo "Next: bash infra/verify.sh"
