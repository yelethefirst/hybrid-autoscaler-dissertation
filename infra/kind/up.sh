#!/usr/bin/env bash
#
# up.sh — bring up a kind cluster from the chosen profile.
#
# Usage:
#   bash infra/kind/up.sh local        # modest laptop dev cluster
#   bash infra/kind/up.sh cloud        # at-§3.10 cluster for measured runs
#
# The node image is pinned by SHA256 digest in kind-local.yaml / kind-cloud.yaml
# per §3.10. No runtime digest resolution is performed by this script — the
# digest is committed to source and should be updated explicitly when the
# Kubernetes version changes.
#
set -euo pipefail

PROFILE="${1:-local}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$PROFILE" in
  local)
    CONFIG="$SCRIPT_DIR/kind-local.yaml"
    NAME="hybrid-autoscaler-local"
    ;;
  cloud)
    CONFIG="$SCRIPT_DIR/kind-cloud.yaml"
    NAME="hybrid-autoscaler-cloud"
    ;;
  *)
    echo "❌  Usage: $0 {local|cloud}"
    exit 1
    ;;
esac

echo "▶  Profile: $PROFILE"
echo "▶  Config:  $CONFIG"

# Tear down any pre-existing cluster of the same name to start clean.
if kind get clusters | grep -qx "$NAME"; then
  echo "▶  Existing cluster '$NAME' found — deleting for a clean start…"
  kind delete cluster --name "$NAME"
fi

# Bring up.
echo "▶  Creating cluster '$NAME'…"
kind create cluster --config "$CONFIG"

# Confirm context.
kubectl cluster-info --context "kind-$NAME"

# Wait for the system pods to settle.
echo "▶  Waiting for kube-system to settle…"
kubectl wait --for=condition=Ready pods --all -n kube-system --timeout=180s

echo ""
echo "✅  Cluster '$NAME' is up."
kubectl get nodes -o wide
echo ""
echo "Next: bash infra/online-boutique/install.sh"
