#!/usr/bin/env bash
# install.sh — deploy DeathStarBench Social Network for the deep-topology
# portability demonstration (roadmap T5.3; supervisor-requested).
#
# ⚠  STATUS: prepared 2026-07-05, UNTESTED until the measurement VM exists —
#    DSB images are largely x86_64-only, which is precisely why this could
#    never run on the arm64 development Mac. First execution = smoke test.
#
# Scope guard: this is a QUALITATIVE portability demo (controller runs
# unmodified against a 28-service deep fan-out app; evidence bundles with
# SHAP captured). It is NOT a statistical A/B and is not pre-registered.
# Run it ONLY after the main campaign is archived (T5.1/T5.2).
#
# Usage (on the measurement VM, after the Online Boutique campaign):
#   bash infra/deathstarbench/install.sh          # deploy
#   bash infra/deathstarbench/install.sh delete   # tear down

set -euo pipefail

# Pinned upstream commit (master HEAD as of 2026-07-05; last release 2024-06-27).
DSB_REPO="https://github.com/delimitrou/DeathStarBench.git"
DSB_COMMIT="6ecb09706140"
NAMESPACE="socialnet"
CLONE_DIR="${DSB_CLONE_DIR:-$HOME/DeathStarBench}"

if [[ "${1:-}" == "delete" ]]; then
  helm uninstall social-network -n "$NAMESPACE" || true
  kubectl delete ns "$NAMESPACE" --ignore-not-found
  echo "✅  Social Network removed."
  exit 0
fi

# Free the cluster's headroom first: the kind-cloud profile cannot host both
# benchmarks under load. The Online Boutique campaign must be DONE + archived.
echo "⚠  Ensure the Online Boutique campaign is archived. Scaling OB to zero…"
read -r -p "   Scale all Online Boutique deployments to 0 replicas? [y/N] " ok
if [[ "$ok" == "y" ]]; then
  kubectl -n default get deploy -o name | xargs -I{} kubectl -n default scale {} --replicas=0
fi

# 1. Clone pinned
if [[ ! -d "$CLONE_DIR" ]]; then
  git clone "$DSB_REPO" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --all --quiet
git -C "$CLONE_DIR" checkout --quiet "$DSB_COMMIT"
echo "✓  DeathStarBench @ $DSB_COMMIT"

# 1b. Build DSB's bundled wrk2 fork (repo-root wrk2/). Verified 2026-07-05
#     against the pinned commit: BOTH social-network lua scripts (standard and
#     determinism variants) `require("socket")` — plain giltene/wrk2 has no
#     luasocket, so the demo load MUST use this fork + luarocks luasocket.
#     Kept as a separate binary (dsb-wrk2); the main campaign's wrk2 is untouched.
if [[ ! -x /usr/local/bin/dsb-wrk2 ]]; then
  echo "▶  Building DSB wrk2 fork (+ luasocket)…"
  sudo apt-get install -y -qq luarocks libssl-dev zlib1g-dev libluajit-5.1-dev
  sudo luarocks install luasocket
  make -C "$CLONE_DIR/wrk2" -j"$(nproc)"
  sudo cp "$CLONE_DIR/wrk2/wrk" /usr/local/bin/dsb-wrk2
fi
echo "✓  dsb-wrk2 ready: $(command -v dsb-wrk2)"

# 2. Deploy the Social Network helm chart
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"
helm upgrade --install social-network \
  "$CLONE_DIR/socialNetwork/helm-chart/socialnetwork" \
  -n "$NAMESPACE" --timeout 10m

echo "▶  Waiting for pods (timeout 10 min)…"
kubectl -n "$NAMESPACE" wait --for=condition=Ready pods --all --timeout=600s

# Guard: controller configs assume these exact Deployment names (the chart
# derives names from each subchart's .Values.name — verified 2026-07-05, but
# assumptions about upstream charts get checked, not trusted).
for target in nginx-thrift compose-post-service; do
  kubectl -n "$NAMESPACE" get deploy "$target" >/dev/null 2>&1 || {
    echo "❌  expected Deployment '$target' not found. Actual deployments:"
    kubectl -n "$NAMESPACE" get deploy -o name
    exit 1
  }
done

# 3. The controller needs CPU requests on its target deployments (ρ = request
#    × 0.5, same request-based derivation as the main experiment — DEV-020).
#    The upstream chart sets none, so pin the two demo targets explicitly.
for target in nginx-thrift compose-post-service; do
  kubectl -n "$NAMESPACE" set resources deploy/"$target" \
    --requests=cpu=100m,memory=128Mi --limits=cpu=200m,memory=256Mi
done

# 4. Initialise the social graph (required before any workload)
echo "▶  Initialising social graph…"
kubectl -n "$NAMESPACE" port-forward svc/nginx-thrift 8081:8080 >/tmp/dsb-pf.log 2>&1 &
PF=$!; sleep 3
# CLI verified at pinned commit: --graph --ip --port --limit; needs aiohttp.
uv run --with aiohttp python "$CLONE_DIR/socialNetwork/scripts/init_social_graph.py" \
  --graph=socfb-Reed98 --ip 127.0.0.1 --port 8081 || {
    echo "⚠  graph init failed — check $CLONE_DIR/socialNetwork/README.md for deps"; }
kill $PF 2>/dev/null || true

cat <<'NEXT'
✅  Social Network deployed.

Demo protocol (see infra/deathstarbench/README.md):
  1. Load:      dsb-wrk2 with the compose-post / mixed lua scripts
  2. Controller: bash bin/run-controller.sh controller/configs/dsb/nginx-thrift-dsb.yaml
  3. Evidence:   experiments/results/dsb-nginx-thrift.jsonl (SHAP per decision)
NEXT
