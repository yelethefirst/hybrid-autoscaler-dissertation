#!/usr/bin/env bash
# tunnels.sh — open Grafana + Prometheus from the measurement VM in your local
# browser while campaigns/trials run (monitoring is read-only observation;
# Grafana is visualisation only, never a measurement source — §3.5).
#
# Usage:
#   bash bin/tunnels.sh <vm-public-ip> [ssh-key]
#
# Then open locally:
#   Grafana    → http://localhost:3000   (admin / admin per values.yaml)
#   Prometheus → http://localhost:9090
#
# How it works: starts kubectl port-forwards ON the VM (inside tmux so they
# survive your laptop disconnecting), then opens SSH -L tunnels from this
# machine to those forwards. Re-run after a VM stop/start (public IP changes).

set -euo pipefail

VM="${1:?usage: bash bin/tunnels.sh <vm-public-ip> [ssh-key]}"
KEY="${2:-$HOME/.ssh/id_ed25519}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new ubuntu@$VM"

echo "▶  Ensuring port-forwards on the VM (tmux session: monitoring)…"
$SSH 'tmux kill-session -t monitoring 2>/dev/null || true
tmux new-session -d -s monitoring \
  "kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80"
tmux split-window -t monitoring \
  "kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090"
tmux ls'

echo "▶  Opening local tunnels (Ctrl-C to close)…"
echo "   Grafana    → http://localhost:3000  (admin/admin)"
echo "   Prometheus → http://localhost:9090"
exec ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -N \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  "ubuntu@$VM"
