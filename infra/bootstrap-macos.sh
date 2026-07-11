#!/usr/bin/env bash
#
# bootstrap-macos.sh — one-time host setup on macOS Apple Silicon.
# Installs Docker Desktop, kind v0.32.0, kubectl, helm, uv. Idempotent.
#
# Run once per developer machine. Re-run safe.
#
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Preconditions
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "❌  bootstrap-macos.sh is for macOS. For Linux/Windows/WSL2, see infra/README.md."
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" && "$ARCH" != "x86_64" ]]; then
  echo "❌  Unsupported architecture: $ARCH"
  exit 1
fi

# Default Apple Silicon (arm64). Intel Macs detected automatically.
KIND_BIN_ARCH="darwin-arm64"
[[ "$ARCH" == "x86_64" ]] && KIND_BIN_ARCH="darwin-amd64"

echo "▶  Detected: macOS / $ARCH"

# ─────────────────────────────────────────────────────────────────────────────
# Homebrew
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v brew >/dev/null 2>&1; then
  echo "▶  Installing Homebrew…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "✓  Homebrew already installed: $(brew --version | head -n1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Docker Desktop (provides Docker Engine 26.x + containerd 1.7)
# §3.10 specifies containerd 1.7 within Docker Engine 26.0.
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "▶  Installing Docker Desktop…"
  brew install --cask docker
  echo ""
  echo "⚠  ACTION REQUIRED:"
  echo "   1. Open Docker Desktop from /Applications."
  echo "   2. Settings → Resources → set CPUs ≥ 6, Memory ≥ 12 GiB, Swap ≥ 2 GiB."
  echo "   3. Wait for the whale icon to be steady (not animated), then re-run this script."
  exit 0
else
  echo "✓  Docker present: $(docker --version)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# kind v0.32.0 — pinned binary (NOT brew, because brew updates roll forward).
# v0.32 is the latest kind release and ships Kubernetes v1.33 as the default
# node image (§3.10). Do not downgrade — the node image digest we pin requires
# v0.32 or later (containerd config v4 support required by newer node images).
# ─────────────────────────────────────────────────────────────────────────────
KIND_VERSION="v0.32.0"
KIND_BIN="/usr/local/bin/kind"
[[ "$ARCH" == "arm64" ]] && KIND_BIN="/opt/homebrew/bin/kind"

if ! command -v kind >/dev/null 2>&1 || [[ "$(kind version 2>/dev/null | awk '{print $2}')" != "$KIND_VERSION" ]]; then
  echo "▶  Installing kind ${KIND_VERSION}…"
  TMP=$(mktemp)
  curl -fsSL -o "$TMP" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${KIND_BIN_ARCH}"
  chmod +x "$TMP"
  sudo mv "$TMP" "$KIND_BIN" 2>/dev/null || mv "$TMP" "$KIND_BIN"
else
  echo "✓  kind present: $(kind version)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# kubectl (Homebrew; client-vs-server skew is fine within ±1 minor)
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v kubectl >/dev/null 2>&1; then
  echo "▶  Installing kubectl…"
  brew install kubectl
else
  echo "✓  kubectl present: $(kubectl version --client 2>/dev/null | head -n1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Helm (for kube-prometheus-stack)
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v helm >/dev/null 2>&1; then
  echo "▶  Installing Helm…"
  brew install helm
else
  echo "✓  Helm present: $(helm version --short)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# uv (Python package manager; deterministic lockfile per §3.10)
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "▶  Installing uv…"
  brew install uv
else
  echo "✓  uv present: $(uv --version)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Python deps (uv reads pyproject.toml; creates .venv; bit-exact reproducible)
# ─────────────────────────────────────────────────────────────────────────────
echo "▶  Syncing Python deps with uv…"
( cd "$(dirname "$0")/.." && uv sync )

echo ""
echo "✅  Bootstrap complete."
echo ""
echo "Next steps:"
echo "  bash infra/kind/up.sh local        # bring up modest local cluster"
echo "  bash infra/online-boutique/install.sh"
echo "  bash infra/observability/install.sh"
echo "  bash infra/verify.sh"
