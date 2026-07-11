#!/usr/bin/env bash
# package-zenodo.sh — bundle the dissertation artefact for Zenodo upload.
#
# Creates a self-contained archive at zenodo-upload/hybrid-autoscaler-<sha>.zip
# containing:
#   - Source code (minus .venv, __pycache__, .DS_Store)
#   - uv.lock (exact dependency lock)
#   - experiments/results/ (selected output files)
#   - Model artefacts (experiments/results/models/)
#   - preregistration/hypotheses.yaml
#   - docs/deviations.md
#   - LICENSE
#   - README.md
#   - REPRODUCIBILITY.md
#
# Does NOT include:
#   - Raw Parquet telemetry (too large; uploaded separately to Zenodo)
#   - .venv/ (can be recreated from uv.lock)
#   - kind/Docker images (described by host-spec.json and pinned manifests)
#
# Usage:
#   bash bin/package-zenodo.sh [--dry-run]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN="${1:-}"
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")
OUT_DIR="$REPO_ROOT/zenodo-upload"
ZIP_NAME="hybrid-autoscaler-${GIT_SHA}.zip"
ZIP_PATH="$OUT_DIR/$ZIP_NAME"

echo "▶  packaging artefact (sha=$GIT_SHA)"
echo "▶  output: $ZIP_PATH"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "▶  DRY-RUN: would create $ZIP_PATH"
  exit 0
fi

mkdir -p "$OUT_DIR"

# Build the zip from the git-tracked files only (excludes .venv etc.)
cd "$REPO_ROOT"
git ls-files | \
  grep -v "^\.venv\|^__pycache__\|^\.DS_Store" | \
  zip -@ "$ZIP_PATH"

# Add generated artefacts not in git (model artefacts, host spec)
if [[ -d experiments/results/models ]]; then
  zip -r "$ZIP_PATH" experiments/results/models/ 2>/dev/null || true
fi
if [[ -f reproducibility/host-spec.json ]]; then
  zip "$ZIP_PATH" reproducibility/host-spec.json
fi

echo "✅  artefact packaged: $ZIP_PATH"
echo "    size: $(du -sh "$ZIP_PATH" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Upload $ZIP_PATH to Zenodo as a new version."
echo "  2. Add the Zenodo DOI to README.md and dissertation §3.13."
echo "  3. Tag the git commit: git tag -a v1.0.0 -m 'Zenodo upload'"
