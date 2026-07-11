#!/usr/bin/env bash
#
# run-forecasting.sh - Phase 3 end-to-end forecaster comparison.
#
# Reads the most recent telemetry Parquet under data/parquet/, runs walk-
# forward CV across all five Section 3.6 forecasters per service, applies the
# three-criterion selection rule, runs the H1 paired-t + Bonferroni-Holm
# comparison vs Seasonal Naive, and writes:
#
#   experiments/results/phase3_<UTC>_selection.csv        (per-service winners)
#   experiments/results/phase3_<UTC>_h1.csv               (H1 verdict table)
#   experiments/results/phase3_<UTC>_model_registry.yaml  (controller registry)
#   experiments/results/models/...                        (selected ML artefacts)
#
# Laptop development keeps XGBoost/LSTM grids intentionally small. Set
# FULL_GRID=1 for the full Section 3.6 grid used by dissertation measurements.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TELEM_DIR="${TELEM_DIR:-$REPO_ROOT/data/parquet}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiments/results}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$OUT_DIR/models}"
NAMESPACE="${NAMESPACE:-default}"
TARGET_METRIC="${TARGET_METRIC:-cpu}"
HORIZON_SECONDS="${HORIZON_SECONDS:-30}"
SAMPLE_INTERVAL_SECONDS="${SAMPLE_INTERVAL_SECONDS:-15}"
PERIOD_SECONDS="${PERIOD_SECONDS:-60}"
INFERENCE_LATENCY_TRIALS="${INFERENCE_LATENCY_TRIALS:-5}"
SEED="${SEED:-0}"
FULL_GRID="${FULL_GRID:-0}"
ALLOW_H1_FAILURE="${ALLOW_H1_FAILURE:-0}"
LSTM_DEVICE="${LSTM_DEVICE:-auto}"

mkdir -p "$OUT_DIR" "$ARTIFACT_DIR"
cd "$REPO_ROOT"

args=(
  --telemetry-dir "$TELEM_DIR"
  --out-dir "$OUT_DIR"
  --artifact-dir "$ARTIFACT_DIR"
  --namespace "$NAMESPACE"
  --target-metric "$TARGET_METRIC"
  --horizon-seconds "$HORIZON_SECONDS"
  --sample-interval-seconds "$SAMPLE_INTERVAL_SECONDS"
  --period-seconds "$PERIOD_SECONDS"
  --inference-latency-trials "$INFERENCE_LATENCY_TRIALS"
  --seed "$SEED"
  --lstm-device "$LSTM_DEVICE"
)

if [[ -n "${N_SPLITS:-}" ]]; then
  args+=(--n-splits "$N_SPLITS")
fi

if [[ -n "${RHO:-}" ]]; then
  args+=(--rho "$RHO")
fi

if [[ -n "${SIGMA_MAX:-}" ]]; then
  args+=(--sigma-max "$SIGMA_MAX")
fi

if [[ "$FULL_GRID" == "1" ]]; then
  args+=(--full-grid)
fi

if [[ "$ALLOW_H1_FAILURE" == "1" ]]; then
  args+=(--allow-h1-failure)
fi

if [[ "${SKIP_LSTM:-0}" == "1" ]]; then
  args+=(--skip-lstm)
fi

if [[ "${SKIP_SARIMA:-0}" == "1" ]]; then
  args+=(--skip-sarima)
fi

uv run python -m forecasting.run_forecasting "${args[@]}"
