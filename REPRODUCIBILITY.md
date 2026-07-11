# Reproducibility Guide

This document describes how to reproduce every result in the dissertation from scratch.

**Author**: Omoyele Sodiq Olabode  
**Institution**: York St John University, 2026  
**License**: MIT (see [LICENSE](LICENSE))

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> hybrid-autoscaler
cd hybrid-autoscaler

# 2. Install Python dependencies (exact versions from uv.lock)
uv sync

# 3. Run the test suite (198 tests, ~18 s) and lint gate
uv run pytest tests/ -q
uv run ruff check . --exclude .venv

# 4. Spin up the local kind cluster
bash infra/kind/up.sh local
bash infra/online-boutique/install.sh
bash infra/observability/install.sh
bash infra/verify.sh   # must exit 0

# 5. Run Phase 1 proof (vertical slice)
MAX_TICKS=30 bash bin/run-controller.sh

# 6. Generate Phase 3 forecasting proof (synthetic data)
N_SPLITS=3 ALLOW_H1_FAILURE=1 bash bin/run-forecasting.sh

# 7. Full A/B experiment — MEASUREMENT VM ONLY (§3.10-A; wrk2 required):
#    a. provision host:      cd infra/terraform && terraform apply
#    b. build environment:   cd infra/ansible && uvx --from ansible-core \
#                              ansible-playbook -i inventory.ini provision.yml
#    c. telemetry campaigns: bash bin/run-campaign.sh <burst|ramp|periodic|trace>
#    d. model re-selection:  FULL_GRID=1 N_SPLITS=5 bash bin/run-forecasting.sh
#    e. regenerate plan:     uv run python experiments/trial_plans/generate_plan.py \
#                              --peak-users <calibrated>   # after the capacity probe
#    f. run the campaign:    uv run python -m experiments.supervisor \
#                              --plan experiments/trial_plans/canonical-v2.yaml
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11.x | via `uv` |
| uv | ≥ 0.4.0 | dependency manager |
| kind | v0.32.0 | Kubernetes in Docker |
| Docker Desktop | ≥ 4.30 | for kind nodes |
| kubectl | compatible with k8s v1.33 | |
| wrk2 | giltene/wrk2 @ 44a94c17 | built by `infra/ansible/provision.yml`; REQUIRED for measured runs (DEV-013: unavailable on macOS — pilot used hey) |
| OpenAI API key | — | required for Phase 7 narration; set `OPENAI_API_KEY` |

---

## Phase-by-Phase Reproduction

### Phase 0 — Cluster Setup
```bash
bash infra/kind/up.sh local
bash infra/online-boutique/install.sh
bash infra/observability/install.sh
bash infra/verify.sh
```
**Exit criterion**: `infra/verify.sh` exits 0. Output saved to `experiments/results/phase0-verify.txt`.

### Phase 1 — Vertical Slice
```bash
MAX_TICKS=30 bash bin/run-controller.sh
```
**Exit criterion**: `experiments/results/phase1-frontend.jsonl` contains ≥1 scale-up AND ≥1 scale-down, both in NOMINAL state.

### Phase 2 — Data Pipeline
```bash
bash bin/run-features.sh   # runs leakage check; exits non-zero if leakage detected
```
**Exit criterion**: exit code 0 on both synthetic and live telemetry.

### Phase 3 — Forecasting
```bash
# Development/proof (120 synthetic samples):
N_SPLITS=3 ALLOW_H1_FAILURE=1 bash bin/run-forecasting.sh

# Dissertation runs (real data, full grid):
# FULL_GRID=1 N_SPLITS=5 bash bin/run-forecasting.sh
```
**Exit criterion**: `experiments/results/phase3_*_h1.csv` shows ≥1 service with `significant_holm=True`.

### Phase 4 — Controller Hardening
```bash
MODEL_REGISTRY=experiments/results/phase3_*_model_registry.yaml \
  MAX_TICKS=15 bash bin/run-controller.sh controller/configs/frontend-phase4.yaml

MODEL_REGISTRY=experiments/results/phase3_*_model_registry.yaml \
  DRY_RUN=1 MAX_TICKS=5 \
  bash bin/run-controller.sh controller/configs/frontend-phase4-fault.yaml
```
**Exit criterion**: evidence bundles show (a) SARIMA loaded from registry, (b) 15 NOMINAL ticks, (c) 5 FALLBACK_FORECASTER_FAULT ticks with HPA-equivalent formula verified.

### Phase 5 — Experiment Harness
```bash
# Dry-run rehearsal (no subprocesses; validates the full 84-trial plan):
uv run python -m experiments.supervisor \
  --plan experiments/trial_plans/canonical-v2.yaml --dry-run

# Real run (measurement VM; cluster + wrk2 required — hey fallback is
# dev-only and must be requested explicitly with ALLOW_HEY_FALLBACK=1):
uv run python -m experiments.supervisor \
  --plan experiments/trial_plans/canonical-v2.yaml
```
**Exit criterion**: result JSONL populated with all 84 trials (2×4×10 A/B +
2 A/A burst pairs), no `error` fields — the supervisor exits non-zero
otherwise. The plan is GENERATED (`experiments/trial_plans/generate_plan.py`);
regenerate rather than hand-edit, and re-run after the capacity probe
(`--peak-users`) and the VM registry re-freeze. The pilot plan
`canonical-10-trials.yaml` is retained for provenance only (DEV-014..017).

### Phase 6 — SHAP Explainability
```python
from explain import Attributor, faithfulness_metrics
from forecasting.xgboost_model import XGBoostForecaster
# (see tests/test_explain.py for usage examples)
```
**Exit criterion**: `fm.passes == True` for XGBoost (insertion_auc > 0.6, deletion_auc > 0.2).

### Phase 7 — LLM Narrative
```python
import openai
from narrate import Narrator
narrator = Narrator(client=openai.OpenAI(), model="gpt-4o-mini")
# (see narrate/narrator.py for usage)
```
**Exit criterion**: FActScore ≥ 0.8 on n=30 narrative samples.

### Phase 8 — Statistical Analysis
```python
from analysis.stats import load_results, run_h0_latency, render_table
df = load_results("experiments/results/results_canonical-ab-10-trials.jsonl")
results = run_h0_latency(df)
print(render_table(results))
```
**Exit criterion**: all five hypotheses (H0–H4) have computed verdicts in the result table.

---

## Zenodo Archive

The Zenodo archive contains:
- This repository at the final commit SHA
- `experiments/results/` (all evidence bundles and model artefacts)
- `reproducibility/host-spec.json` (measurement host)
- Raw Parquet telemetry (separate upload; DOI in README)

DOI: _to be assigned after final measured runs_

---

## Deviations from Specification

See [docs/deviations.md](docs/deviations.md) for all recorded deviations and their impact on validity.

---

## Contact

For questions about reproducibility, open an issue on GitHub or contact the author at the email address in the dissertation front matter.
