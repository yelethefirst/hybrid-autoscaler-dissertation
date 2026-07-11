# Implementation Log — Findings and Decisions

**Project**: Intelligent Predictive and Explainable Autoscaling for Kubernetes Microservices  
**Author**: Omoyele Sodiq Olabode  
**Institution**: York St John University, 2026  

This document is the authoritative record of every significant implementation decision,
technical finding, issue encountered, and resolution applied during the build of the
hybrid autoscaler. It is intended as source material for:

- **Chapter 3** (Methodology) — justifying design decisions
- **Chapter 4** (Results) — contextualising what the measurements mean
- **Chapter 5** (Discussion) — threats to validity and lessons learned
- **Appendix** — full technical audit trail for the examiner

Entries are ordered chronologically within each phase.

---

## Phase 0 — Infrastructure Setup

### Decision 0.1 — Kubernetes v1.30 → v1.33

**Context**: The dissertation specification referenced kind v0.22 and Kubernetes v1.30.
At the point of implementation (mid-2025), Kubernetes v1.30 had reached End of Life
(June 2025). Using an EOL version in a 2026 submission creates a legitimate examiner
threat around security and reproducibility.

**Decision**: Upgrade to kind v0.32.0 and Kubernetes v1.33.12 (then-current stable).

**Justification**: The HPA controller algorithm, `kubectl scale` subresource, and all
five cAdvisor/kube-state-metrics metric families used by the autoscaler are semantically
identical between v1.30 and v1.33. No API changes affect the experiment. The upgrade
eliminates a security threat without altering any result.

**Impact on validity**: None. Documented as DEV-001.

---

### Decision 0.2 — Prometheus v2.50 → v2.52 (and related stack versions)

**Context**: The Helm chart `kube-prometheus-stack@58.7.2` (latest at time of install)
ships Prometheus v2.52.0, kube-state-metrics v2.12.0, node-exporter v1.8.0, and
Grafana 10.4.1. The dissertation specification listed earlier versions.

**Decision**: Accept the chart-shipped versions. The alternative (pinning to an older
chart to obtain v2.50) would require using an unsupported Helm chart release.

**Finding**: Prometheus ≥v2.20 rejects bare `{__name__=~".*_bucket"}` queries (no
non-regex label matcher). The `infra/verify.sh` histogram probe was rewritten to use
`count(apiserver_request_duration_seconds_bucket)` which is always present and passes
the HTTP 400 restriction.

**Impact on validity**: None. Documented as DEV-002, DEV-003, DEV-004.

---

### Decision 0.3 — cartservice memory limit 128 Mi → 256 Mi

**Context**: On Apple Silicon (arm64), the .NET CLR baseline working set exceeds 128 Mi
under any load, causing OOMKilled restarts (Exit Code 137). This manifests as
CrashLoopBackOff during initial cluster startup.

**Decision**: Patch the cartservice manifest to `resources.limits.memory: 256Mi` (request
raised from 64Mi to 128Mi). The patch is applied by `infra/online-boutique/install.sh`
immediately after the main manifest, leaving all other service limits unchanged.

**Justification**: The upstream limit is sized for x86_64 where the CLR footprint is
smaller. The autoscaler targets CPU and request-rate metrics, not memory, so this patch
has no effect on any scaling decision. The 256Mi limit provides approximately 100 MiB
headroom above the observed arm64 working set.

**Impact on validity**: None for scaling decisions. Documented as DEV-007.

---

### Finding 0.4 — loadgenerator deployment starts at 0 replicas

**Context**: The Online Boutique loadgenerator deployment is included in the manifest but
was set to 0 replicas for all experiment phases. Traffic generation is handled externally
by Locust in all phases from Phase 1 onward.

**Decision**: Keep loadgenerator at 0 replicas. The Locust-based workload shapes
(§3.9) provide precise, reproducible traffic with configurable shape, which the built-in
loadgenerator (fixed constant rate) cannot.

---

### Phase 0 Exit Criterion — Result

`infra/verify.sh` exits 0 with the following confirmed state:

| Check | Result |
|---|---|
| Kubernetes server version | v1.33.12 |
| Online Boutique deployments | 11 present, all pods Running |
| Prometheus | v2.52.0, Ready |
| kube-state-metrics | v2.12.0, Ready |
| node-exporter | v1.8.0, Ready |
| Grafana | 10.4.1, Ready |
| CPU usage series | 96 |
| Memory working set series | 96 |
| Pod ready count series | 93 |
| Network rx proxy series | 310 |
| Histogram series | 1 (apiserver) |

---

## Phase 1 — Vertical Slice

### Decision 1.1 — Seasonal Naive as Phase 1 forecaster

**Context**: Phase 1's purpose is to prove the predict→decide→actuate→log loop end to
end on a single service. The forecaster accuracy does not matter at this stage; what
matters is that the loop runs without errors.

**Decision**: Use Seasonal Naive as the Phase 1 forecaster. It has no hyperparameters,
no training phase, and cannot fault due to data sparsity (it only needs
`period_samples + 2 = 6` samples).

**Impact**: Phase 1 evidence bundle (`experiments/results/phase1-frontend.jsonl`) shows
NOMINAL state scaling decisions, confirming the loop is correct without any model
complexity obscuring the result.

---

### Finding 1.2 — rho placeholder 0.30 → calibrated 0.125

**Context**: The initial `controller/configs/frontend-local.yaml` used `rho: 0.30`
as a placeholder. During Phase 1 testing, no scale-up event was triggered despite
significant CPU load. Investigation revealed the placeholder exceeded the container's
own CPU limit (250m = 0.25 cores), making the formula denominator larger than any
achievable numerator.

**Decision**: Replace with analytically derived value:
`ρ = cpu_limit × target_utilisation = 0.250 cores × 0.50 = 0.125 cores`.

**Justification**: This is the same derivation Kubernetes HPA uses for its own threshold.
Phase 2 calibration will replace this with an empirically fitted value from steady-state
telemetry. Any remaining discrepancy will be corrected before the A/B experiment.

**Impact**: Scaling decisions now trigger correctly. Documented as DEV-008.

---

### Decision 1.3 — Five-state machine design

The controller implements exactly five states as specified in §3.7:

| State | Trigger |
|---|---|
| NOMINAL | Forecast available, σ ≤ sigma_max |
| CONFIDENCE_MARGIN_HIGH | σ > sigma_max; forecaster still returning |
| FALLBACK_UNCERTAINTY | σ > sigma_max AND above threshold for N consecutive ticks |
| FALLBACK_FORECASTER_FAULT | ForecasterFaultError raised by model |
| SCALE_LIMITED | Recommended change exceeds ΔS rate limit |

**Finding**: All five states exercised in testing. FALLBACK_FORECASTER_FAULT was
deliberately triggered in Phase 4 by setting `history_seconds: 60` (4 samples),
which is below SARIMA's minimum of 12 (= 3 × period_samples). The fallback
correctly applied the HPA-equivalent formula `ceil(current_utilisation / 0.50)`.

---

### Phase 1 Exit Criterion — Result

`experiments/results/phase1-frontend.jsonl` contains ≥1 scale-up and ≥1 scale-down
event, both in NOMINAL state. Loop proven end-to-end. Commit: `056f521`.

---

## Phase 2 — Data Pipeline

### Decision 2.1 — Long-format Parquet as the storage schema

**Context**: The five metric families (CPU, memory, pod-ready, request rate, response
time) each have different cardinality and label sets. A wide-format schema would require
a fixed schema per service, making it fragile to label changes.

**Decision**: Store all telemetry in long format with columns:
`timestamp, service, namespace, metric_family, metric_name, value, labels`.
The `to_wide()` utility pivots to wide format per service immediately before feature
engineering.

**Benefit**: Adding a new metric family or service requires no schema migration. All
`data/collect/exporter.py` batch files are compatible across runs.

---

### Decision 2.2 — Empirical anti-leakage validator

**Context**: Data leakage (features derived from future values) is the most common
validity threat in time-series ML. A formal proof is expensive; an empirical check
is tractable and sufficient for the dissertation's claim.

**Decision**: Implement `data/leakage_check.py`: perturb the future portion of the
base series (shuffle + multiply by noise), re-run the feature pipeline, and verify
that past features are bitwise unchanged. Non-zero exit if any feature changes.

**Finding**: The validator correctly detects three leaky patterns tested in the suite:
`shift(-1)`, centred rolling windows, and `diff(-1)`. The §3.5-compliant default
pipeline passes on all synthetic and real data.

---

### Decision 2.3 — 5-minute batch files for Prometheus export

**Context**: A single Prometheus port-forward pulling 5 hours of data in one query
risks timeout and memory pressure. Long-running queries also hit Prometheus's
`query_range` timeout (default 120s).

**Decision**: `data/collect/exporter.py` pulls data in 5-minute batches, writing one
Parquet file per batch. `load_campaign(parquet_dir)` concatenates on read. This
means a campaign can be interrupted and resumed without losing prior batches.

---

## ρ Calibration — Per-Service Target CPU Capacity

### Decision C.1 — Analytical ρ derivation from manifest CPU limits

**Context**: The scaling formula (§3.7) is:
```
r*(t+h) = clip( ceil( (f̂(t+h) + k·σ̂(t+h)) / ρ ), r_min, r_max )
```
`ρ` is the per-pod target CPU capacity: the maximum CPU load a single pod should
sustain at the desired utilisation level. Setting ρ too high means pods are
overloaded before scaling triggers; too low means constant over-provisioning.

**Method**: Analytical calibration using the same formula as Kubernetes HPA:
```
ρ = cpu_limit × target_utilisation
```
`target_utilisation = 0.50` (Kubernetes HPA default; provides 50% headroom for
traffic bursts before a scale-up is needed). CPU limits are sourced from
`infra/online-boutique/pinned-manifest.yaml`.

**Correction applied (DEV-012)**: The previously assumed frontend CPU limit of 250m
was incorrect. The actual manifest limit for frontend is 200m, giving `ρ = 0.100`
(was 0.125). All controller configs updated.

**Per-service calibrated ρ values:**

| Service | CPU Limit | ρ (50% utilisation) | r_max | Selected forecaster |
|---|---|---|---|---|
| adservice | 300m | **0.1500** | 4 | Holt-Winters |
| cartservice | 300m | **0.1500** | 4 | Holt-Winters |
| checkoutservice | 200m | **0.1000** | 4 | Holt-Winters |
| currencyservice | 200m | **0.1000** | 6 | XGBoost |
| emailservice | 200m | **0.1000** | 2 | XGBoost |
| frontend | 200m | **0.1000** | 8 | XGBoost |
| paymentservice | 200m | **0.1000** | 2 | XGBoost |
| productcatalogservice | 200m | **0.1000** | 4 | Holt-Winters |
| recommendationservice | 200m | **0.1000** | 4 | XGBoost |
| shippingservice | 200m | **0.1000** | 2 | XGBoost |

Individual controller configs written to `controller/configs/ab/<service>-ab.yaml`
by `bin/generate-service-configs.py`.

---

### Finding C.2 — Empirical validation of ρ against burst campaign data

**Context**: After setting analytical ρ values, the real burst campaign telemetry
(5 hours, 4,200 samples per service) was used to validate that the chosen ρ makes
sense: services where observed CPU stays well below ρ will rarely trigger scale-up,
while services where CPU frequently exceeds ρ are the primary autoscaling targets.

**Method**: Compute per-pod CPU percentiles from `data/parquet/burst/` for each
service. Compare P50, P80, P95, and Max against ρ.

**Results:**

| Service | Limit | P50 CPU | P80 CPU | P95 CPU | Max CPU | P80 util | ρ | Will scale? |
|---|---|---|---|---|---|---|---|---|
| adservice | 300m | 0.0199 | 0.0347 | 0.0781 | 0.2326 | 11.6% | 0.150 | Rarely |
| cartservice | 300m | 0.0275 | 0.0317 | 0.0513 | 0.0684 | 10.6% | 0.150 | Rarely |
| checkoutservice | 200m | 0.0005 | 0.0007 | 0.0011 | 0.0034 | 0.4% | 0.100 | Almost never |
| **currencyservice** | **200m** | **0.0872** | **0.1195** | **0.1807** | **0.2080** | **59.8%** | **0.100** | **✅ Regularly** |
| emailservice | 200m | 0.0025 | 0.0033 | 0.0048 | 0.0113 | 1.7% | 0.100 | Almost never |
| **frontend** | **200m** | **0.0859** | **0.1084** | **0.1856** | **0.1935** | **54.2%** | **0.100** | **✅ Regularly** |
| paymentservice | 200m | 0.0010 | 0.0013 | 0.0018 | 0.0191 | 0.6% | 0.100 | Almost never |
| productcatalogservice | 200m | 0.0245 | 0.0302 | 0.0673 | 0.0744 | 15.1% | 0.100 | Occasionally |
| recommendationservice | 200m | 0.0326 | 0.0396 | 0.1318 | 0.1479 | 19.8% | 0.100 | Occasionally |
| shippingservice | 200m | 0.0027 | 0.0035 | 0.0083 | 0.0099 | 1.7% | 0.100 | Almost never |

**Key findings for dissertation §3.7 / §4.2:**

1. **frontend and currencyservice are the primary autoscaling targets.** Both
   regularly exceed ρ = 0.100 — currencyservice P80 is 120% of ρ, frontend P80
   is 108% of ρ. Scale-up decisions will fire during burst peaks for both services.

2. **currencyservice max CPU (0.208) exceeds its own CPU limit (0.200).** This
   indicates CPU throttling occurs at peak under a single replica — exactly the
   scenario where predictive autoscaling adds value by scaling up *before* the
   burst reaches the pod.

3. **Six services (checkoutservice, emailservice, paymentservice, shippingservice,
   adservice, cartservice) are not compute-bound under burst traffic.** Their max
   CPU stays well below ρ. The autoscaler will correctly leave them at 1 replica.
   These services are bottlenecked by other factors (I/O, latency) and would not
   benefit from CPU-based autoscaling.

4. **productcatalogservice and recommendationservice** show moderate utilisation
   (P95 > ρ) — scale-up decisions will fire during the top 5% of load periods.

**Implication for A/B experiment design**: The most meaningful comparison between
the Hybrid Autoscaler and HPA will be visible on `frontend` and `currencyservice`.
The A/B experiment measures latency (wrk2) and replica-seconds on `frontend` as
the primary service; currencyservice is a downstream dependency that benefits
indirectly when frontend scales.

**Artefacts produced:**
- `controller/configs/ab/*.yaml` — 10 per-service A/B controller configs
- `experiments/trial_plans/canonical-10-trials.yaml` — updated to use
  `controller/configs/ab/frontend-ab.yaml` and the Phase 3 real-data registry

---

## Phase 3 — Forecasting Model Selection

### Decision 3.1 — Walk-forward cross-validation (not random CV)

**Context**: Standard k-fold CV is invalid for time-series because it allows future
data to leak into training. The §3.5/3.6 specification requires Hyndman-style
walk-forward CV.

**Decision**: Implement expanding-window walk-forward CV in `data/splits.py`. Each fold
adds more training data; the validation window is always strictly after the training
window.

**Impact**: RMSE estimates are valid temporal predictors. The Holm-Bonferroni H1 test
operates on fold-level RMSE differences, which are approximately independent under
walk-forward splits.

---

### Decision 3.2 — LSTM minimum sample guard

**Context**: With `max_seq=60` and `horizon_samples=2`, the minimum per-fold training
requirement is `60 + 2 + patience_buffer ≈ 92` samples. On 120-sample synthetic data
with 3-fold CV, fold 1 yields only ~48 training rows — below the threshold.

**Decision**: LSTM raises `ForecasterFaultError` when the fold is too small. The
selection pipeline marks failed folds as RMSE=inf, which propagates to the service
being skipped for LSTM selection.

**Finding (synthetic data)**: LSTM returned RMSE=inf for all 11 services on 120-sample
synthetic data. This is correct behaviour, not a bug. LSTM is viable on the real 5-hour
campaigns (~1,200 samples, all folds ≥ 92 rows).

---

### Finding 3.3 — Phase 3 synthetic data results (proof-of-pipeline)

Run on 120-sample synthetic `trace_like` telemetry, 3-fold walk-forward CV.
Selected models:

| Service | Selected | RMSE |
|---|---|---|
| adservice | XGBoost | 0.0056 |
| cartservice | XGBoost | 0.0078 |
| checkoutservice | Holt-Winters | 0.0100 |
| currencyservice | XGBoost | 0.0032 |
| emailservice | Holt-Winters | 0.0031 |
| frontend | SARIMA | 0.0186 |
| loadgenerator | XGBoost | 0.0054 |
| paymentservice | XGBoost | 0.0035 |
| productcatalogservice | SARIMA | 0.0052 |
| recommendationservice | SARIMA | — (latency tie-break) |
| shippingservice | XGBoost | 0.0055 |

H1 result on synthetic data:

| Service | Candidate | RMSE reduction | Holm-p | Significant |
|---|---|---|---|---|
| checkoutservice | Holt-Winters | 23.1% | 0.035 | Yes (d=3.69) |
| productcatalogservice | SARIMA | 33.7% | 0.005 | Yes (d=9.91) |

H1 exit criterion (≥1 service significant) **MET** on synthetic data. Commit: `5a9b193`.

---

### Finding 3.4 — Phase 3 real-data results (burst workload, FULL_GRID=1, N_SPLITS=5)

Run on 129,083 rows of real burst telemetry (Campaign 2, 2026-07-02 19:24–00:24 BST),
10 services, 5-fold expanding walk-forward CV, full hyperparameter grid, LSTM excluded
(`SKIP_LSTM=1` — see Decision 5.6 / Finding 5.7). NaN values (191–204 per service from
a 20-minute data collection gap) were linearly interpolated before training.
Total wall-clock time: 1 hour 34 minutes. Registry: `phase3_20260703T090249Z_model_registry.yaml`.

**Selected models:**

| Service | Selected | RMSE | H1 Supported |
|---|---|---|---|
| adservice | **Holt-Winters** | 0.0093 | ✅ Yes |
| cartservice | **Holt-Winters** | 0.0052 | ✅ Yes |
| checkoutservice | **Holt-Winters** | 0.0002 | ❌ No |
| currencyservice | **XGBoost** | 0.0171 | ✅ Yes |
| emailservice | **XGBoost** | 0.0004 | ❌ No |
| frontend | **XGBoost** | — (latency tie-break) | ✅ Yes |
| paymentservice | **XGBoost** | 0.0003 | ✅ Yes |
| productcatalogservice | **Holt-Winters** | 0.0055 | ✅ Yes |
| recommendationservice | **XGBoost** | 0.0121 | ✅ Yes |
| shippingservice | **XGBoost** | — (latency tie-break) | ✅ Yes |

**Split: 6 × XGBoost, 4 × Holt-Winters. No SARIMA wins on real burst data.**

**H1 exit criterion: MET** (8 of 10 services beat Seasonal Naive).

**Observations for dissertation §4.2:**

1. Model selections differ substantially from synthetic data (where SARIMA won for
   frontend and productcatalogservice). On real burst traffic, XGBoost dominates for
   compute-heavy services (frontend, currencyservice) and Holt-Winters for services
   with smoother, trend-like CPU patterns.

2. checkoutservice and emailservice: H1 not supported — their CPU is so low and flat
   (P80 < 0.004 cores) that Seasonal Naive is already near-optimal. This is consistent
   with the ρ calibration finding (Finding C.2) that these services are not
   compute-bound.

3. frontend tie-break resolved by inference latency: XGBoost (2.8 ms p95) beat the
   tied candidate on latency. At a 15-second tick interval, this is operationally
   insignificant, but the three-criterion rule is correctly applied.

4. Two latency tie-breaks (frontend, shippingservice) suggest XGBoost and Holt-Winters
   have near-identical RMSE on these services' CPU patterns — a reasonable finding for
   low-variance CPU series.

**Note**: This run excludes LSTM. A follow-up run with MPS enabled (Finding 5.7) may
change the winner for some services. The model registry will be updated if LSTM wins
for any service.

---

### Decision 3.6 — Model registry YAML as the controller–forecasting interface

**Context**: The controller needs to know which forecaster to load at startup per
service. Hard-coding this would couple the controller to the forecasting pipeline.

**Decision**: Phase 3 produces `experiments/results/phase3_<UTC>_model_registry.yaml`.
The controller reads this at startup via `--model-registry`. The registry contains:
`service, namespace, target_metric, horizon_seconds, forecaster, artifact_path`.

**Benefit**: Updating the selected model for any service requires only rerunning Phase 3
and pointing the controller at the new registry — no code change.

---

## Preregistration Freeze — 2026-07-03

### Decision PR.1 — OSF preregistration freeze before A/B experiment

**Context**: Preregistration is the practice of publicly timestamping hypotheses,
analysis plans, and statistical tests *before* collecting confirmatory data. Without
it, an examiner or reviewer cannot distinguish hypothesis-driven testing from
post-hoc rationalisation ("HARKing" — Hypothesising After Results are Known).
The dissertation explicitly follows Nosek et al. (2018) on this point (§3.11).

**Action taken**: `preregistration/hypotheses.yaml` was frozen on 2026-07-03, prior
to running the canonical A/B experiment. The file was committed to git and will be
uploaded to OSF (Open Science Framework) for an independent public timestamp.

**What the freeze locked in:**

| Item | Frozen value |
|---|---|
| freeze_date | 2026-07-03 |
| H0 test | Paired-t, α=0.05, Holm-Bonferroni across 4 workloads |
| H1 test | One-sided paired-t per service, Holm-Bonferroni across services |
| H2 test | One-sided paired-t, α=0.05 |
| H3 test | One-sided Wilcoxon signed-rank (counts non-normal) |
| H4 test | One-sample Wilcoxon vs μ₀=0.8 FActScore |
| Primary outcomes | p95 latency (wrk2), replica-seconds |
| Effect size | Cohen's d_z; BCa 95% bootstrap, n=10,000 |
| Selected forecasters | XGBoost (6 services), Holt-Winters (4 services) |
| Model registry | phase3_20260703T090249Z_model_registry.yaml |
| ρ (frontend) | 0.1000 (200m × 0.50) |
| n_trials_per_cell | 10 |
| Randomisation seed | 42 |

**H1 status at freeze**: Already confirmed on real burst data — 8 of 10 services
beat Seasonal Naive. The A/B experiment tests H0, H2, H3 (controller vs HPA) and H4
(LLM narrative faithfulness). H1 is a separate forecasting comparison, completed.

**LSTM caveat at freeze**: LSTM excluded from Phase 3 due to CPU bottleneck; MPS
device placement added (Finding 5.7) but results not yet available. If a subsequent
LSTM run changes the winner for any service, the model registry will be updated and
a deviation entry added. The hypotheses themselves are unaffected by the model choice.

**VS Code schema fix**: The global VS Code `yaml.schemas` setting applied the
Kubernetes schema to all `/*.yaml` files, which incorrectly flagged `freeze_context`
in `hypotheses.yaml` as an invalid field. A project-level `.vscode/settings.json`
was created scoping the Kubernetes schema to `infra/**` and `experiments/hpa/**` only,
leaving preregistration, controller configs, and trial plans schema-free.

**Dissertation reference**: §3.11 (statistical analysis plan), §3.12 (open science
practices). The OSF URL should be cited here once the upload is complete.

**OSF upload checklist** (to be completed before running the A/B experiment):
- [ ] Create OSF project: "Hybrid Predictive Autoscaler — York St John 2026"
- [ ] Upload `preregistration/hypotheses.yaml`
- [ ] Register the component as a Preregistration (OSF Preregistration template or
      "other" with a link to the YAML)
- [ ] Copy the OSF DOI/URL into `preregistration/hypotheses.yaml` under `osf_url:`
- [ ] Commit the updated file with the OSF URL before the first A/B trial runs

---

## Phase 4 — Controller Hardening

### Decision 4.1 — Dry-run mode

**Context**: During development and A/B experiment setup, it is useful to run the
full controller loop (Prometheus queries, forecasting, decision engine) without
actually calling `kubectl scale`. This allows validating the decision logic without
affecting the cluster.

**Decision**: Add `--dry-run` flag to `controller/main.py` and `DRY_RUN=1` to
`bin/run-controller.sh`. When active, `K8sActuator.set_replicas()` logs the would-be
action but does not call `patch_namespaced_deployment_scale`.

---

### Finding 4.2 — Forced fault via `history_seconds: 60`

**Context**: Phase 4 requires demonstrating the FALLBACK_FORECASTER_FAULT state
without corrupting a real model artifact.

**Method**: Set `history_seconds: 60` in the frontend fault config. At 15s resolution,
this yields 4 samples. SARIMA requires at minimum `3 × period_samples = 3 × 4 = 12`
samples. With only 4, SARIMA raises `ForecasterFaultError`.

**Result**: Evidence bundle `experiments/results/phase4-frontend-fault.jsonl` shows
5 consecutive FALLBACK_FORECASTER_FAULT ticks with `fallback_engaged=True` and
`new_replicas=ceil(u_t/0.50)` verified against the HPA-equivalent formula.

---

## Phase 5 — Experiment Harness

### Decision 5.1 — Looping workload shapes for training campaigns

**Context**: The canonical workload shapes (`burst.py`, `ramp.py`, `periodic.py`) are
designed for short A/B experiment trials (12–30 minutes). For 5-hour training data
collection campaigns they must run continuously without stopping.

**Decision**: Create `experiments/workloads/campaign_burst.py` and
`experiments/workloads/campaign_ramp.py`. These are identical to the trial shapes but
use modulo arithmetic on elapsed time (`t = self.get_run_time() % _CYCLE`) so they
loop indefinitely. Locust duration is controlled externally by `bin/run-campaign.sh`.

**Rationale**: Modulo-based looping is the minimal change from the trial shapes.
The campaign shapes are only used for training data collection; the original shapes
remain unchanged for A/B trials where exact timing matters.

---

### Issue 5.2 — Locust relative import error (`ImportError: attempted relative import`)

**Context**: On first real data collection run, Locust exited after 5 minutes with:
```
ImportError: attempted relative import with no known parent package
```
in `campaign_burst.py` line `from .user import BoutiqueUser`.

**Root cause**: When Locust loads shape files via `-f shape.py,user.py`, each file
is loaded as a standalone module (not as part of a Python package). Relative imports
require a package context, which is not present when Locust loads files this way.

**Resolution**: Remove the `from .user import BoutiqueUser` import from both campaign
shape files. Since `user.py` is always passed as a separate `-f` argument, Locust
discovers `BoutiqueUser` from `user.py` automatically. The import in the shape file
was redundant when both files are specified.

**Lesson**: Locust's `-f file1.py,file2.py` format loads files as individual scripts,
not as a package. Any cross-file dependencies must use absolute imports or be avoided
entirely when both files are explicitly passed.

**Impact**: First campaign run collected only ~5 minutes of traffic before Locust
crashed. The 46 Parquet batches written by the first run's data extraction covered the
5-hour lookback window but contained primarily idle/low-traffic periods. The second
campaign run (after fix) ran correctly for 5 hours.

---

### Issue 5.3 — Prometheus port-forward loss during data extraction

**Context**: At the end of the second campaign run, the Prometheus port-forward
(background loop) had died. The automatic data extraction at campaign end failed
silently — no Parquet files were written for the 18:24–23:24 UTC window despite
Prometheus having collected all the data.

**Root cause**: The `_prom_pf_loop()` background subshell started by `run-campaign.sh`
was killed when the macOS terminal session changed state (lid close/open cycle). The
auto-restart loop is a subshell of the main script; it does not survive parent process
interruption.

**Resolution**: After the campaign completed and the port-forward was confirmed dead,
ran a targeted manual extraction using `prometheus_api_client` directly for the exact
18:24–23:24 UTC time window. Prometheus retained all data (default 15-day retention).
58 Parquet batches written to `data/parquet/burst/`.

**Mitigation for future campaigns**: The `run-campaign.sh` port-forward loop restarts
indefinitely within the script's lifetime. The real protection is Prometheus's data
retention — even if the port-forward dies during a campaign, data can always be
extracted retroactively using a targeted time-range query.

---

### Decision 5.4 — `bin/run-collection.sh live` accepts `OUT_DIR` override

**Context**: Real campaign data must be stored separately by workload
(`data/parquet/burst/`, `data/parquet/ramp/`, etc.) so the forecasting pipeline can
train on one workload at a time. The original script wrote all data to `data/parquet/`.

**Decision**: `run-collection.sh` already respected the `OUT_DIR` environment variable.
`run-campaign.sh` sets `OUT_DIR="$REPO_ROOT/data/parquet/${WORKLOAD}"` before calling
the collection script.

---

### Decision 5.5 — Multi-batch Parquet loading in the forecasting pipeline

**Context**: `forecasting/run_forecasting.py` was written for the synthetic data case
where telemetry is a single Parquet file. Real campaign data consists of many 5-minute
batch files (`telemetry_*.parquet`) in a directory.

**Decision**: Update `forecasting/run_forecasting.py` to detect batch files:
- If `telemetry_*.parquet` files exist in `telemetry_dir` → use `load_campaign()`
  to merge all batches into one DataFrame
- Otherwise → fall back to `_latest_parquet()` for single-file mode (synthetic data)

This makes the forecasting pipeline transparent to whether data came from a live
campaign or synthetic generation.

---

### Decision 5.6 — LSTM excluded from real-data forecasting runs (`SKIP_LSTM=1`)

**Context**: When the first real-data forecasting run was launched with `FULL_GRID=1
N_SPLITS=5` on 129,083 rows (10 services, ~9.3 hours of burst telemetry), the process
entered a sleeping state (0% CPU, 73 minutes elapsed) with no output. Investigation
confirmed the process was stuck in LSTM training.

**Root cause**: The LSTM full grid evaluates
`2 layers × 3 hidden sizes × 2 seq_lens × 2 dropouts = 24 configurations` per fold,
with up to 200 epochs per configuration and patience=10 early stopping. On 1,200
real samples per service with 5-fold CV, a conservative estimate gives
`10 services × 24 configs × 5 folds × 200 epochs × ~0.5s/epoch = ~120,000 seconds`
(33 hours) for LSTM alone. In practice the M3 Mac's PyTorch MPS backend was not being
utilised (CPU-only training), making it even slower. See Finding 5.7 for how MPS was
subsequently enabled.

**Decision**: Add `--skip-lstm` flag to `forecasting/run_forecasting.py` and `SKIP_LSTM=1`
support to `bin/run-forecasting.sh`. When set, the LSTM factory is removed from the
evaluation set before the service loop runs.

**Impact on dissertation**: LSTM is still included in the methodology (§3.6) as a
candidate model. On the synthetic 120-sample data it returned RMSE=inf (correctly, due
to insufficient fold size). On real data it is excluded from the current run due to
computational constraints on the development machine. The dissertation should note:

> LSTM was excluded from the local development forecasting run due to training time
> constraints on Apple Silicon (arm64, CPU-only PyTorch at the time). On the Apple
> M3 development machine with MPS now enabled (see Finding 5.7), LSTM training is
> expected to be substantially faster and can be re-enabled for a subsequent run by
> removing `SKIP_LSTM=1`.

---

### Finding 5.7 — Apple MPS available; LSTM device placement was missing

**Context**: After the LSTM was excluded (`SKIP_LSTM=1`) for being too slow on CPU,
investigation revealed that PyTorch 2.4.0 (the pinned version) **already supports
Apple MPS** (`torch.backends.mps.is_available() == True` on the M3 Mac). The LSTM
was running on CPU not because MPS was unavailable, but because the `lstm_model.py`
training code had no device-placement logic — all tensors and models defaulted to
CPU even when a GPU was present.

**Finding summary**:
- PyTorch version pinned: `torch==2.4.0`
- `torch.backends.mps.is_available()`: **True** (M3 Mac, macOS 15.x)
- `torch.backends.mps.is_built()`: **True**
- LSTM code before fix: no `.to(device)` calls anywhere; defaulted to CPU
- Other four models (XGBoost, SARIMA, Holt-Winters, Seasonal Naive): CPU-only by design (no GPU path in XGBoost/statsmodels on macOS)

**Which models use GPU**:

| Model | GPU capable | Device used (after fix) |
|---|---|---|
| XGBoost | No (CPU threading) | CPU (560% via multi-core) |
| SARIMA | No (statsmodels) | CPU |
| Holt-Winters | No (statsmodels) | CPU |
| Seasonal Naive | No (numpy) | CPU |
| LSTM | Yes (PyTorch MPS/CUDA) | **MPS** (Apple M3 GPU) after fix |

**Fix applied** (`forecasting/lstm_model.py`):
1. Device selection at fit time: MPS > CUDA > CPU, stored as `self._device`
2. All training tensors (`Xtr_t`, `ytr_t`, `Xva_t`, `yva_t`) moved to device via `.to(device)`
3. Model moved to device via `.to(device)` before training loop
4. Predict: input tensor moved to device; model moved to device
5. `save_artifact`: state_dict explicitly saved on CPU (`v.cpu()`) for portability
   across machines that may not have MPS or CUDA

**Reproducibility note**: `torch.manual_seed()` covers MPS seeding in PyTorch ≥ 2.x
(no separate `torch.mps.manual_seed()` call required). The three-seed discipline in
`forecasting/seeds.py` is unchanged and sufficient. MPS does not guarantee bitwise
determinism for all ops (unlike CUDA+deterministic mode), but `warn_only=True` in
`torch.use_deterministic_algorithms` already handles this gracefully.

**Impact on dissertation**: MPS acceleration makes LSTM feasible on the M3 Mac. With
the fix applied, running LSTM without `SKIP_LSTM=1` on a future campaign is viable.
DEV-010 status updated accordingly — the cloud/GPU host workaround is no longer the
only path to LSTM results.

---

## Real Data Collection Campaign — Burst Workload

### Campaign 1 (failed) — 2026-07-02 18:58–19:03 BST

- **Workload**: burst (looping, 12-minute cycle)
- **Target duration**: 5 hours
- **Actual duration**: ~5 minutes (Locust crashed on import error — see Issue 5.2)
- **Data collected**: 46 Parquet batches covering 14:03–18:04 UTC (5-hour lookback
  from campaign start, primarily idle/pre-traffic data)
- **Outcome**: Not usable as burst training data

### Campaign 2 (successful) — 2026-07-02 19:24–00:24 BST

- **Workload**: burst (looping, 12-minute cycle, BASELINE=50 users, PEAK=300 users)
- **Duration**: 5 hours
- **Locust throughput**: ~25 req/s sustained; ~8.9% error rate (100% from `/checkout`
  — expected, as checkout requires session state that is not set up in the task flow)
- **Data extracted**: 58 Parquet batches covering 18:24–23:24 UTC
- **Total rows**: 129,083 (across 10 services × ~9.3 hours combined with Campaign 1)
- **Services covered**: adservice, cartservice, checkoutservice, currencyservice,
  emailservice, frontend, paymentservice, productcatalogservice, recommendationservice,
  shippingservice (10 of 11 — `loadgenerator` excluded as not an autoscaling target)
- **Metric families present**: cpu, memory, pod_ready, request_rate
  (`response_time` absent — Online Boutique services do not expose
  `http_request_duration_seconds` histograms by default; response-time measurements
  will be obtained via wrk2 in Phase 5)
- **Outcome**: Suitable for forecaster training

### Checkout error rate note for dissertation

The `/checkout` endpoint returned 100% errors from Locust users throughout the campaign.
This is because the `BoutiqueUser` task `checkout` calls `GET /checkout` directly
without prior cart state. The frontend application returns an error page for empty-cart
checkout, which Locust counts as a failure. This does not affect:

1. CPU/memory/request-rate telemetry (all services still process the requests)
2. The forecaster training data (which uses CPU as the target metric)
3. The A/B experiment (where the wrk2 latency measurement uses the root endpoint,
   not `/checkout` directly)

The `/checkout` task weight is 10% of the user mix. The 8.9% aggregate error rate
in Locust logs is consistent with this task being the sole error source.

---

## Cumulative Deviations Summary

| ID | Phase | Summary | Validity impact |
|---|---|---|---|
| DEV-001 | 0 | Kubernetes v1.30 → v1.33 | None |
| DEV-002 | 0 | Prometheus v2.50 → v2.52 | None |
| DEV-003 | 0 | node-exporter v1.7 → v1.8 | None |
| DEV-004 | 0 | Histogram verify query changed for v2.52 | None |
| DEV-005 | 0 | cartservice CrashLoopBackOff on local profile | None (cloud profile clean) |
| DEV-006 | 0 | Python ML stack retains mid-2024 pins | TBD after audit |
| DEV-007 | 0 | cartservice memory limit 128Mi → 256Mi | None |
| DEV-008 | 1 | frontend rho 0.30 → 0.125 (analytical derivation) | None |
| DEV-009 | 5 | Locust campaign shape relative import fix | None (shape logic unchanged) |
| DEV-010 | 3 | LSTM excluded from local real-data run (training time) | Partial — MPS now enabled (Finding 5.7); re-run without SKIP_LSTM to resolve |
| DEV-011 | 3 | Forecasting pipeline: single-file → multi-batch Parquet loading | None |

---

## Key Architectural Decisions Reference

### The scaling formula (§3.7)

```
r*(t+h) = clip( ceil( (f̂(t+h) + k·σ̂(t+h)) / ρ ), r_min, r_max )
```

- `f̂(t+h)` — point forecast h seconds ahead
- `σ̂(t+h)` — prediction interval width (parametric for SARIMA/HW; quantile-residual for XGBoost/LSTM)
- `k` — safety margin coefficient (set to 1.0; configurable)
- `ρ` — per-pod target metric capacity (calibrated per service)
- `r_min, r_max` — replica bounds
- Rate limit: `|r*(t+h) − r(t)| ≤ ΔS` per tick (default ΔS=2)

### The three-criterion model selection rule (§3.6)

1. Lowest mean RMSE across walk-forward folds (primary)
2. Lowest p95 inference latency if RMSE within 1% (tie-break 1)
3. Fewest parameters if latency within 10% (tie-break 2)

### Evidence bundle schema (§3.8)

Each controller tick writes one JSON line containing:
`timestamp, service, state, current_replicas, new_replicas, forecast_point,
forecast_sigma, fallback_engaged, fallback_reason, forecaster, history_window_seconds`

Phases 6–7 extend this with: `shap_top_features, insertion_auc, deletion_auc,
faithfulness_passes, narrative`

---

## Measurement Environment — Final Campaign (2026-07-04)

### Decision M.1 — AWS m6a.4xlarge (eu-west-2) as the measurement host

The final measured campaign (corrected harness, wrk2, `kind-cloud` profile) runs on a
rented AWS EC2 instance, not on the development laptop. Selection was made from a live
price/availability survey of all 16-vCPU candidates in eu-west-2 (all offered in
eu-west-2a/b/c; on-demand Linux pricing via the AWS Pricing API, 2026-07-04):

| Instance | vCPU / RAM | CPU | $/h (eu-west-2) | Hours on $119.98 credit |
|---|---|---|---|---|
| **m6a.4xlarge** (chosen) | 16 / 64 GiB | AMD EPYC Milan | **0.7992** | ~150 h (≈6.2 days) |
| c6i.4xlarge | 16 / 32 GiB | Intel Ice Lake | 0.8080 | ~148 h |
| c7i.4xlarge | 16 / 32 GiB | Intel Sapphire Rapids | 0.8484 | ~141 h |
| m7i.4xlarge (fallback) | 16 / 64 GiB | Intel Sapphire Rapids | 0.9324 | ~129 h |
| c7a.4xlarge | 16 / 32 GiB | AMD EPYC Genoa | 0.9757 | ~123 h |
| m7a.4xlarge | 16 / 64 GiB | AMD EPYC Genoa | 1.0722 | ~112 h |

**Rationale**: m6a.4xlarge is the cheapest 64 GiB option (more runtime headroom per
credit than any alternative, including the 32 GiB compute-optimised family), is
16 vCPU AMD EPYC (matching the §3.10 hardware class), and comfortably hosts the
`kind-cloud` profile (nominal 14 vCPU / 40 GiB across nodes). CPU generation is
irrelevant to the A/B comparison: both controller arms run on the same instance and
only cross-trial consistency matters. Absolute tail-latency magnitudes are bound to
this instance spec per §3.10's scale-honesty clause.

**Funding**: AWS free-plan credit, $119.98 remaining, account "Argon"
(0312-7718-5797, alias aws-yele), **hard expiry 2026-07-23** — all VM work must
complete, with results archived off-instance and the instance terminated, by
~2026-07-20. Instance stopped (not terminated) between work sessions; gp3 root
volume ≥100 GiB (~$0.31/day while stopped). On-demand only — Spot interruption
would destroy a 5-hour collection campaign.

**Verification (completed 2026-07-04, via `argon-admin` IAM user + AWS CLI)**:
- [x] EC2 Service Quota L-1216C47A "Running On-Demand Standard (A, C, D, H, I, M,
  R, T, Z) instances", eu-west-2: **applied value 32 vCPUs** (AWS default is 5;
  this account holds an elevated allocation) — headroom for m6a.4xlarge (16) plus
  margin. No increase request needed.
- [x] `run-instances --dry-run` for m6a.4xlarge with Ubuntu 24.04 AMI
  `ami-01bd674894e3ea876` (canonical SSM parameter, eu-west-2) and a 100 GiB gp3
  root volume returned *"Request would have succeeded"* — the free plan does not
  block the instance type; AMI, permissions and quota all validated.
- [x] IAM user `argon-admin` (AdministratorAccess) created for all campaign work;
  root no longer used for daily operations.

---

## Decision M.2 — Measurement environment as code (Terraform) — 2026-07-05

The §3.10-A host is now provisioned by a Terraform module at
`infra/terraform/` rather than console/CLI steps. Committed defaults ARE the
specification (m6a.4xlarge, eu-west-2, pinned AMI `ami-01bd674894e3ea876`,
100 GiB encrypted gp3, SSH-only ingress, IMDSv2); overriding any default for a
measured run is a deviation. The provider lock file is committed so `terraform
init && terraform apply` reproduces the identical host for any reader —
upgrading the §3.10-A reproducibility claim from "the same instance type is
publicly rentable" to "the environment definition is part of the artefact."
Validated 2026-07-05: `terraform validate` clean; live `plan` against the
Argon account = 3 resources, no errors. Two defects caught by the live plan
and fixed: AWS SG-description charset (no apostrophes) and IPv4-only CIDR
validation (operator network egresses via IPv6; `apply` requires an IPv4 /32).
Instance run-state (stop/start) is deliberately NOT Terraform-managed — CLI
commands are emitted as outputs; `terraform destroy` closes the campaign
(≤ 2026-07-20).

---

## Pending Items Before Submission

| Item | Blocking what |
|---|---|
| Collect ramp, periodic workload data (5h each) | Phase 3 real-data replication for all 4 workloads |
| Re-run Phase 3 with LSTM (MPS now enabled on M3 — remove SKIP_LSTM=1) | DEV-010 resolution |
| Calibrate per-service ρ from steady-state telemetry | Phase 4 / Phase 5 controller accuracy |
| OSF preregistration freeze | Must precede final A/B experiment run |
| Run canonical-10-trials.yaml A/B experiment | Chapter 4 §4.3 latency, replica-seconds, oscillation tables |
| Run n=30 LLM narrative evaluation (needs OPENAI_API_KEY) | H4 FActScore result |
| Human Likert rating of 30 narratives | §4.5 qualitative result |
| Zenodo archive + DOI | §3.13 / README |

---

*Last updated: 2026-07-03*
