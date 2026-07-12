# hybrid-autoscaler

Reference implementation for the dissertation **"Intelligent Predictive and Explainable Autoscaling for Kubernetes Microservices Using Hybrid Time-Series Forecasting and Large Language Models"** (Omoyele Sodiq Olabode, York St John University, 2026).

This repository is the artefact contribution of the dissertation and is the published **reproducibility package** described in Section 3.10 of the thesis.

> **Status (2026-07-12):** all seven phases complete. The full 84-trial A/B campaign ran on the Hetzner CPX62 measurement instance (Phases 5–7). All experimental data, model artefacts, per-trial logs, SHAP attributions and LLM narratives are committed to this repository. Key results: **H1 confirmed** (Holt-Winters outperforms Seasonal Naive on all 10 services); **H0 partially confirmed** (40–86% p95 latency reduction on 3 of 4 workloads); **H2 not confirmed** (Hybrid uses 5–8× more replica-seconds); **H4 not confirmed** (FActScore 0.648 < threshold 0.80). See [Results](#results) below.

---

## What this is

A predictive, explainable Horizontal Pod Autoscaler for Kubernetes microservices. It produces short-horizon (30–60 s) demand forecasts, maps them to replica recommendations via a constrained decision engine with deterministic HPA fallback, and accompanies every scaling decision with a SHAP attribution and a retrieval-bounded LLM narrative.

Evaluated against the default Kubernetes HPA on the **Online Boutique** benchmark under four workload conditions (burst, ramp, periodic, Alibaba-trace replay) on a **kind v0.32** cluster running Kubernetes v1.33 on a Hetzner CPX62 instance.

---

## Results

All results are from 84 paired A/B trials (10 per arm per workload) on the Hetzner CPX62 measurement instance. Latency measured with wrk2 (coordinated-omission correction). Statistical tests: Wilcoxon signed-rank with BCa bootstrap 95% CIs and Holm-Bonferroni correction.

### H1 — Forecasting vs Seasonal Naive baseline
**✅ Confirmed** for all 10 Online Boutique services.
Holt-Winters exponential smoothing was selected on every service. Effect sizes large to very large (Cohen's d_z 3.30–13.94), Holm-corrected p < 0.001 on all ten.

### H0 — p95 tail latency: Hybrid vs HPA

| Workload | Hybrid p95 | HPA p95 | Reduction | d_z | p |
|---|---|---|---|---|---|
| Ramp | 33 ms | 245 ms | **−86%** | −13.50 | < 0.001 |
| Burst | 60 ms | 101 ms | **−40%** | −1.44 | 0.004 |
| Trace-replay | 975 ms | 1,089 ms | **−11%** | −0.88 | 0.042 |
| Periodic | 399 ms | 373 ms | +7% (ns) | +0.27 | 0.42 |

**⚠️ Partially confirmed** — significant on 3 of 4 workloads. Predictive look-ahead provides no advantage over periodic sinusoidal traffic handled by HPA's stabilisation window.

### H2 — Replica-seconds efficiency: Hybrid vs HPA
**❌ Not confirmed** — Hybrid uses 5–8× more replica-seconds across all workloads (burst 5.5×, ramp 7.5×, periodic 7.2×, trace-replay 7.6×). The latency improvement is purchased at a real resource cost; the controller is best characterised as a latency-first premium, not a cost-neutral enhancement. See [Future Work](#future-work) item 2 for the remedy.

### H3 — Scaling oscillation rate
**⚪ Not measurable** — oscillation counts were not captured in the results pipeline (measurement gap, not a null result).

### H4 — LLM narrator FActScore ≥ 0.80
**❌ Not confirmed** — mean FActScore 0.648 (range 0.43–0.86) across 30 GPT-4o-mini narratives. The narrator generates plausible but numerically imprecise paraphrases correctly flagged by the FActScore judge. See [Future Work](#future-work) item 5 for the fix.

---

## Repository layout

```
hybrid-autoscaler/
├── README.md                 (this file)
├── future_work.docx          (detailed 8-direction future work document)
├── dissertation/             (final dissertation docx)
├── pyproject.toml            (uv-managed Python deps, pinned)
├── docs/
│   ├── SEEDS.md
│   ├── IMPLEMENTATION_LOG.md
│   └── deviations.md         (DEV-001 … DEV-019)
├── infra/
│   ├── terraform/            (Hetzner CPX62 as code)
│   ├── bootstrap-macos.sh
│   ├── kind/                 (kind-local.yaml, kind-cloud.yaml, up.sh)
│   ├── online-boutique/      (install.sh, pin-images.sh)
│   ├── observability/        (Prometheus, Grafana, dashboards/)
│   └── verify.sh
├── data/                     (schema, collection, features, splits)
├── forecasting/              (Seasonal Naive, Holt-Winters, XGBoost, LSTM, SARIMA)
├── controller/               (control loop, decision engine, evidence writer)
├── experiments/
│   ├── run_phase5_analysis.py
│   ├── run_phase6_shap.py
│   ├── run_phase7_narrator.py
│   └── results/
│       ├── results_canonical-ab-v2.jsonl   ← 84-trial canonical A/B results
│       ├── phase5_stats.json               ← H0/H2/H3 statistical tests
│       ├── phase7_narratives.jsonl         ← 30 LLM narratives
│       ├── phase7_h4.json                  ← H4 FActScore test result
│       ├── evidence/                       ← 40 Hybrid per-trial decision bundles (3,920 decisions)
│       ├── shap/                           ← Phase 6 SHAP attributions (1,200 across 40 trials)
│       ├── locust/                         ← Per-trial Locust HTTP stats CSVs (336 files)
│       ├── logs/                           ← Per-trial wrk2/locust/controller logs (250 files)
│       ├── models/                         ← Trained forecasting model artefacts
│       └── archive/                        ← Invalidated runs and campaign execution logs
├── explain/                  (SHAP attribution + faithfulness metrics)
├── narrate/                  (LLM narrator + FActScore evaluator)
├── analysis/                 (pre-registered H0–H4 statistical tests)
├── preregistration/          (OSF-frozen hypotheses — osf.io/srewu)
└── reproducibility/          (host-spec capture)
```

---

## Quickstart (local dev, macOS Apple Silicon)

```bash
# 1. Install host dependencies
bash infra/bootstrap-macos.sh

# 2. Bring up the local kind cluster (M-series friendly)
bash infra/kind/up.sh local

# 3. Deploy Online Boutique pinned to v0.10.5
bash infra/online-boutique/install.sh

# 4. Resolve and persist image digests
bash infra/online-boutique/pin-images.sh

# 5. Install observability stack (Prometheus, Grafana, kube-state-metrics)
bash infra/observability/install.sh

# 6. Verify Phase 0 exit criterion
bash infra/verify.sh
```

## Cluster profiles

| Profile | Use | Nodes | Resources |
|---|---|---|---|
| `kind-local.yaml` | Laptop dev (macOS M-series) | 1 cp + 2 workers | 2 vCPU / 4 GiB per worker |
| `kind-cloud.yaml` | Measured runs (§3.10-A) | 1 cp + 3 workers | 4 vCPU / 12 GiB per worker |

---

## Measurement host specification

All results in the dissertation were produced on:

- **Host hardware:** Hetzner CPX62 — 16 vCPU AMD EPYC (shared), 32 GiB RAM, Nuremberg nbg1 datacenter
- **Host OS:** Ubuntu 24.04 LTS
- **Container runtime:** Docker Engine 26, cgroup v2 (DEV-018)
- **Kubernetes:** v1.33 via kind v0.32 (1 control-plane + 3 worker nodes, kind-cloud profile)
- **Load generator:** wrk2 (coordinated-omission correction) + Locust 2.44.0, co-located on measurement host
- **Observability:** Prometheus v2.52 (15 s scrape), kube-state-metrics v2.12, node-exporter v1.8, Grafana v10.4
- **Host fingerprint:** `reproducibility/host-spec.json`

> Pre-registered host was AWS EC2 m6a.4xlarge (DEV-018). The Hetzner CPX62 preserves the 16 vCPU floor and AMD EPYC microarchitecture; tail-latency magnitudes are bound to this specification.

---

## Reproducibility pillars

- **Measurement host as code:** `infra/terraform/` recreates the Hetzner CPX62 instance with `terraform apply`
- **Image SHA256 digests**, not mutable tags (`infra/online-boutique/pin-images.sh`)
- **`uv`-managed Python pinning** with exact-version lockfile (`uv.lock`)
- **Seed management** documented in `docs/SEEDS.md`
- **Pre-registration** frozen on OSF before A/B data collection: [osf.io/srewu](https://osf.io/srewu)
- **All raw data committed** to this repository (evidence bundles, SHAP outputs, per-trial logs, model artefacts)
- **Deviation log** in `docs/deviations.md` (DEV-001 to DEV-019)

---

## Future Work

Eight directions for extending this work are identified in the dissertation (§6.4) and detailed in [`future_work.docx`](future_work.docx):

1. **Multi-service deployment with DAG propagation** — scale all 10 services via topological ordering; eliminates the backend bottleneck and reveals the true system-level latency gain
2. **Adaptive confidence margin** — replace fixed k=1.5 with a Kalman-filter or RL-driven policy to directly address the H2 replica-seconds finding
3. **Online Holt-Winters parameter updates** — keep the forecaster current with real-time traffic drift without full retraining
4. **Multi-metric autoscaling** — extend beyond CPU to request queue depth, memory, and network ingress
5. **Structured JSON narrator output** — enforce output schema to guarantee numerical precision and push FActScore above 0.80
6. **Hysteresis rate limiter + H3 instrumentation** — log scale-event counts and replace the static ΔS cap with an empirically-derived hysteresis band
7. **Reinforcement learning end-to-end controller** — learn optimal k and ΔS jointly via a reward of (latency penalty + λ × replica cost)
8. **DeathStarBench + production fintech traces** — evaluate on a 28-service benchmark and real payment-processing traffic to test external validity

---

## Licence

MIT. See `LICENSE`.
