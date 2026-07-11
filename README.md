# hybrid-autoscaler

Reference implementation for the dissertation **"Intelligent Predictive and Explainable Autoscaling for Kubernetes Microservices Using Hybrid Time-Series Forecasting and Large Language Models"** (Omoyele Sodiq Olabode, York St John University, 2026).

This repository is the artefact contribution of the dissertation and is intended to be the published **reproducibility package** described in Section 3.10 of the thesis.

> **Status (2026-07-05):** all layers implemented and unit-tested (197 tests). A pilot A/B campaign ran on the development environment; its measurement defects are documented in `docs/deviations.md` (DEV-013–017) and the corrected final campaign runs on the cloud measurement environment (§3.10-A). Current plan and progress: `../AUDIT_REMEDIATION_ROADMAP.md`; decision history: `docs/IMPLEMENTATION_LOG.md`.

---

## What this is

A predictive, explainable Horizontal Pod Autoscaler for Kubernetes microservices. It produces short-horizon (30–60 s) demand forecasts, maps them to replica recommendations via a constrained decision engine with deterministic HPA fallback, and accompanies every scaling decision with a faithfulness-validated SHAP attribution and a retrieval-bounded LLM narrative.

It is evaluated against the default Kubernetes Horizontal Pod Autoscaler on the **Online Boutique** benchmark under four workload conditions (burst, ramp, periodic, Alibaba-trace replay) on a **kind v0.32** cluster running Kubernetes v1.33.

## Layout

```
hybrid-autoscaler/
├── README.md                 (this file)
├── pyproject.toml            (uv-managed Python deps, pinned)
├── .gitignore
├── docs/
│   ├── SEEDS.md              (seed-management convention; reproducibility-critical)
│   ├── IMPLEMENTATION_LOG.md (decisions, findings, and campaign records)
│   └── deviations.md         (every divergence from the dissertation spec, DEV-001…)
├── infra/
│   ├── README.md
│   ├── terraform/            (measurement host §3.10-A as code — see its README)
│   ├── bootstrap-macos.sh    (install Docker Desktop, kind, kubectl, helm, uv)
│   ├── kind/
│   │   ├── kind-local.yaml   (modest cluster for laptop dev)
│   │   ├── kind-cloud.yaml   (at-§3.10-spec cluster for measured runs)
│   │   └── up.sh
│   ├── online-boutique/
│   │   ├── README.md
│   │   ├── install.sh        (deploys v0.10.5 manifest)
│   │   └── pin-images.sh     (resolves and persists SHA256 image digests)
│   ├── observability/
│   │   ├── README.md
│   │   ├── install.sh
│   │   ├── values.yaml       (kube-prometheus-stack pinned chart; 15 s scrape)
│   │   └── servicemonitor.yaml
│   └── verify.sh             (Phase 0 exit-criterion check)
├── data/                     (schema, synthetic/live collection, features, splits)
├── forecasting/              (five forecaster families, intervals, selection, H1)
├── controller/               (control loop, decision engine, evidence writer, configs)
├── experiments/              (supervisor, trial plans, workloads, analysis, results)
├── explain/                  (SHAP attribution + faithfulness metrics)
├── narrate/                  (retrieval-bounded LLM narrator + FActScore)
├── analysis/                 (pre-registered H0–H4 statistical tests)
├── preregistration/          (OSF-frozen hypotheses — osf.io/srewu)
└── reproducibility/          (host-spec capture)
```

## Quickstart (local dev, macOS Apple Silicon)

```bash
# 1. Install host dependencies (one-time)
bash infra/bootstrap-macos.sh

# 2. Bring up the local kind cluster (modest profile, M-series friendly)
bash infra/kind/up.sh local

# 3. Deploy Online Boutique pinned to v0.10.5
bash infra/online-boutique/install.sh

# 4. (Optional but recommended) Resolve and persist image digests for reproducibility
bash infra/online-boutique/pin-images.sh

# 5. Install the observability stack (Prometheus 2.52, kube-state-metrics 2.12,
#    node-exporter 1.8, Grafana 10.4) and the Online Boutique ServiceMonitor
bash infra/observability/install.sh

# 6. Verify the Phase 0 exit criterion
bash infra/verify.sh
```

The verify script confirms: all 11 Online Boutique pods Ready; the full Prometheus stack Ready; the five metric families specified in §3.5 (CPU usage, memory working set, pod-ready count, request/error rate, response-time histograms) are queryable for every Deployment.

## Cluster profiles

| Profile | Use | Nodes | Resources |
|---------|-----|------|-----------|
| `kind-local.yaml` | Laptop dev | 1 cp + 2 workers | 2 vCPU / 4 GiB per worker |
| `kind-cloud.yaml` | Measured runs (§3.10) | 1 cp + 3 workers | 4 vCPU / 12 GiB per worker (cp: 2 vCPU / 4 GiB) |

Run final measured trials on the **cloud profile** on a single VM sized close to §3.10 (≈16 vCPU, 64 GiB RAM, NVMe). Local development uses the local profile; deviations from §3.10 are honestly reported in §3.13 (Threats to Validity) of the dissertation.

## Reproducibility pillars

This repository implements every reproducibility pillar required by §3.10:

- **Measurement host as code**: the §3.10-A EC2 environment is a Terraform module (`infra/terraform/`, committed provider lock file) — `terraform apply` recreates the identical host.
- **Image SHA256 digests**, not mutable tags (see `infra/online-boutique/pin-images.sh`).
- **`uv`-managed Python dependency pinning** with exact-version lockfile (`pyproject.toml` + `uv.lock`).
- **Seed-management convention** documented in `docs/SEEDS.md` (NumPy / PyTorch / dataloader-worker seeds; deterministic CUDA disabled).
- **Pinned Kubernetes, containerd, Prometheus stack and Online Boutique versions** as configured in the manifests.
- **Per-trial Git commit SHA** logged by the experiment supervisor (added in Phase 5).

## Host specification (placeholder — fill in for final runs)

Per §3.10, the dissertation reports the exact hardware on which the measured results were produced. Update this section before final experiments:

- **Host hardware:** ___ (e.g. AWS c7a.4xlarge: 16 vCPU AMD EPYC 9R14, 32 GiB RAM)
- **Host OS:** ___ (e.g. Ubuntu 24.04 LTS, kernel 6.8)
- **Container runtime:** ___ (Docker Engine ___, containerd ___, cgroup v2)
- **Kubernetes runtime:** v1.33 via kind v0.32 (1 control-plane + 3 worker nodes)
- **Load generator host:** ___ (separate VM, same zone/subnet)

## Licence

MIT (to be added in Phase 9 alongside the Zenodo DOI).
