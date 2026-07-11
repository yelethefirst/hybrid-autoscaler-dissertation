# infra/ — Phase 0 cluster, observability and pinning

Everything here is infrastructure-as-code for the experiment. Anything that affects measured results is **version-pinned** and committed.

## Files

| File | Purpose |
|------|---------|
| `bootstrap-macos.sh` | One-time install of Docker Desktop, kind v0.22, kubectl, helm, uv on macOS Apple Silicon. |
| `kind/kind-local.yaml` | Modest cluster (1 cp + 2 workers, 2 vCPU / 4 GiB per worker) for laptop development. |
| `kind/kind-cloud.yaml` | At-§3.10 cluster (1 cp + 3 workers, 4 vCPU / 12 GiB per worker) for measured runs. |
| `kind/up.sh` | `bash kind/up.sh local` or `bash kind/up.sh cloud` — brings up the chosen profile. |
| `online-boutique/install.sh` | Deploys Online Boutique v0.10.5 manifest. |
| `online-boutique/pin-images.sh` | Resolves every image to its SHA256 digest and writes a pinned manifest. |
| `observability/install.sh` | Helm-installs kube-prometheus-stack (Prometheus 2.50 / kube-state-metrics 2.12 / node-exporter 1.7 / Grafana 10.4) + the ServiceMonitor for Online Boutique. |
| `verify.sh` | Phase 0 exit-criterion check. |

## Pinned versions (single source of truth)

| Component | Version | Source |
|-----------|---------|--------|
| Kubernetes | v1.30.x | kind node image `kindest/node:v1.30.0` |
| kind | v0.22.0 | binary download (§3.10) |
| containerd | 1.7 (via Docker Desktop) | §3.10 |
| Online Boutique | v0.10.5 | upstream tag |
| kube-prometheus-stack (helm chart) | 58.7.2 | ships Prometheus 2.50.x (verify post-install) |
| Prometheus | v2.50.x | sub-chart of kube-prometheus-stack |
| kube-state-metrics | v2.12.x | sub-chart |
| node-exporter | v1.7.x | sub-chart |
| Grafana | v10.4.x | sub-chart |
| Locust | 2.44.0 | pyproject.toml (§3.9) |
| wrk2 | giltene/wrk2 master | built from source in Phase 5 (§3.9) |

The `verify.sh` script asserts the runtime versions match where checkable.

## Conventions

- All scripts are idempotent — safe to re-run.
- All scripts use `set -euo pipefail` and exit non-zero on any failure.
- All resource limits are explicit (no implicit `latest` tags after Phase 0).
- Per-trial state is torn down between trials by the supervisor (Phase 5), not by these scripts.
