# observability/ — Prometheus, kube-state-metrics, node-exporter, Grafana

All four §3.10 telemetry components are installed via the `prometheus-community/kube-prometheus-stack` Helm chart, pinned to **version 58.7.2** which bundles:

| Component | Version | §3.10 spec | Status |
|-----------|---------|-----------|--------|
| Prometheus | 2.50.x | v2.50 | ✓ |
| kube-state-metrics | 2.12.x | v2.12 | ✓ |
| node-exporter | 1.7.x | v1.7 | ✓ |
| Grafana | 10.4.x | v10.4 | ✓ |

`verify.sh` checks the running images against these targets and warns if the chart has drifted.

## Scrape interval

15 seconds (`values.yaml: prometheus.prometheusSpec.scrapeInterval`). This matches the Kubernetes HPA sync period and is the single source of truth for §3.5 (data collection at 15s resolution).

## Service discovery

`servicemonitor.yaml` selects any Service in the `default` namespace carrying the label `monitoring.hybrid-autoscaler/scrape: "true"`. Online Boutique services do not carry that label by default — we add it per-service in Phase 1/2 as we wire each into the data pipeline.

For the §3.5 metric families that don't depend on application-level scrapes (CPU, memory, pod-ready state, host metrics), the chart's built-in cAdvisor, kube-state-metrics and node-exporter ServiceMonitors cover them out of the box.

## Files

- `install.sh` — adds the helm repo and installs/upgrades the chart with the pinned version.
- `values.yaml` — chart overrides (scrape interval, resources, namespace selectors, Grafana enable).
- `servicemonitor.yaml` — opt-in ServiceMonitor for Online Boutique services.

## Access

After install:

```bash
# Prometheus UI:
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
# → http://localhost:9090

# Grafana (default admin / admin — change before any cloud deployment):
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
# → http://localhost:3000
```
