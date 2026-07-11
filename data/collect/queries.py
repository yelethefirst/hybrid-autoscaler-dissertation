"""PromQL templates for the five §3.5 metric families.

Each function returns a list of `(metric_family, metric_name, query)` tuples
for a single service. The exporter iterates these for every service and time
range.

These queries assume the kube-prometheus-stack scrape labels that the
Phase 0 install produces (`namespace`, `pod`, `container`). The Online
Boutique application metrics (`http_requests_per_second`, etc.) require
the per-service ServiceMonitor we wire in Phase 2 calibration; until then,
network/cAdvisor proxies are used.
"""

from __future__ import annotations

from typing import List, Tuple

from ..schema import MetricFamily

QueryTuple = Tuple[MetricFamily, str, str]


def build_queries(service: str, namespace: str = "default") -> List[QueryTuple]:
    """Return the full §3.5 query set for one service."""
    pod_match = f'pod=~"{service}-.*"'
    ns_match = f'namespace="{namespace}"'

    queries: List[QueryTuple] = [
        # 1. CPU usage rate (cores/sec, summed over containers in matching pods)
        (
            MetricFamily.CPU,
            "container_cpu_usage_seconds_total",
            f'sum by (pod, namespace) ('
            f'rate(container_cpu_usage_seconds_total{{'
            f'{ns_match},{pod_match},'
            f'container!="",container!="POD"}}[1m]))',
        ),

        # 2. Memory working set (bytes, summed across containers in matching pods)
        (
            MetricFamily.MEMORY,
            "container_memory_working_set_bytes",
            f'sum by (pod, namespace) ('
            f'container_memory_working_set_bytes{{'
            f'{ns_match},{pod_match},'
            f'container!="",container!="POD"}})',
        ),

        # 3. Pod ready count
        (
            MetricFamily.POD_READY,
            "kube_pod_status_ready",
            f'sum by (namespace) ('
            f'kube_pod_status_ready{{condition="true",'
            f'{ns_match},{pod_match}}})',
        ),

        # 4a. Request rate — application-level if available (Online Boutique
        #     ServiceMonitor); otherwise falls back to network rx packets as
        #     a proxy. Both forms are queried; the analysis layer prefers the
        #     application metric when present.
        (
            MetricFamily.REQUEST_RATE,
            "http_requests_per_second",
            f'sum by (pod, namespace) ('
            f'rate(http_requests_total{{{ns_match},{pod_match}}}[1m]))',
        ),
        (
            MetricFamily.REQUEST_RATE,
            "container_network_receive_packets_per_second",
            f'sum by (pod, namespace) ('
            f'rate(container_network_receive_packets_total{{'
            f'{ns_match},{pod_match}}}[1m]))',
        ),

        # 4b. Error rate
        (
            MetricFamily.REQUEST_RATE,
            "http_requests_errors_per_second",
            f'sum by (pod, namespace) ('
            f'rate(http_requests_total{{{ns_match},{pod_match},'
            f'code=~"5.."}}[1m]))',
        ),

        # 5. Response-time histogram quantiles — p50, p95, p99
        (
            MetricFamily.RESPONSE_TIME,
            "http_request_duration_seconds_p50",
            f'histogram_quantile(0.50, '
            f'sum by (le, pod, namespace) ('
            f'rate(http_request_duration_seconds_bucket{{'
            f'{ns_match},{pod_match}}}[1m])))',
        ),
        (
            MetricFamily.RESPONSE_TIME,
            "http_request_duration_seconds_p95",
            f'histogram_quantile(0.95, '
            f'sum by (le, pod, namespace) ('
            f'rate(http_request_duration_seconds_bucket{{'
            f'{ns_match},{pod_match}}}[1m])))',
        ),
        (
            MetricFamily.RESPONSE_TIME,
            "http_request_duration_seconds_p99",
            f'histogram_quantile(0.99, '
            f'sum by (le, pod, namespace) ('
            f'rate(http_request_duration_seconds_bucket{{'
            f'{ns_match},{pod_match}}}[1m])))',
        ),
    ]
    return queries
