"""Data collection and feature engineering (§3.5).

Layout:
    data.schema        — Pydantic models + schema version string
    data.synthetic     — synthetic multivariate generator (no cluster needed)
    data.collect       — Prometheus → Parquet exporter
    data.features      — lags, rolling stats, diffs, upstream exogenous
    data.leakage_check — anti-leakage validator (§3.5 absolute rule)
    data.splits        — time-ordered splits + walk-forward CV

The §3.5 metric families are the single source of truth across the package:
    cpu                — container_cpu_usage_seconds_total (cores/sec)
    memory             — container_memory_working_set_bytes (bytes)
    pod_ready          — kube_pod_status_ready (count)
    request_rate       — application request rate (req/sec)
    response_time      — application response-time histogram quantiles
"""

from .schema import (
    METRIC_FAMILIES,
    SCHEMA_VERSION,
    FeatureSchema,
    MetricFamily,
    TelemetryRow,
)

__all__ = [
    "SCHEMA_VERSION",
    "METRIC_FAMILIES",
    "MetricFamily",
    "TelemetryRow",
    "FeatureSchema",
]
