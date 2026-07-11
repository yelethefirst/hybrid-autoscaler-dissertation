"""Prometheus collection (§3.5).

Pulls the five §3.5 metric families for all configured services over a
time range and writes long-format Parquet in 5-minute batches.
"""

from .exporter import PrometheusExporter, run_campaign
from .queries import build_queries

__all__ = ["PrometheusExporter", "run_campaign", "build_queries"]
