"""Prometheus client for the control loop.

Wraps `prometheus-api-client` with the small set of queries the Phase 1
vertical slice needs. Phase 2 (data-collection pipeline) replaces this with
a richer feature-engineering layer; this module remains the canonical PromQL
helper for live queries inside the control loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from prometheus_api_client import PrometheusConnect


class PrometheusClient:
    """Thin facade over PrometheusConnect for the metrics we care about."""

    def __init__(self, base_url: str, timeout_seconds: float = 5.0):
        self._client = PrometheusConnect(
            url=base_url,
            disable_ssl=True,
            headers=None,
        )
        self._client._session.request = _with_timeout(  # type: ignore[attr-defined]
            self._client._session.request, timeout_seconds
        )

    # ------------------------------------------------------------------ #
    # CPU rate per service (Phase 1 metric)                               #
    # ------------------------------------------------------------------ #
    def cpu_total_cores(
        self,
        service: str,
        namespace: str,
        lookback_seconds: int,
        step_seconds: int = 15,
        end: Optional[datetime] = None,
    ) -> pd.Series:
        """Total CPU rate (cores) summed across pods of a service.

        Uses the cAdvisor signal `container_cpu_usage_seconds_total` filtered
        to non-init, non-pause containers belonging to pods whose name starts
        with the service name. This mirrors the §3.5 first metric family.

        Returns a `pd.Series` indexed by UTC timestamp with 1 sample per
        `step_seconds`. Empty Series if Prometheus returned no data.
        """
        end = end or datetime.now(timezone.utc)
        start = end - timedelta(seconds=lookback_seconds)
        query = (
            'sum by () ('
            'rate(container_cpu_usage_seconds_total{'
            f'namespace="{namespace}",'
            f'pod=~"{service}-.*",'
            'container!="",container!="POD"'
            "}[1m])"
            ")"
        )
        result = self._client.custom_query_range(
            query=query, start_time=start, end_time=end, step=f"{step_seconds}s"
        )
        return _to_series(result)

    def cpu_avg_utilisation(
        self,
        service: str,
        namespace: str,
        end: Optional[datetime] = None,
    ) -> Optional[float]:
        """Instantaneous average per-pod CPU utilisation (fraction).

        Computed as (sum of CPU rate over containers in matching pods)
        ÷ (sum of CPU limit over those containers). Returns a fraction
        suitable for use as `u_t` in the §3.7 HPA-equivalent rule. None if
        the query returns no data.
        """
        end = end or datetime.now(timezone.utc)
        # Utilisation is computed against CPU REQUESTS, exactly as the
        # Kubernetes HPA does. The pilot divided by limits, which made the
        # hybrid's HPA-equivalent fallback tolerate ~2x more load per pod
        # than the real HPA arm (2026-07-05 code review; DEV-020).
        query = (
            "("
            "sum(rate(container_cpu_usage_seconds_total{"
            f'namespace="{namespace}",pod=~"{service}-.*",'
            'container!="",container!="POD"}[1m]))'
            ") / "
            "("
            "sum(kube_pod_container_resource_requests{"
            f'namespace="{namespace}",pod=~"{service}-.*",resource="cpu"'
            "})"
            ")"
        )
        result = self._client.custom_query(query=query, params={"time": end.timestamp()})
        if not result:
            return None
        try:
            return float(result[0]["value"][1])
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    def replica_count(
        self,
        service: str,
        namespace: str,
        end: Optional[datetime] = None,
    ) -> Optional[int]:
        """Number of Ready pods for the service (telemetry-side count)."""
        end = end or datetime.now(timezone.utc)
        query = (
            'sum(kube_pod_status_ready{condition="true",'
            f'namespace="{namespace}",pod=~"{service}-.*"'
            "})"
        )
        result = self._client.custom_query(query=query, params={"time": end.timestamp()})
        if not result:
            return None
        try:
            return int(float(result[0]["value"][1]))
        except (KeyError, IndexError, ValueError, TypeError):
            return None


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
def _to_series(result: list) -> pd.Series:
    """Convert a prometheus_api_client range-query result list to pd.Series."""
    if not result:
        return pd.Series(dtype=float)
    values = result[0].get("values", [])
    if not values:
        return pd.Series(dtype=float)
    index = pd.to_datetime([float(v[0]) for v in values], unit="s", utc=True)
    data = [float(v[1]) for v in values]
    return pd.Series(data=data, index=index, name="value")


def _with_timeout(original_request, timeout_seconds: float):
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original_request(method, url, **kwargs)

    return wrapped
