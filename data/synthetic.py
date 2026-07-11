"""Synthetic multivariate telemetry generator.

Produces realistic-looking time series of the §3.5 metric families for an
arbitrary set of services. Used to exercise the data pipeline, feature
engineering, leakage validator and splits before the live cluster is up.
Once Phase 0 is executed and live telemetry is available, this generator
remains useful as a fixture for unit and integration tests.

Design notes
------------
* The generator is **deterministic** for a given seed (§3.6 seed discipline).
* It models the §3.5 cross-service coupling: a service's CPU is partially
  driven by upstream services' request rate plus its own seasonality and
  noise, following the form in Hu et al. (2022).
* Workloads ("burst", "ramp", "periodic", "trace_like") shape the
  *upstream* request-rate signal; downstream services react to it.
* Output is a long-format `pd.DataFrame` whose columns match
  `data.schema.TelemetryRow`, so it can be Parquet-written directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .schema import SCHEMA_VERSION, MetricFamily

WorkloadKind = Literal["burst", "ramp", "periodic", "trace_like", "steady"]


# ---------------------------------------------------------------------- #
# Service topology (subset of Online Boutique; extendable)               #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ServiceSpec:
    """One service's synthetic-generation parameters."""

    name: str
    baseline_cpu_cores: float = 0.05          # idle CPU
    baseline_memory_bytes: float = 200e6      # idle memory
    baseline_replicas: int = 1
    # CPU per request/sec received. Total CPU ≈ baseline + rps × cpu_per_rps + noise.
    cpu_per_rps: float = 0.003
    # Random-walk noise σ as a fraction of baseline.
    noise_cpu_ratio: float = 0.05
    # Request-rate baseline (req/s) when no upstream driver applies.
    baseline_rps: float = 5.0
    # Names of upstream services whose request rate drives this one.
    upstream: List[str] = field(default_factory=list)


# Default topology approximating Online Boutique's call graph.
DEFAULT_TOPOLOGY: List[ServiceSpec] = [
    ServiceSpec("frontend",              cpu_per_rps=0.005, baseline_rps=10.0),
    ServiceSpec("productcatalogservice", cpu_per_rps=0.002, upstream=["frontend"]),
    ServiceSpec("currencyservice",       cpu_per_rps=0.001, upstream=["frontend"]),
    ServiceSpec("cartservice",           cpu_per_rps=0.003, upstream=["frontend"]),
    ServiceSpec("recommendationservice", cpu_per_rps=0.004, upstream=["frontend", "productcatalogservice"]),
    ServiceSpec("shippingservice",       cpu_per_rps=0.002, upstream=["frontend"]),
    ServiceSpec("checkoutservice",       cpu_per_rps=0.004, upstream=["frontend"]),
    ServiceSpec("paymentservice",        cpu_per_rps=0.002, upstream=["checkoutservice"]),
    ServiceSpec("emailservice",          cpu_per_rps=0.001, upstream=["checkoutservice"]),
    ServiceSpec("adservice",             cpu_per_rps=0.002, upstream=["frontend"]),
    ServiceSpec("loadgenerator",         cpu_per_rps=0.001, baseline_rps=0.0),
]


# ---------------------------------------------------------------------- #
# Workload drivers (shape the upstream RPS)                              #
# ---------------------------------------------------------------------- #
def _workload_rps(
    workload: WorkloadKind,
    n_samples: int,
    sample_interval_s: int,
    *,
    baseline_users: int = 10,
    peak_users: int = 100,
) -> np.ndarray:
    """Return an array of length n_samples giving upstream RPS over time."""
    rps_per_user = 0.5  # one user → ~0.5 RPS sustained

    if workload == "steady":
        return np.full(n_samples, baseline_users * rps_per_user, dtype=float)

    if workload == "burst":
        # 2 min low, 2 min peak, 2 min low — repeating
        period_s = 360
        low_s, peak_s = 120, 120
        out = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            t_in_period = (i * sample_interval_s) % period_s
            if t_in_period < low_s:
                users = baseline_users
            elif t_in_period < low_s + peak_s:
                users = peak_users
            else:
                users = baseline_users
            out[i] = users * rps_per_user
        return out

    if workload == "ramp":
        # 60s linear up, 120s hold, 60s linear down, then steady — repeating
        period_s = 360
        ramp_up_s, hold_s, ramp_down_s = 60, 120, 60
        out = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            t_in_period = (i * sample_interval_s) % period_s
            if t_in_period < ramp_up_s:
                frac = t_in_period / ramp_up_s
                users = baseline_users + (peak_users - baseline_users) * frac
            elif t_in_period < ramp_up_s + hold_s:
                users = peak_users
            elif t_in_period < ramp_up_s + hold_s + ramp_down_s:
                frac = (t_in_period - ramp_up_s - hold_s) / ramp_down_s
                users = peak_users - (peak_users - baseline_users) * frac
            else:
                users = baseline_users
            out[i] = users * rps_per_user
        return out

    if workload == "periodic":
        # 60s sinusoid between baseline and peak
        period_s = 60
        amp = (peak_users - baseline_users) / 2
        mid = (peak_users + baseline_users) / 2
        out = np.array(
            [
                (mid + amp * math.sin(2 * math.pi * (i * sample_interval_s) / period_s))
                * rps_per_user
                for i in range(n_samples)
            ],
            dtype=float,
        )
        return out

    if workload == "trace_like":
        # Non-stationary pseudo-trace: AR(1) walk + slow drift, clipped ≥ 0.
        rng = np.random.default_rng(0)  # deterministic for this synthetic
        x = np.zeros(n_samples, dtype=float)
        x[0] = baseline_users
        ar = 0.85
        for i in range(1, n_samples):
            x[i] = (
                ar * x[i - 1]
                + (1 - ar) * baseline_users
                + rng.normal(0, (peak_users - baseline_users) * 0.05)
            )
            # Slow drift toward peak in the middle third
            if n_samples // 3 <= i < 2 * n_samples // 3:
                x[i] += (peak_users - baseline_users) * 0.5 / n_samples
        x = np.clip(x, 0.0, None)
        return x * rps_per_user

    raise ValueError(f"unknown workload kind: {workload}")


# ---------------------------------------------------------------------- #
# Public API                                                              #
# ---------------------------------------------------------------------- #
def generate(
    workload: WorkloadKind = "periodic",
    duration_seconds: int = 600,
    sample_interval_seconds: int = 15,
    services: Optional[List[ServiceSpec]] = None,
    start: Optional[datetime] = None,
    seed: int = 0,
    baseline_users: int = 10,
    peak_users: int = 100,
) -> pd.DataFrame:
    """Generate a synthetic long-format telemetry DataFrame.

    Parameters
    ----------
    workload : WorkloadKind
        Shape of the upstream RPS driver.
    duration_seconds : int
        Length of the campaign in seconds.
    sample_interval_seconds : int
        Cadence (§3.5: 15 s).
    services : optional list of ServiceSpec
        Service topology. Defaults to a subset of Online Boutique.
    start : optional datetime
        UTC start timestamp. Defaults to 2026-05-01T00:00:00Z (deterministic).
    seed : int
        RNG seed (§3.6 seed discipline).
    """
    if services is None:
        services = DEFAULT_TOPOLOGY
    if start is None:
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    n_samples = duration_seconds // sample_interval_seconds
    timestamps = pd.date_range(start, periods=n_samples, freq=f"{sample_interval_seconds}s", tz="UTC")

    rng = np.random.default_rng(seed)

    # 1. Drive top-of-graph services (those with no upstream) by the workload.
    rps: Dict[str, np.ndarray] = {}
    driver_rps = _workload_rps(
        workload, n_samples, sample_interval_seconds,
        baseline_users=baseline_users, peak_users=peak_users,
    )
    for svc in services:
        if not svc.upstream:
            base = svc.baseline_rps if svc.name != "frontend" else 0.0
            noise = rng.normal(0, 1.0, n_samples)
            rps[svc.name] = np.clip(base + driver_rps + noise, 0.0, None)

    # 2. Downstream services: weighted sum of upstream RPS with attenuation.
    for svc in services:
        if not svc.upstream:
            continue
        upstream_sum = np.zeros(n_samples, dtype=float)
        for u in svc.upstream:
            attenuation = 0.6  # downstream sees ~60% of upstream traffic per hop
            upstream_sum += attenuation * rps[u]
        noise = rng.normal(0, 0.5, n_samples)
        rps[svc.name] = np.clip(svc.baseline_rps + upstream_sum + noise, 0.0, None)

    # 3. Build long-format rows for all five §3.5 metric families.
    rows: List[Dict] = []
    for svc in services:
        s_rps = rps[svc.name]
        # CPU = baseline + rps × cpu_per_rps + multiplicative noise
        cpu_noise = rng.normal(1.0, svc.noise_cpu_ratio, n_samples)
        cpu = np.clip(
            (svc.baseline_cpu_cores + s_rps * svc.cpu_per_rps) * cpu_noise,
            0.0,
            None,
        )
        # Memory drifts slowly with load
        mem_noise = rng.normal(1.0, 0.02, n_samples)
        memory = (svc.baseline_memory_bytes + s_rps * 1e5) * mem_noise
        # Replica count is fixed in synthetic mode (controller decides it for real)
        ready = np.full(n_samples, float(svc.baseline_replicas))
        # Response-time p95 inflates with CPU (simple monotone)
        rt_p95 = 0.02 + 0.5 * np.clip(cpu, 0, None) + rng.normal(0, 0.005, n_samples)
        rt_p95 = np.clip(rt_p95, 0.001, None)
        # Error rate small + spikes when CPU > 1.5× baseline
        err = np.where(
            cpu > 1.5 * (svc.baseline_cpu_cores + 0.1),
            rng.uniform(0.0, 0.01, n_samples),
            0.0,
        )

        for i, ts in enumerate(timestamps):
            rows.append(_row(ts, svc.name, MetricFamily.CPU,
                             "container_cpu_usage_seconds_total", cpu[i]))
            rows.append(_row(ts, svc.name, MetricFamily.MEMORY,
                             "container_memory_working_set_bytes", memory[i]))
            rows.append(_row(ts, svc.name, MetricFamily.POD_READY,
                             "kube_pod_status_ready", ready[i]))
            rows.append(_row(ts, svc.name, MetricFamily.REQUEST_RATE,
                             "http_requests_per_second", s_rps[i]))
            rows.append(_row(ts, svc.name, MetricFamily.RESPONSE_TIME,
                             "http_request_duration_seconds",
                             float(rt_p95[i]), {"quantile": "0.95"}))
            rows.append(_row(ts, svc.name, MetricFamily.REQUEST_RATE,
                             "http_requests_errors_per_second", float(err[i])))

    df = pd.DataFrame(rows)
    df.attrs["schema_version"] = SCHEMA_VERSION
    df.attrs["workload"] = workload
    df.attrs["seed"] = seed
    return df


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
def _row(ts, service, fam, metric_name, value, labels=None):
    return {
        "timestamp": ts,
        "service": service,
        "namespace": "default",
        "metric_family": fam.value if hasattr(fam, "value") else fam,
        "metric_name": metric_name,
        "value": float(value),
        "labels": labels or {},
    }


def to_wide(long_df: pd.DataFrame, service: str) -> pd.DataFrame:
    """Pivot a long-format telemetry frame to wide per-service.

    Returns one row per timestamp with columns:
        timestamp, cpu, memory, pod_ready, request_rate, response_time_p95,
        request_rate_errors
    """
    sub = long_df[long_df["service"] == service].copy()
    if sub.empty:
        return pd.DataFrame(columns=["timestamp"])

    cpu = _pick(sub, MetricFamily.CPU, "container_cpu_usage_seconds_total", "cpu")
    mem = _pick(sub, MetricFamily.MEMORY, "container_memory_working_set_bytes", "memory")
    ready = _pick(sub, MetricFamily.POD_READY, "kube_pod_status_ready", "pod_ready")
    rps = _pick(sub, MetricFamily.REQUEST_RATE, "http_requests_per_second", "request_rate")
    err = _pick(sub, MetricFamily.REQUEST_RATE, "http_requests_errors_per_second", "request_rate_errors")
    rt = _pick(sub, MetricFamily.RESPONSE_TIME, "http_request_duration_seconds", "response_time_p95")

    out = (
        cpu.merge(mem, on="timestamp", how="outer")
           .merge(ready, on="timestamp", how="outer")
           .merge(rps, on="timestamp", how="outer")
           .merge(err, on="timestamp", how="outer")
           .merge(rt, on="timestamp", how="outer")
           .sort_values("timestamp")
           .reset_index(drop=True)
    )
    return out


def _pick(df: pd.DataFrame, family: MetricFamily, metric_name: str, out_col: str) -> pd.DataFrame:
    fam = family.value if hasattr(family, "value") else family
    sel = df[(df["metric_family"] == fam) & (df["metric_name"] == metric_name)]
    sel = sel[["timestamp", "value"]].rename(columns={"value": out_col})
    return sel
