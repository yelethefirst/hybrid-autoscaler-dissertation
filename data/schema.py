"""Schema definitions and version pinning for telemetry and features (§3.5).

The schema version is committed to git and embedded in Parquet metadata so
later phases (forecasting, SHAP, analysis) can fail loudly on mismatch.
Bumping `SCHEMA_VERSION` is a deliberate, breaking action.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0.0"
"""Bump on any breaking change to TelemetryRow / FeatureSchema layouts."""


class MetricFamily(str, Enum):
    """The five §3.5 metric families. Single source of truth."""

    CPU = "cpu"
    MEMORY = "memory"
    POD_READY = "pod_ready"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"


METRIC_FAMILIES: List[MetricFamily] = list(MetricFamily)


class TelemetryRow(BaseModel):
    """One Prometheus sample in long format.

    Stored as Parquet with the same column names. A telemetry Parquet file
    is therefore a stack of rows of this shape.
    """

    model_config = ConfigDict(use_enum_values=True, frozen=True)

    timestamp: datetime = Field(description="UTC scrape timestamp.")
    service: str = Field(description="Kubernetes Deployment name.")
    namespace: str = Field(default="default")
    metric_family: MetricFamily
    metric_name: str = Field(
        description=(
            "The fully qualified Prometheus metric name, e.g. "
            "'container_cpu_usage_seconds_total'. Lets us distinguish "
            "histogram-quantile values within the response_time family."
        )
    )
    value: float
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional labels (quantile, container, etc.).",
    )


class FeatureSchema(BaseModel):
    """Catalogue of engineered features (§3.5) for a single service-metric.

    This is a *description* of features rather than a row container; the
    actual feature DataFrames are wide format (one column per feature). The
    schema serialises into Parquet metadata for downstream consumers.
    """

    model_config = ConfigDict(frozen=True)

    service: str
    metric_family: MetricFamily
    base_column: str = Field(description="Raw metric column, e.g. 'cpu'.")

    # §3.5: lags at 1, 2, 3, 5, 10 scrape intervals
    lag_intervals: List[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10])

    # §3.5: rolling means and variances at 30s, 60s, 120s
    rolling_windows_seconds: List[int] = Field(default_factory=lambda: [30, 60, 120])

    # §3.5: first difference + log first difference
    diff_first: bool = True
    diff_first_log: bool = True

    # §3.5: upstream request-rate exogenous feature for downstream services
    upstream_services: List[str] = Field(default_factory=list)

    sample_interval_seconds: int = 15
    schema_version: str = SCHEMA_VERSION

    def expected_feature_names(self) -> List[str]:
        """Return the deterministic list of feature column names this schema produces."""
        base = self.base_column
        names: List[str] = [base]  # the raw value at t
        for k in self.lag_intervals:
            names.append(f"{base}_lag{k}")
        for w in self.rolling_windows_seconds:
            n = w // self.sample_interval_seconds
            names.append(f"{base}_rmean{w}s")
            names.append(f"{base}_rvar{w}s")
            if n <= 0:
                raise ValueError(
                    f"rolling window {w}s is below sample interval "
                    f"{self.sample_interval_seconds}s"
                )
        if self.diff_first:
            names.append(f"{base}_diff1")
        if self.diff_first_log:
            names.append(f"{base}_logdiff1")
        for upstream in self.upstream_services:
            names.append(f"{upstream}_request_rate_t")
        return names
