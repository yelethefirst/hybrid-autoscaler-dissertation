"""Pydantic models for the Phase 5 A/B experiment design (§3.9).

TrialSpec describes a single trial (one cell of the 2×4 design).
TrialResult is the per-trial output written to the JSONL result log.
TrialPlan is the full experiment plan loaded from YAML before the run.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Autoscaler(str, Enum):
    HPA = "hpa"
    HYBRID = "hybrid"


class Workload(str, Enum):
    BURST = "burst"
    RAMP = "ramp"
    PERIODIC = "periodic"
    TRACE_REPLAY = "trace_replay"


class TrialSpec(BaseModel):
    """One trial in the A/B design."""

    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())

    trial_id: str = Field(description="Unique identifier, e.g. burst-hybrid-001")
    autoscaler: Autoscaler
    workload: Workload
    seed: int = Field(description="RNG seed for reproducible load-generator startup order")
    measure_start_offset_seconds: int = Field(
        default=0,
        description=(
            "Seconds after Locust start before the measurement tool begins — "
            "aligns the wrk2 window with the profile phase under study "
            "(burst: 300 → the burst itself; ramp: 600 → the plateau). DEV-014."
        ),
    )
    locust_file: Path
    locust_host: str = "http://localhost:30080"
    locust_users: int = 500
    locust_spawn_rate: int = 50
    locust_duration_seconds: int
    wrk2_threads: int = 4
    wrk2_connections: int = 100
    wrk2_rate: int = 100
    wrk2_duration_seconds: int
    wrk2_url: str = "http://localhost:30080/"
    controller_config: Optional[Path] = None
    model_registry: Optional[Path] = None
    env: Dict[str, str] = Field(default_factory=dict, description="Extra env vars for Locust")


class TrialResult(BaseModel):
    """Metrics collected from one completed trial."""

    model_config = ConfigDict(use_enum_values=True)

    trial_id: str
    autoscaler: Autoscaler
    workload: Workload
    seed: int
    start_time: datetime
    end_time: datetime

    # Latency (wrk2 output — coordinated-omission-corrected)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    p999_latency_ms: Optional[float] = None
    mean_latency_ms: Optional[float] = None

    # Throughput
    requests_total: Optional[int] = None
    errors_total: Optional[int] = None
    success_rate: Optional[float] = None
    throughput_rps: Optional[float] = None

    # Scaling efficiency (from Prometheus)
    replica_seconds: Optional[float] = Field(
        default=None,
        description="Sum of spec.replicas × time over the trial (resource consumption proxy).",
    )
    peak_replicas: Optional[int] = None
    oscillation_count: Optional[int] = Field(
        default=None,
        description="Number of replica-count direction reversals in the evidence bundle.",
    )
    time_to_stability_seconds: Optional[float] = Field(
        default=None,
        description="Seconds from workload start until replicas first reach steady state.",
    )

    # Forecasting accuracy (from evidence bundle; hybrid only)
    mean_forecast_rmse: Optional[float] = None
    fallback_fraction: Optional[float] = Field(
        default=None,
        description="Fraction of ticks where fallback state was active.",
    )

    # Paths
    evidence_path: Optional[Path] = None
    wrk2_output_path: Optional[Path] = None
    locust_csv_prefix: Optional[Path] = None

    # Raw wrk2 stdout (truncated to 4 KiB for the JSONL log)
    wrk2_raw: Optional[str] = None
    load_tool: Optional[str] = Field(
        default=None,
        description="Measurement tool that produced the latency numbers (wrk2 | hey).",
    )

    # Any error that caused the trial to abort
    error: Optional[str] = None


class TrialPlan(BaseModel):
    """Full experiment plan loaded from YAML before execution."""

    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())

    name: str
    description: str = ""
    created_at: Optional[str] = None
    result_dir: Path = Path("experiments/results")
    n_trials_per_cell: int = Field(
        default=10,
        description="Number of repeated trials per (autoscaler, workload) cell.",
    )
    hpa_manifest: Path = Path("experiments/hpa/frontend-hpa.yaml")
    controller_config: Path = Path("controller/configs/frontend-phase4.yaml")
    model_registry: Optional[Path] = None
    stabilisation_wait_seconds: int = Field(
        default=120,
        description="Seconds to wait for the cluster to return to 1 replica between trials.",
    )
    trials: List[TrialSpec]
    extra: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrialPlan":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
