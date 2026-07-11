"""Pydantic configuration for the decision engine and control loop.

All parameters are explicit (no implicit defaults that affect measured
behaviour). Configurations are loaded from YAML and committed to version
control — see `controller/configs/frontend-local.yaml` for an example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EngineConfig(BaseModel):
    """Per-service configuration for the constrained decision engine (§3.7).

    Parameter names below match the dissertation notation directly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ─── target service ──────────────────────────────────────────────────
    service: str = Field(description="Kubernetes Deployment name to scale.")
    namespace: str = Field(default="default")

    # ─── §3.7 engine parameters ──────────────────────────────────────────
    r_min: int = Field(ge=1, description="Lower replica safety bound.")
    r_max: int = Field(ge=1, description="Upper replica safety bound.")
    delta_s: int = Field(
        ge=1,
        description="ΔS — maximum |replica change| per tick. Rate-limit clamp.",
    )
    rho: float = Field(
        gt=0,
        description=(
            "Per-pod target capacity in the same units as the forecast. "
            "If the forecast is total CPU rate (cores), ρ is target cores per "
            "pod (e.g. 0.3). If the forecast is total request rate (req/s), "
            "ρ is target req/s per pod."
        ),
    )
    k: float = Field(
        default=1.5,
        ge=0,
        description="Confidence-margin coefficient. §3.7 default is 1.5.",
    )
    sigma_max: float = Field(
        gt=0,
        description=(
            "Uncertainty fallback threshold. When σ̂(t+h) > sigma_max the "
            "engine falls back to the HPA-equivalent rule."
        ),
    )
    sigma_warn_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "CONFIDENCE_MARGIN_HIGH triggers when σ̂(t+h) > sigma_warn_ratio · "
            "sigma_max. Default 0.7 (70%% of the fallback threshold)."
        ),
    )

    # ─── horizons + cadence (§3.5, §3.6) ─────────────────────────────────
    horizon_seconds: int = Field(
        default=30,
        ge=15,
        description="Forecast horizon h, typically 30 or 60 s per §3.6.",
    )
    tick_seconds: int = Field(
        default=15,
        ge=1,
        description="Control-loop period. Matches Prometheus scrape (§3.5).",
    )
    history_seconds: int = Field(
        default=600,
        ge=60,
        description="How much history to pass to the forecaster each tick.",
    )

    # ─── HPA-equivalent fallback parameters (§3.7) ───────────────────────
    target_utilisation: float = Field(
        default=0.5,
        gt=0,
        description=(
            "u* in the HPA-equivalent fallback rule r_(t+1) = ⌈r_t · u_t / u*⌉. "
            "For CPU-based fallback, this is the target average CPU utilisation."
        ),
    )
    hpa_stabilisation_window_seconds: int = Field(
        default=300,
        ge=0,
        description="W in §3.7 — Kubernetes HPA default is 300 s.",
    )

    # ─── metric source (Phase 1) ─────────────────────────────────────────
    metric_source: Literal["cpu"] = Field(
        default="cpu",
        description=(
            "'cpu' uses container_cpu_usage_seconds_total rate (the HPA "
            "default signal). 'requests' was accepted here but the control "
            "loop only ever reads CPU history, so a request-rate model would "
            "silently receive CPU inputs — rejected at validation until the "
            "loop supports it (2026-07-05 code review)."
        ),
    )

    # ─── evidence bundle output path ─────────────────────────────────────
    evidence_path: Path = Field(
        default=Path("experiments/results/evidence.jsonl"),
        description="JSONL file the evidence bundle writer appends to.",
    )

    # ─── validators ──────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _bounds_consistent(self) -> "EngineConfig":
        if self.r_max < self.r_min:
            raise ValueError(f"r_max ({self.r_max}) < r_min ({self.r_min})")
        if self.horizon_seconds < self.tick_seconds:
            raise ValueError(
                f"horizon_seconds ({self.horizon_seconds}) must be ≥ "
                f"tick_seconds ({self.tick_seconds})"
            )
        if self.history_seconds < 2 * self.horizon_seconds:
            raise ValueError(
                "history_seconds must be at least 2× horizon_seconds so a "
                "minimal Seasonal Naive baseline can run."
            )
        return self

    # ─── loader ──────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "EngineConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
