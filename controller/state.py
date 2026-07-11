"""State enum and Decision result model for the decision engine (§3.7, §3.8).

Five mutually exclusive states are defined in §3.7 Figure 3.2. State precedence
when more than one condition is technically true:

    FALLBACK_FORECASTER_FAULT   > all others (safety: forecaster cannot be trusted)
    FALLBACK_UNCERTAINTY        > predictive states (σ above threshold)
    SCALE_LIMITED               > CONFIDENCE_MARGIN_HIGH > NOMINAL
                                   (rate-limit clamp is the next most operationally
                                    salient signal after the two fallbacks)

`Decision` is the per-tick record that becomes the §3.8 evidence-bundle line.
It is a Pydantic model so it serialises cleanly to the JSONL audit store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EngineState(str, Enum):
    """Five-state machine of the constrained decision engine (§3.7)."""

    NOMINAL = "NOMINAL"
    """Forecaster healthy, σ below the warning threshold, predictive control active."""

    CONFIDENCE_MARGIN_HIGH = "CONFIDENCE_MARGIN_HIGH"
    """σ approaching σ_max; over-provision (k·σ) term is materially active."""

    FALLBACK_UNCERTAINTY = "FALLBACK_UNCERTAINTY"
    """σ above σ_max; HPA-equivalent rule replaces the predictive recommendation."""

    FALLBACK_FORECASTER_FAULT = "FALLBACK_FORECASTER_FAULT"
    """Forecaster raised; HPA-equivalent rule used, alert flagged for review."""

    SCALE_LIMITED = "SCALE_LIMITED"
    """Recommendation magnitude exceeded ΔS per period; actuation clamped."""


class Decision(BaseModel):
    """A single scaling decision and the evidence behind it (§3.8 bundle row)."""

    model_config = ConfigDict(use_enum_values=True, frozen=True)

    # ─── identification ──────────────────────────────────────────────────
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    service: str
    namespace: str
    horizon_seconds: int

    # ─── inputs (forecast) ───────────────────────────────────────────────
    forecast_point: Optional[float] = Field(
        default=None,
        description="f̂(t+h). None if forecaster faulted.",
    )
    forecast_sigma: Optional[float] = Field(
        default=None, description="σ̂(t+h). None if forecaster faulted."
    )
    forecaster_name: Optional[str] = None
    forecaster_fault_reason: Optional[str] = None

    # ─── inputs (observed) ───────────────────────────────────────────────
    current_replicas: int
    observed_metric: Optional[float] = Field(
        default=None,
        description="u_t used by the HPA-equivalent fallback rule.",
    )

    # ─── outputs ─────────────────────────────────────────────────────────
    recommended_replicas: int = Field(
        description="r*(t+h) from §3.7, BEFORE the rate-limit clamp."
    )
    new_replicas: int = Field(
        description="r_(t+1) actually applied, AFTER clip + rate-limit (or fallback rule)."
    )
    state: EngineState
    rate_limited: bool = Field(
        default=False,
        description="True iff the predictive recommendation was clamped by ΔS.",
    )
    fallback_engaged: bool = Field(
        default=False,
        description="True iff HPA-equivalent fallback rule produced the new_replicas.",
    )

    # ─── for downstream phases (SHAP + LLM) ──────────────────────────────
    feature_window_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Summary stats of the input feature window. Populated in Phase 1; "
            "Phase 6 replaces with per-decision SHAP attribution vector."
        ),
    )
