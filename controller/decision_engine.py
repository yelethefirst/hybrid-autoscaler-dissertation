"""Constrained decision engine — §3.7.

Pure function of (forecast, current state) → Decision. No I/O. This is the
single most safety-critical component in the controller: it is the layer
that prevents prediction error from amplifying into outage.

The §3.7 formula, reproduced verbatim:

    r*(t+h) = clip( ⌈ ( f̂(t+h) + k · σ̂(t+h) ) ÷ ρ_t ⌉ , r_min , r_max )
    Δr_t    = sign( r*(t+h) − r_t ) · min( | r*(t+h) − r_t | , ΔS )
    r_(t+1) = r_t + Δr_t

If σ̂(t+h) exceeds σ_max, or the forecaster reports failure, the engine
returns instead the HPA-equivalent rule:

    r_(t+1) = ⌈ r_t · u_t ÷ u* ⌉

The §3.7 stabilisation window W (300 s by default) is honoured for the
HPA-equivalent path. Phase 1 implements a simple recent-recommendation
ring buffer; Phase 4 will harden this to match Kubernetes HPA exactly.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

from .config import EngineConfig
from .state import Decision, EngineState


class DecisionEngine:
    """Stateless-per-tick decision engine with a tiny stabilisation buffer.

    The engine itself holds only a small ring buffer of recent recommendations
    so the HPA-equivalent fallback can respect the §3.7 stabilisation window.
    Everything else is computed per call from the inputs.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        # Ring buffer of (timestamp_seconds, recommended_replicas) for the
        # stabilisation window. Used only by the HPA-equivalent fallback path.
        buf_size = max(
            1, config.hpa_stabilisation_window_seconds // max(config.tick_seconds, 1)
        )
        self._recent_recs: Deque[Tuple[float, int]] = deque(maxlen=buf_size)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def decide(
        self,
        *,
        current_replicas: int,
        forecast_point: Optional[float],
        forecast_sigma: Optional[float],
        forecaster_name: Optional[str],
        forecaster_fault_reason: Optional[str],
        observed_metric: Optional[float],
        now_seconds: float,
    ) -> Decision:
        """Run one tick of the §3.7 decision engine.

        Parameters
        ----------
        current_replicas : int
            r_t — current replica count of the target Deployment.
        forecast_point, forecast_sigma, forecaster_name : optional
            From the forecaster. None iff the forecaster faulted.
        forecaster_fault_reason : optional
            Set iff the forecaster raised; non-None forces FALLBACK_FORECASTER_FAULT.
        observed_metric : optional
            u_t — current observed signal (e.g. average per-pod CPU
            utilisation as a fraction). Used by the HPA-equivalent fallback.
        now_seconds : float
            Monotonic timestamp for the stabilisation buffer.
        """
        c = self.config

        # ─── 1. Forecaster fault → FALLBACK_FORECASTER_FAULT ─────────────
        if forecaster_fault_reason is not None:
            new = self._hpa_equivalent(current_replicas, observed_metric, now_seconds)
            return Decision(
                service=c.service,
                namespace=c.namespace,
                horizon_seconds=c.horizon_seconds,
                forecast_point=None,
                forecast_sigma=None,
                forecaster_name=forecaster_name,
                forecaster_fault_reason=forecaster_fault_reason,
                current_replicas=current_replicas,
                observed_metric=observed_metric,
                recommended_replicas=new,
                new_replicas=new,
                state=EngineState.FALLBACK_FORECASTER_FAULT,
                rate_limited=False,
                fallback_engaged=True,
            )

        # Defensive: missing or invalid forecast values shouldn't occur if the
        # forecaster honoured its contract, but treat them as a fault.
        if (
            forecast_point is None
            or forecast_sigma is None
            or not math.isfinite(forecast_point)
            or not math.isfinite(forecast_sigma)
            or forecast_sigma < 0
        ):
            new = self._hpa_equivalent(current_replicas, observed_metric, now_seconds)
            return Decision(
                service=c.service,
                namespace=c.namespace,
                horizon_seconds=c.horizon_seconds,
                forecast_point=None,
                forecast_sigma=None,
                forecaster_name=forecaster_name,
                forecaster_fault_reason="non-finite or negative forecast output",
                current_replicas=current_replicas,
                observed_metric=observed_metric,
                recommended_replicas=new,
                new_replicas=new,
                state=EngineState.FALLBACK_FORECASTER_FAULT,
                rate_limited=False,
                fallback_engaged=True,
            )

        # ─── 2. Uncertainty above σ_max → FALLBACK_UNCERTAINTY ───────────
        if forecast_sigma > c.sigma_max:
            new = self._hpa_equivalent(current_replicas, observed_metric, now_seconds)
            return Decision(
                service=c.service,
                namespace=c.namespace,
                horizon_seconds=c.horizon_seconds,
                forecast_point=forecast_point,
                forecast_sigma=forecast_sigma,
                forecaster_name=forecaster_name,
                current_replicas=current_replicas,
                observed_metric=observed_metric,
                recommended_replicas=new,
                new_replicas=new,
                state=EngineState.FALLBACK_UNCERTAINTY,
                rate_limited=False,
                fallback_engaged=True,
            )

        # ─── 3. Predictive path — §3.7 main formula ──────────────────────
        # r*(t+h) = clip( ⌈ ( f̂ + k·σ̂ ) / ρ ⌉ , r_min , r_max )
        biased_demand = forecast_point + c.k * forecast_sigma
        raw_replicas = math.ceil(biased_demand / c.rho)
        r_star = max(c.r_min, min(c.r_max, raw_replicas))

        # Δr = sign(r* − r_t) · min(|r* − r_t|, ΔS); then r_(t+1) = r_t + Δr.
        delta_raw = r_star - current_replicas
        delta_clamped = max(-c.delta_s, min(c.delta_s, delta_raw))
        new = current_replicas + delta_clamped
        rate_limited = abs(delta_raw) > c.delta_s

        # Record this recommendation for stabilisation tracking.
        self._recent_recs.append((now_seconds, new))

        # ─── 4. State precedence: SCALE_LIMITED > CMH > NOMINAL ──────────
        if rate_limited:
            state = EngineState.SCALE_LIMITED
        elif forecast_sigma > c.sigma_warn_ratio * c.sigma_max:
            state = EngineState.CONFIDENCE_MARGIN_HIGH
        else:
            state = EngineState.NOMINAL

        return Decision(
            service=c.service,
            namespace=c.namespace,
            horizon_seconds=c.horizon_seconds,
            forecast_point=forecast_point,
            forecast_sigma=forecast_sigma,
            forecaster_name=forecaster_name,
            current_replicas=current_replicas,
            observed_metric=observed_metric,
            recommended_replicas=r_star,
            new_replicas=new,
            state=state,
            rate_limited=rate_limited,
            fallback_engaged=False,
        )

    # ------------------------------------------------------------------ #
    # HPA-equivalent fallback                                             #
    # ------------------------------------------------------------------ #
    def _hpa_equivalent(
        self,
        current_replicas: int,
        observed_metric: Optional[float],
        now_seconds: float,
    ) -> int:
        """Compute r_(t+1) under the §3.7 HPA-equivalent rule.

        r_(t+1) = ⌈ r_t · u_t / u* ⌉ , bounded by [r_min, r_max] and respecting
        the stabilisation window W: for scale-down, the new replica count is
        the *maximum* recent recommendation within W (HPA's default behaviour);
        for scale-up, no stabilisation delay is applied (also HPA default).
        """
        c = self.config
        if observed_metric is None or not math.isfinite(observed_metric):
            # Cannot compute the ratio: hold steady within bounds.
            return max(c.r_min, min(c.r_max, current_replicas))

        ratio = observed_metric / c.target_utilisation
        proposed = math.ceil(current_replicas * ratio)
        proposed = max(c.r_min, min(c.r_max, proposed))

        # Stabilisation window: for scale-down, take max over recent recs in W.
        if proposed < current_replicas and self._recent_recs:
            cutoff = now_seconds - c.hpa_stabilisation_window_seconds
            recent_window = [r for (t, r) in self._recent_recs if t >= cutoff]
            if recent_window:
                proposed = max(proposed, max(recent_window))
                proposed = max(c.r_min, min(c.r_max, proposed))

        # Record the fallback recommendation too.
        self._recent_recs.append((now_seconds, proposed))
        return proposed
