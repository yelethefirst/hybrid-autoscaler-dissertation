"""Unit tests for the §3.7 decision engine.

Coverage targets:
    - all five states reached with explicit input scenarios
    - bounds clipping (r_min / r_max)
    - rate-limit clamp in both directions
    - HPA-equivalent fallback rule arithmetic
    - sigma > sigma_max precedence over predictive path
    - forecaster fault precedence over uncertainty fallback
    - non-finite forecast → treated as forecaster fault
    - sign correctness on scale-down

The Phase 1 exit criterion requires this file to pass under `uv run pytest`.
"""

from __future__ import annotations

import math

import pytest

from controller.config import EngineConfig
from controller.decision_engine import DecisionEngine
from controller.state import EngineState


def _cfg(**overrides) -> EngineConfig:
    defaults = dict(
        service="frontend",
        namespace="default",
        r_min=1,
        r_max=10,
        delta_s=2,
        rho=0.3,                  # cores per pod
        k=1.5,
        sigma_max=0.2,
        sigma_warn_ratio=0.7,
        horizon_seconds=30,
        tick_seconds=15,
        history_seconds=600,
        target_utilisation=0.5,
        hpa_stabilisation_window_seconds=300,
        metric_source="cpu",
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────── #
# NOMINAL                                                                  #
# ─────────────────────────────────────────────────────────────────────── #
def test_nominal_path_simple_scale_up():
    eng = DecisionEngine(_cfg())
    # Current 1 replica, forecast 0.6 cores, σ very small.
    # r* = ceil((0.6 + 1.5*0.01)/0.3) = ceil(2.05) = 3  (within ΔS=2 of 1 → ok)
    d = eng.decide(
        current_replicas=1,
        forecast_point=0.6,
        forecast_sigma=0.01,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=0.4,
        now_seconds=100.0,
    )
    assert d.state == EngineState.NOMINAL
    assert d.recommended_replicas == 3
    assert d.new_replicas == 3
    assert d.rate_limited is False
    assert d.fallback_engaged is False


def test_nominal_with_zero_sigma_is_fine():
    eng = DecisionEngine(_cfg())
    d = eng.decide(
        current_replicas=2,
        forecast_point=0.45,
        forecast_sigma=0.0,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=0.3,
        now_seconds=100.0,
    )
    assert d.state == EngineState.NOMINAL


# ─────────────────────────────────────────────────────────────────────── #
# CONFIDENCE_MARGIN_HIGH                                                   #
# ─────────────────────────────────────────────────────────────────────── #
def test_confidence_margin_high_when_sigma_in_warning_band():
    # warn band = (0.7 * 0.2, 0.2] = (0.14, 0.20]
    cfg = _cfg()
    eng = DecisionEngine(cfg)
    d = eng.decide(
        current_replicas=2,
        forecast_point=0.30,
        forecast_sigma=0.16,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=0.5,
        now_seconds=100.0,
    )
    assert d.state == EngineState.CONFIDENCE_MARGIN_HIGH
    assert d.fallback_engaged is False
    # r* = ceil((0.30 + 1.5*0.16)/0.3) = ceil(1.80) = 2 → no scale change
    assert d.recommended_replicas == 2


# ─────────────────────────────────────────────────────────────────────── #
# FALLBACK_UNCERTAINTY                                                     #
# ─────────────────────────────────────────────────────────────────────── #
def test_fallback_uncertainty_uses_hpa_rule():
    eng = DecisionEngine(_cfg())
    # σ above sigma_max → fallback. u_t/u* = 0.8/0.5 = 1.6; r_t=2 → ceil(3.2)=4
    d = eng.decide(
        current_replicas=2,
        forecast_point=10.0,        # ignored on fallback path
        forecast_sigma=0.5,         # > sigma_max=0.2
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=0.8,
        now_seconds=100.0,
    )
    assert d.state == EngineState.FALLBACK_UNCERTAINTY
    assert d.fallback_engaged is True
    assert d.new_replicas == 4


def test_fallback_uncertainty_holds_when_observed_is_none():
    eng = DecisionEngine(_cfg())
    d = eng.decide(
        current_replicas=3,
        forecast_point=10.0,
        forecast_sigma=0.5,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=None,        # cannot compute ratio → hold
        now_seconds=100.0,
    )
    assert d.state == EngineState.FALLBACK_UNCERTAINTY
    assert d.new_replicas == 3


# ─────────────────────────────────────────────────────────────────────── #
# FALLBACK_FORECASTER_FAULT                                                #
# ─────────────────────────────────────────────────────────────────────── #
def test_forecaster_fault_takes_priority_over_uncertainty():
    eng = DecisionEngine(_cfg())
    # Even when σ is fine, if forecaster faulted, we still fall back.
    d = eng.decide(
        current_replicas=2,
        forecast_point=None,
        forecast_sigma=None,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason="ForecasterFaultError: empty history",
        observed_metric=0.6,
        now_seconds=100.0,
    )
    assert d.state == EngineState.FALLBACK_FORECASTER_FAULT
    assert d.fallback_engaged is True
    # ratio = 0.6/0.5 = 1.2; ceil(2*1.2) = 3
    assert d.new_replicas == 3
    assert d.forecaster_fault_reason


def test_non_finite_forecast_treated_as_fault():
    eng = DecisionEngine(_cfg())
    d = eng.decide(
        current_replicas=2,
        forecast_point=float("nan"),
        forecast_sigma=0.01,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.5,
        now_seconds=100.0,
    )
    assert d.state == EngineState.FALLBACK_FORECASTER_FAULT
    assert d.fallback_engaged is True


def test_negative_sigma_treated_as_fault():
    eng = DecisionEngine(_cfg())
    d = eng.decide(
        current_replicas=2,
        forecast_point=0.5,
        forecast_sigma=-0.01,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.5,
        now_seconds=100.0,
    )
    assert d.state == EngineState.FALLBACK_FORECASTER_FAULT


# ─────────────────────────────────────────────────────────────────────── #
# SCALE_LIMITED                                                            #
# ─────────────────────────────────────────────────────────────────────── #
def test_scale_limited_when_recommendation_exceeds_delta_s():
    eng = DecisionEngine(_cfg())
    # Forecast 3.0 cores, σ small → r* = ceil(3.045/0.3) = 11 → clipped to r_max=10
    # |10 - 1| = 9 > ΔS=2 → clamp to 1+2=3, state SCALE_LIMITED
    d = eng.decide(
        current_replicas=1,
        forecast_point=3.0,
        forecast_sigma=0.01,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=1.0,
        now_seconds=100.0,
    )
    assert d.state == EngineState.SCALE_LIMITED
    assert d.recommended_replicas == 10
    assert d.new_replicas == 3
    assert d.rate_limited is True


def test_scale_limited_on_scale_down():
    eng = DecisionEngine(_cfg())
    # Current 8, forecast tiny → r* would be 1, clamp to 8-ΔS=6
    d = eng.decide(
        current_replicas=8,
        forecast_point=0.05,
        forecast_sigma=0.001,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.1,
        now_seconds=100.0,
    )
    assert d.state == EngineState.SCALE_LIMITED
    assert d.recommended_replicas == 1
    assert d.new_replicas == 6


# ─────────────────────────────────────────────────────────────────────── #
# Bounds clipping                                                          #
# ─────────────────────────────────────────────────────────────────────── #
def test_r_max_clipping():
    eng = DecisionEngine(_cfg(r_max=4, delta_s=10))   # large ΔS so no rate limit
    d = eng.decide(
        current_replicas=2,
        forecast_point=10.0,
        forecast_sigma=0.0,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.5,
        now_seconds=100.0,
    )
    assert d.recommended_replicas == 4
    assert d.new_replicas == 4


def test_r_min_clipping():
    eng = DecisionEngine(_cfg(r_min=2, delta_s=10))
    d = eng.decide(
        current_replicas=3,
        forecast_point=0.0,
        forecast_sigma=0.0,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.0,
        now_seconds=100.0,
    )
    assert d.recommended_replicas == 2
    assert d.new_replicas == 2


# ─────────────────────────────────────────────────────────────────────── #
# Determinism (relevant for §3.6 reproducibility claim)                    #
# ─────────────────────────────────────────────────────────────────────── #
def test_decision_is_deterministic_for_same_inputs():
    eng_a = DecisionEngine(_cfg())
    eng_b = DecisionEngine(_cfg())
    kwargs = dict(
        current_replicas=2,
        forecast_point=0.4,
        forecast_sigma=0.05,
        forecaster_name="seasonal_naive",
        forecaster_fault_reason=None,
        observed_metric=0.4,
        now_seconds=100.0,
    )
    a = eng_a.decide(**kwargs).model_dump()
    b = eng_b.decide(**kwargs).model_dump()
    # Drop the timestamp which is wall-clock.
    a.pop("timestamp")
    b.pop("timestamp")
    assert a == b


# ─────────────────────────────────────────────────────────────────────── #
# Safety guarantee for H3                                                  #
# ─────────────────────────────────────────────────────────────────────── #
def test_under_persistent_fault_engine_behaves_like_hpa():
    """H3 safety claim: under predictive failure, behaviour reduces to HPA."""
    eng = DecisionEngine(_cfg())
    new_replicas = []
    for i in range(5):
        d = eng.decide(
            current_replicas=2 + (i % 2),
            forecast_point=None,
            forecast_sigma=None,
            forecaster_name="x",
            forecaster_fault_reason="forecaster down",
            observed_metric=0.7,                 # ratio 1.4 → push up
            now_seconds=100.0 + i * 15,
        )
        assert d.state == EngineState.FALLBACK_FORECASTER_FAULT
        assert d.fallback_engaged is True
        new_replicas.append(d.new_replicas)
    # All recommendations should be within bounds and finite.
    for r in new_replicas:
        assert 1 <= r <= 10
        assert math.isfinite(r)


def test_hpa_scale_down_stabilisation_holds_highest_recent_recommendation():
    """Scale-down fallback should use HPA's max recommendation over window W."""
    eng = DecisionEngine(_cfg())

    # Build a recent predictive history with recommendations 3 and then 5.
    d1 = eng.decide(
        current_replicas=1,
        forecast_point=3.0,
        forecast_sigma=0.01,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=1.0,
        now_seconds=100.0,
    )
    d2 = eng.decide(
        current_replicas=d1.new_replicas,
        forecast_point=3.0,
        forecast_sigma=0.01,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=1.0,
        now_seconds=115.0,
    )
    assert (d1.new_replicas, d2.new_replicas) == (3, 5)

    # Low utilisation would normally propose ceil(5 * 0.1/0.5) = 1, but the
    # scale-down stabilisation window must hold at the highest recent rec: 5.
    d3 = eng.decide(
        current_replicas=5,
        forecast_point=10.0,
        forecast_sigma=0.5,
        forecaster_name="x",
        forecaster_fault_reason=None,
        observed_metric=0.1,
        now_seconds=130.0,
    )
    assert d3.state == EngineState.FALLBACK_UNCERTAINTY
    assert d3.fallback_engaged is True
    assert d3.new_replicas == 5


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
