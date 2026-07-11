"""Seasonal Naive forecaster (§3.6 baseline).

The Seasonal Naive forecaster predicts that the value at time t+h equals the
value one season-period ago, at the same within-period offset:

    f̂(t + h) = x(t + h - P)

where P is the seasonal period. The prediction interval half-width σ̂ is the
standard deviation of the in-sample one-step seasonal-naive residuals:

    residual(t) = x(t) - x(t - P)
    σ̂          = std(residual)

This is the Hyndman & Athanasopoulos (2021) baseline that all other
forecasters must beat to be selected (Hypothesis H1, §1.5).

Justification for being the vertical-slice forecaster
-----------------------------------------------------
It has no hyperparameters, no training, no random seed; it is therefore
trivially deterministic and lets us validate the end-to-end loop
(predict → decide → actuate → log) without confounds from model behaviour.
"""

from __future__ import annotations

import math

import pandas as pd

from .base import Forecast, Forecaster, ForecasterFaultError


class SeasonalNaive(Forecaster):
    """Seasonal Naive forecaster with residual-based σ.

    Parameters
    ----------
    period_seconds : int
        The seasonal period P, in seconds. For workloads like the §3.9
        periodic profile (60-second sinusoid) this is 60. For a non-seasonal
        signal, set this to the sample interval (15 s) and Seasonal Naive
        degenerates to a one-step persistence baseline, which is still a
        defensible reactive comparator.
    sample_interval_seconds : int
        Cadence of the input series (15 s per §3.5).
    min_history_samples : int
        Minimum number of clean (non-NaN) samples required before producing
        a forecast. Defaults to period_samples + 2 — one full period to look
        back over for the point estimate, plus two more samples so the
        residual std (with ddof=1) is defined. A forecaster fault is raised
        if fewer samples are present.
    """

    name = "seasonal_naive"

    def __init__(
        self,
        period_seconds: int = 60,
        sample_interval_seconds: int = 15,
        min_history_samples: int | None = None,
    ):
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        if period_seconds % sample_interval_seconds != 0:
            raise ValueError(
                f"period_seconds ({period_seconds}) must be a whole multiple "
                f"of sample_interval_seconds ({sample_interval_seconds})"
            )

        self.period_seconds = period_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.period_samples = period_seconds // sample_interval_seconds
        self.min_history_samples = min_history_samples or (self.period_samples + 2)

    # ------------------------------------------------------------------ #
    # Forecaster interface                                                #
    # ------------------------------------------------------------------ #
    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if horizon_seconds > self.period_seconds:
            # Beyond one season the seasonal-naive prediction wraps around;
            # the dissertation only forecasts at h ∈ {30, 60} s with P ≥ 60.
            raise ValueError(
                f"horizon_seconds ({horizon_seconds}) must be ≤ period "
                f"({self.period_seconds})"
            )

        clean = history.dropna()
        if len(clean) == 0:
            raise ForecasterFaultError("history is empty after dropping NaN")

        if len(clean) < self.min_history_samples:
            raise ForecasterFaultError(
                f"history has {len(clean)} clean samples; minimum is "
                f"{self.min_history_samples} (2 full periods of "
                f"{self.period_samples} samples each)"
            )

        horizon_samples = max(1, round(horizon_seconds / self.sample_interval_seconds))

        # Hyndman & Athanasopoulos (2021) Seasonal Naive:
        #   ŷ(T+h | T) = y(T + h − m·⌈h/m⌉),
        # where m = period_samples. For h ≤ m, ⌈h/m⌉ = 1, so the prediction is
        # y(T + h − m), which in negative indexing is clean.iloc[-(m − h + 1)].
        idx_back = self.period_samples - horizon_samples + 1
        if idx_back < 1 or idx_back > len(clean):
            raise ForecasterFaultError(
                f"cannot look back {idx_back} samples (have {len(clean)})"
            )
        point = float(clean.iloc[-idx_back])

        # σ from in-sample one-step seasonal-naive residuals.
        residuals = clean.diff(self.period_samples).dropna()
        if len(residuals) < 2:
            raise ForecasterFaultError("not enough residuals to estimate sigma")
        sigma = float(residuals.std(ddof=1))

        if not math.isfinite(point) or not math.isfinite(sigma):
            raise ForecasterFaultError(
                f"non-finite forecast (point={point}, sigma={sigma})"
            )
        # σ is by construction non-negative; clamp tiny negatives from float noise.
        sigma = max(sigma, 0.0)
        return Forecast(point=point, sigma=sigma)

