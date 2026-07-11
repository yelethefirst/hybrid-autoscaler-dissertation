"""Holt-Winters exponential smoothing forecaster (§3.6).

§3.6: "Holt-Winters exponential smoothing with optimised additive trend
and damped seasonality."

Implementation uses `statsmodels.tsa.holtwinters.ExponentialSmoothing`,
which fits α, β, γ and φ (damping) by L-BFGS-B. The seasonal period is
inferred from the sample-interval-to-seasonal-period ratio supplied by
the caller (default 60 s seasonality at 15 s cadence = 4 samples/period).

Prediction interval
-------------------
§3.6 specifies parametric intervals for SARIMA and Holt-Winters. After
fitting, the in-sample residuals are used to compute σ̂ via
`forecasting.intervals.parametric_sigma`, then attached to every forecast.

Refitting strategy
------------------
Holt-Winters is cheap to refit. Following Hyndman & Athanasopoulos (2021)
practice for short horizons and small state, we refit on every prediction
call — this keeps the model fully up-to-date with the observed history,
which matters when the workload changes regime (§3.9 burst). For long
campaigns or expensive series, callers may set `refit_each_predict=False`
and call `fit()` explicitly before `predict()`.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import Forecast, Forecaster, ForecasterFaultError
from .intervals import parametric_sigma


class HoltWinters(Forecaster):
    """Holt-Winters with additive trend, damped seasonality (§3.6)."""

    name = "holt_winters"

    def __init__(
        self,
        period_seconds: int = 60,
        sample_interval_seconds: int = 15,
        *,
        refit_each_predict: bool = True,
        min_history_samples: Optional[int] = None,
    ):
        if period_seconds <= 0 or sample_interval_seconds <= 0:
            raise ValueError("period_seconds and sample_interval_seconds must be positive")
        if period_seconds % sample_interval_seconds != 0:
            raise ValueError(
                f"period_seconds ({period_seconds}) must be a whole multiple of "
                f"sample_interval_seconds ({sample_interval_seconds})"
            )

        self.period_seconds = period_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.period_samples = period_seconds // sample_interval_seconds
        # §3.6 / Hyndman: need ≥ 2 seasonal periods to fit seasonal HW.
        self.min_history_samples = min_history_samples or (2 * self.period_samples + 2)
        self.refit_each_predict = refit_each_predict
        self._fitted_results = None
        self._fitted_n = 0

    # ------------------------------------------------------------------ #
    # Forecaster interface                                                #
    # ------------------------------------------------------------------ #
    def fit(self, history: pd.Series) -> "HoltWinters":
        clean = history.dropna().astype(float)
        if len(clean) < self.min_history_samples:
            raise ForecasterFaultError(
                f"Holt-Winters needs ≥ {self.min_history_samples} clean samples "
                f"(have {len(clean)}) for period {self.period_samples}"
            )
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as e:
            raise RuntimeError(
                "HoltWinters requires statsmodels. Install via uv: `uv sync`."
            ) from e

        with warnings.catch_warnings():
            # statsmodels emits noisy ConvergenceWarnings on short series.
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                clean.values,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=self.period_samples,
                initialization_method="estimated",
            )
            self._fitted_results = model.fit(optimized=True, use_brute=False)
        self._fitted_n = len(clean)
        return self

    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")

        if self.refit_each_predict or self._fitted_results is None:
            self.fit(history)

        steps_ahead = max(1, round(horizon_seconds / self.sample_interval_seconds))
        try:
            forecast_arr = self._fitted_results.forecast(steps=steps_ahead)
        except Exception as e:
            raise ForecasterFaultError(f"Holt-Winters forecast failed: {e}") from e

        point = float(np.asarray(forecast_arr)[-1])
        if not math.isfinite(point):
            raise ForecasterFaultError(f"Holt-Winters produced non-finite point: {point}")

        # Parametric 1σ interval from in-sample residuals (§3.6).
        residuals = np.asarray(self._fitted_results.resid).ravel()
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) < 2:
            raise ForecasterFaultError("Holt-Winters has too few residuals for σ")
        sigma = parametric_sigma(residuals)
        return Forecast(point=point, sigma=sigma)

    def shap_attribution(
        self, history: pd.Series, horizon_seconds: int, top_k: int = 10
    ) -> dict:
        """Decompose the Holt-Winters forecast into level/trend/seasonal components.

        Returns the additive contribution of each component to the point
        forecast, expressed as a pseudo-attribution (analogous to SHAP values
        but derived analytically from the model's state equations).
        """
        if self._fitted_results is None:
            return {}
        try:
            steps = max(1, round(horizon_seconds / self.sample_interval_seconds))
            res = self._fitted_results
            level = float(np.asarray(res.level).ravel()[-1])
            # Damped trend: φ^1 + φ^2 + … + φ^steps
            params = res.params if isinstance(res.params, dict) else dict(res.params)
            phi = float(params.get("damping_trend", 1.0))
            trend_last = float(np.asarray(res.trend).ravel()[-1])
            if abs(phi - 1.0) < 1e-6:
                trend_sum = steps * trend_last
            else:
                trend_sum = trend_last * phi * (1 - phi**steps) / (1 - phi)
            # Seasonal: pick the seasonal index steps ahead
            seasonal_arr = np.asarray(res.season).ravel()
            n_season = len(seasonal_arr)
            idx = (n_season - 1 + steps) % n_season if n_season else 0
            seasonal = float(seasonal_arr[idx]) if n_season else 0.0

            return {
                "method": "hw_components",
                "top_features": {
                    "level": round(level, 6),
                    "trend": round(trend_sum, 6),
                    "seasonal": round(seasonal, 6),
                },
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Parsimony tiebreaker                                                #
    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        # α, β, γ, φ + initial level + initial trend + (period_samples) initial seasonals
        return 6 + self.period_samples
