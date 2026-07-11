"""SARIMA forecaster with AICc stepwise order selection (§3.6).

§3.6: "SARIMA with orders selected by AICc-driven stepwise search."

Implementation uses `statsmodels.tsa.statespace.SARIMAX` to fit candidate
(p, d, q) × (P, D, Q, m) orders and selects the lowest-AICc model. A bounded
grid is used by default rather than full stepwise (pmdarima.auto_arima) to
keep the dependency surface small; the grid covers the orders the §3.6
literature typically converges on for sub-minute autoscaling telemetry.

Prediction interval
-------------------
§3.6 specifies parametric intervals for SARIMA. The fitted SARIMAX result
exposes `get_forecast(...).conf_int(alpha)` for a parametric Gaussian
interval; we use `alpha=2*(1-Φ(1))` ≈ 0.3173 to get a ±1σ band, then
convert the half-width back to σ̂. (Equivalent to taking the parametric
`mse_forecasts` square-root, but works uniformly across statsmodels
versions.)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Forecast, Forecaster, ForecasterFaultError


# ±1σ confidence level: P(|Z| ≤ 1) ≈ 0.6827 → α ≈ 0.3173
ONE_SIGMA_ALPHA = 2 * (1 - 0.84134474606854293)


@dataclass(frozen=True)
class SARIMAOrder:
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    m: int

    def as_tuple(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        return (self.p, self.d, self.q), (self.P, self.D, self.Q, self.m)


# Bounded order grid covering the §3.6 use case (short-horizon CPU/RPS).
# All orders ≤ 2 keep fitting cheap (each candidate fits in O(n)).
def _default_orders(period_samples: int) -> List[SARIMAOrder]:
    candidates: List[SARIMAOrder] = []
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                for P in (0, 1):
                    for D in (0, 1):
                        for Q in (0, 1):
                            # Skip the degenerate all-zero order
                            if p == d == q == P == D == Q == 0:
                                continue
                            candidates.append(
                                SARIMAOrder(p=p, d=d, q=q, P=P, D=D, Q=Q, m=period_samples)
                            )
    return candidates


class SARIMA(Forecaster):
    """SARIMA with AICc stepwise order selection (§3.6)."""

    name = "sarima"

    def __init__(
        self,
        period_seconds: int = 60,
        sample_interval_seconds: int = 15,
        *,
        order_candidates: Optional[List[SARIMAOrder]] = None,
        min_history_samples: Optional[int] = None,
        refit_each_predict: bool = False,
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
        self.order_candidates = order_candidates or _default_orders(self.period_samples)
        # SARIMA with seasonal differencing needs at least 2 full periods.
        self.min_history_samples = min_history_samples or (3 * self.period_samples)
        self.refit_each_predict = refit_each_predict
        self._best_results = None
        self._best_order: Optional[SARIMAOrder] = None
        self._fitted_n = 0

    # ------------------------------------------------------------------ #
    # Forecaster interface                                                #
    # ------------------------------------------------------------------ #
    def fit(self, history: pd.Series) -> "SARIMA":
        clean = history.dropna().astype(float)
        if len(clean) < self.min_history_samples:
            raise ForecasterFaultError(
                f"SARIMA needs ≥ {self.min_history_samples} clean samples "
                f"(have {len(clean)}) for period {self.period_samples}"
            )

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as e:
            raise RuntimeError(
                "SARIMA requires statsmodels. Install via uv: `uv sync`."
            ) from e

        best_results = None
        best_order: Optional[SARIMAOrder] = None
        best_aicc = float("inf")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.order_candidates:
                try:
                    arima_order, seasonal_order = order.as_tuple()
                    model = SARIMAX(
                        clean.values,
                        order=arima_order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=False,
                    )
                    res = model.fit(disp=False, method="lbfgs", maxiter=50)
                except Exception:
                    continue
                aicc = _aicc_from_aic(res.aic, k=res.df_model + 1, n=len(clean))
                if math.isfinite(aicc) and aicc < best_aicc:
                    best_aicc = aicc
                    best_results = res
                    best_order = order

        if best_results is None:
            raise ForecasterFaultError("SARIMA: no candidate order converged")

        self._best_results = best_results
        self._best_order = best_order
        self._fitted_n = len(clean)
        return self

    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")

        if self.refit_each_predict or self._best_results is None:
            self.fit(history)

        steps_ahead = max(1, round(horizon_seconds / self.sample_interval_seconds))

        try:
            fc_obj = self._best_results.get_forecast(steps=steps_ahead)
            point = float(np.asarray(fc_obj.predicted_mean)[-1])
            conf = fc_obj.conf_int(alpha=ONE_SIGMA_ALPHA)
            # statsmodels returns ndarray of shape (steps_ahead, 2) [low, high]
            arr = np.asarray(conf)
            low, high = float(arr[-1, 0]), float(arr[-1, 1])
        except Exception as e:
            raise ForecasterFaultError(f"SARIMA forecast failed: {e}") from e

        sigma = max((high - low) / 2.0, 0.0)
        if not math.isfinite(point) or not math.isfinite(sigma):
            raise ForecasterFaultError(
                f"SARIMA produced non-finite output (point={point}, sigma={sigma})"
            )
        return Forecast(point=point, sigma=sigma)

    # ------------------------------------------------------------------ #
    # Parsimony tiebreaker                                                #
    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        if self._best_results is not None:
            return int(self._best_results.df_model) + 1   # + σ
        if self._best_order is None:
            return 0
        o = self._best_order
        return o.p + o.q + o.P + o.Q + 1


def _aicc_from_aic(aic: float, k: int, n: int) -> float:
    """AICc = AIC + 2k(k+1)/(n−k−1). Per Hyndman-Athanasopoulos."""
    if n - k - 1 <= 0:
        return float("inf")
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))
