"""Abstract base class and shared types for forecasters.

§3.6 contract: every forecaster, regardless of family (Seasonal Naive,
Holt-Winters, SARIMA, XGBoost, LSTM), exposes the same `predict(history,
horizon_seconds) -> Forecast` surface. The decision engine in §3.7 consumes
the `Forecast.point` and `Forecast.sigma` to compute its confidence-aware
recommendation, and treats a raised `ForecasterFaultError` as the trigger
for the FALLBACK_FORECASTER_FAULT state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd


class Forecast(NamedTuple):
    """A single point forecast with a one-standard-deviation interval.

    Attributes
    ----------
    point : float
        Predicted value at the requested horizon, in the same units as the
        input history (§3.6 "point estimate").
    sigma : float
        One-standard-deviation half-width of the prediction interval, in the
        same units as `point` (§3.6 "one-standard-deviation prediction
        interval"). Must be ≥ 0. The decision engine uses this directly as
        σ̂(t+h) in the §3.7 formula.
    """

    point: float
    sigma: float


class ForecasterFaultError(RuntimeError):
    """Raised by a forecaster when it cannot produce a valid prediction.

    Examples include insufficient history, NaN/Inf in the prediction, model
    state corruption, or any other condition that would make the forecast
    unsafe to feed into the decision engine. The engine catches this and
    transitions to FALLBACK_FORECASTER_FAULT (§3.7).
    """


class Forecaster(ABC):
    """Abstract base class for all forecasters.

    Subclasses must:
      - set a unique `name` attribute (used in evidence-bundle logs);
      - implement `predict(history, horizon_seconds) -> Forecast`.

    They may also override `fit(history)` if they have an internal state
    that should be learned ahead of prediction (Holt-Winters, SARIMA,
    XGBoost, LSTM). The default `fit` is a no-op, which is correct for
    Seasonal Naive.

    `history` is a `pd.Series` indexed by timezone-aware UTC timestamps,
    sampled at a regular cadence (15 s by default per §3.5). `horizon_seconds`
    is the forecast horizon, typically 30 or 60 seconds per §3.6.

    Implementations should also expose:
      - `n_parameters() -> int` so the §3.6 parsimony tiebreaker can rank them.
    """

    name: str = "abstract"

    def fit(self, history: pd.Series) -> "Forecaster":
        """Optional training step. Default is a no-op. Returns self."""
        return self

    @abstractmethod
    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        """Return a point forecast and 1-σ interval at the given horizon.

        Implementations may raise `ForecasterFaultError` to signal that the
        prediction cannot be trusted (the decision engine will fall back).
        """
        raise NotImplementedError

    def n_parameters(self) -> int:
        """Return the model's parameter count. Defaults to 0 (Seasonal Naive)."""
        return 0

    def shap_attribution(
        self, history: pd.Series, horizon_seconds: int, top_k: int = 10
    ) -> dict:
        """Return a SHAP-style attribution dict for the most recent prediction.

        Returns a dict with at least:
            method      : str — "shap_tree", "hw_components", or "none"
            top_features: dict[feature_name, attribution_value]

        Default returns an empty dict (no attribution available). Subclasses
        that support attribution should override this method. Called by the
        control loop after a successful predict() so attribution failures
        never affect the scaling decision.
        """
        return {}
