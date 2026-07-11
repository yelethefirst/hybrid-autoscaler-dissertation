"""Forecasting models for the hybrid predictive autoscaler.

Implements the five forecaster families specified in §3.6:
    - Seasonal Naive       (`SeasonalNaive`)
    - Holt-Winters         (`HoltWinters`)
    - SARIMA               (`SARIMA`)
    - XGBoost              (`XGBoostForecaster`)
    - LSTM                 (`LSTMForecaster`)

Plus the supporting machinery from Phase 3:
    - Metrics (`rmse`, `mae`, `pinball_loss`, `mape`) in `metrics`
    - Prediction intervals in `intervals`
    - Seed discipline in `seeds`
    - Model selection harness in `selection`
    - H1 hypothesis test in `comparison`
"""

from .base import Forecast, Forecaster, ForecasterFaultError
from .holt_winters import HoltWinters
from .sarima import SARIMA
from .seasonal_naive import SeasonalNaive

# Heavy ML deps are imported lazily by the class itself when fit() is called.
from .lstm_model import LSTMForecaster
from .xgboost_model import XGBoostForecaster

__all__ = [
    # Base
    "Forecast", "Forecaster", "ForecasterFaultError",
    # The five families (§3.6)
    "SeasonalNaive", "HoltWinters", "SARIMA",
    "XGBoostForecaster", "LSTMForecaster",
]
