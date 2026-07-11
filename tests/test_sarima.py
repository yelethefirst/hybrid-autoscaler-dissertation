"""Tests for the SARIMA forecaster with AICc stepwise selection (§3.6)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from data.synthetic import generate, to_wide
from forecasting import SARIMA
from forecasting.base import ForecasterFaultError
from forecasting.sarima import SARIMAOrder, _aicc_from_aic, _default_orders


def _periodic_series(n_samples: int = 80):
    df_long = generate(workload="periodic", duration_seconds=n_samples * 15, seed=0)
    wide = to_wide(df_long, service="frontend")
    return wide.set_index("timestamp")["cpu"]


def test_default_order_grid_is_non_empty():
    orders = _default_orders(period_samples=4)
    assert len(orders) > 0
    # No degenerate all-zero order
    for o in orders:
        assert not (o.p == o.d == o.q == o.P == o.D == o.Q == 0)


def test_sarima_fit_and_predict_smoke():
    s = _periodic_series(n_samples=80)
    # Restrict the grid to a small subset so the test runs quickly.
    small_grid = [
        SARIMAOrder(p=1, d=0, q=0, P=0, D=0, Q=0, m=4),
        SARIMAOrder(p=0, d=0, q=1, P=0, D=0, Q=0, m=4),
        SARIMAOrder(p=1, d=0, q=1, P=0, D=0, Q=0, m=4),
    ]
    fc = SARIMA(period_seconds=60, sample_interval_seconds=15, order_candidates=small_grid)
    fc.fit(s)
    f = fc.predict(s, horizon_seconds=30)
    assert np.isfinite(f.point)
    assert f.sigma >= 0


def test_sarima_too_little_history_raises_fault():
    s = _periodic_series(n_samples=10)
    fc = SARIMA(period_seconds=60, sample_interval_seconds=15)
    with pytest.raises(ForecasterFaultError):
        fc.fit(s)


def test_aicc_formula_known_values():
    # AICc = AIC + 2k(k+1)/(n−k−1)
    # AIC=100, k=5, n=50 → AICc = 100 + 60/44 ≈ 101.36
    val = _aicc_from_aic(aic=100.0, k=5, n=50)
    assert val == pytest.approx(100 + (2 * 5 * 6) / (50 - 5 - 1), rel=1e-9)


def test_aicc_infinite_when_n_too_small():
    assert _aicc_from_aic(aic=10.0, k=5, n=5) == float("inf")
