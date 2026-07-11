"""Tests for the Holt-Winters forecaster (§3.6).

Skip cleanly when statsmodels is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from forecasting import HoltWinters
from forecasting.base import ForecasterFaultError
from data.synthetic import generate, to_wide


def _periodic_series(n_samples: int = 60):
    df_long = generate(workload="periodic", duration_seconds=n_samples * 15, seed=0)
    wide = to_wide(df_long, service="frontend")
    return wide.set_index("timestamp")["cpu"]


def test_holt_winters_fit_and_predict():
    s = _periodic_series(n_samples=80)
    fc = HoltWinters(period_seconds=60, sample_interval_seconds=15)
    fc.fit(s)
    f = fc.predict(s, horizon_seconds=30)
    assert np.isfinite(f.point)
    assert f.sigma >= 0


def test_holt_winters_short_history_raises_fault():
    s = _periodic_series(n_samples=8)   # below 2-period minimum
    fc = HoltWinters(period_seconds=60, sample_interval_seconds=15)
    with pytest.raises(ForecasterFaultError):
        fc.fit(s)


def test_holt_winters_period_validation():
    with pytest.raises(ValueError, match="multiple"):
        HoltWinters(period_seconds=50, sample_interval_seconds=15)


def test_holt_winters_parameter_count_reasonable():
    fc = HoltWinters(period_seconds=60, sample_interval_seconds=15)
    n = fc.n_parameters()
    # 6 fixed (α,β,γ,φ,level,trend) + period_samples seasonals = 10
    assert n == 10
