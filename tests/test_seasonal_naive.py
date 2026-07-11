"""Unit tests for the Seasonal Naive forecaster (§3.6 baseline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import ForecasterFaultError, SeasonalNaive


def _series(values, start="2026-05-26T00:00:00Z", freq_seconds=15):
    idx = pd.date_range(start=start, periods=len(values), freq=f"{freq_seconds}s", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_predict_returns_value_one_period_ago_for_h_equal_period():
    # Period 60s, sample 15s → 4 samples/period. Two periods of history.
    history = _series([1, 2, 3, 4, 5, 6, 7, 8], freq_seconds=15)
    f = SeasonalNaive(period_seconds=60, sample_interval_seconds=15).predict(
        history, horizon_seconds=60
    )
    # At h=period, prediction = x[-period_samples + period_samples] = last value
    assert f.point == 8.0


def test_predict_at_short_horizon_looks_back_correctly():
    # h=30s with period=60s, sample=15s → horizon_samples=2, period_samples=4.
    # Hyndman-Athanasopoulos:
    #   ŷ(T+h) = y(T + h − m·⌈h/m⌉) = y(T + 2 − 4) = y(T − 2)
    # i.e. clean.iloc[-(m − h + 1)] = clean.iloc[-3] = value at index 5 = 60.
    history = _series([10, 20, 30, 40, 50, 60, 70, 80], freq_seconds=15)
    f = SeasonalNaive(period_seconds=60, sample_interval_seconds=15).predict(
        history, horizon_seconds=30
    )
    assert f.point == 60.0


def test_sigma_is_residual_std():
    # Deterministic residuals: x[t] - x[t-period].
    values = [1, 2, 3, 4,    2, 4, 6, 8]   # residuals = [1, 2, 3, 4] → std ≈ 1.291
    history = _series(values, freq_seconds=15)
    f = SeasonalNaive(period_seconds=60, sample_interval_seconds=15).predict(
        history, horizon_seconds=15
    )
    expected_sigma = float(np.std([1, 2, 3, 4], ddof=1))
    assert f.sigma == pytest.approx(expected_sigma, rel=1e-9)
    assert f.sigma > 0


def test_empty_history_raises_fault():
    fc = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    with pytest.raises(ForecasterFaultError):
        fc.predict(_series([]), horizon_seconds=30)


def test_too_little_history_raises_fault():
    # Default min_history_samples = 2 * period_samples = 8; pass only 2.
    fc = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    with pytest.raises(ForecasterFaultError):
        fc.predict(_series([1.0, 2.0]), horizon_seconds=30)


def test_horizon_above_period_rejected():
    fc = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    history = _series(list(range(16)), freq_seconds=15)
    with pytest.raises(ValueError, match="horizon_seconds"):
        fc.predict(history, horizon_seconds=120)


def test_nan_history_handled():
    fc = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    values = [1, 2, 3, 4, np.nan, 6, 7, 8]
    # NaN dropped — residual calculation still works on the remaining samples.
    f = fc.predict(_series(values, freq_seconds=15), horizon_seconds=30)
    assert np.isfinite(f.point)
    assert np.isfinite(f.sigma)


def test_constant_series_yields_zero_sigma():
    fc = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    f = fc.predict(_series([5.0] * 8, freq_seconds=15), horizon_seconds=30)
    assert f.point == 5.0
    assert f.sigma == pytest.approx(0.0)


def test_period_must_be_multiple_of_sample_interval():
    with pytest.raises(ValueError, match="multiple"):
        SeasonalNaive(period_seconds=50, sample_interval_seconds=15)
