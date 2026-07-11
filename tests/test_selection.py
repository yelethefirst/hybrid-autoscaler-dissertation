"""Tests for the model-selection harness (§3.6 three-criterion rule)."""

from __future__ import annotations

import time

import pandas as pd

from data.synthetic import generate, to_wide
from forecasting import Forecast, Forecaster, SeasonalNaive
from forecasting.base import ForecasterFaultError
from forecasting.selection import evaluate_forecasters


# ────────────────────────────────────────────────────────────────────────
# Mock forecasters so the selection harness can be tested without
# depending on the full §3.6 grid each time.
# ────────────────────────────────────────────────────────────────────────
class _ConstantForecaster(Forecaster):
    name = "constant"
    def __init__(self, value: float = 0.0, sigma: float = 0.01,
                 params: int = 1, delay_s: float = 0.0):
        self.value = value
        self.sigma = sigma
        self._params = params
        self._delay = delay_s
    def fit(self, history): return self
    def predict(self, history, horizon_seconds):
        if self._delay > 0:
            time.sleep(self._delay)
        return Forecast(point=self.value, sigma=self.sigma)
    def n_parameters(self): return self._params


class _PersistForecaster(Forecaster):
    name = "persist"
    def fit(self, history): return self
    def predict(self, history, horizon_seconds):
        clean = history.dropna()
        if len(clean) == 0:
            raise ForecasterFaultError("empty history")
        return Forecast(point=float(clean.iloc[-1]), sigma=0.05)
    def n_parameters(self): return 0


def _series(n: int = 80) -> pd.Series:
    df_long = generate(workload="periodic", duration_seconds=n * 15, seed=0)
    return to_wide(df_long, service="frontend").set_index("timestamp")["cpu"]


# ────────────────────────────────────────────────────────────────────────
# Selection-rule tests
# ────────────────────────────────────────────────────────────────────────
def test_selection_runs_walk_forward_and_returns_report():
    s = _series(n=100)
    report = evaluate_forecasters(
        s,
        factories={
            "seasonal_naive": lambda: SeasonalNaive(period_seconds=60),
            "persist": lambda: _PersistForecaster(),
        },
        service="frontend",
        horizon_seconds=30,
        n_splits=3,
        inference_latency_trials=3,
    )
    assert report.service == "frontend"
    assert set(report.scores.keys()) == {"seasonal_naive", "persist"}
    assert report.selected_name in ("seasonal_naive", "persist")


def test_lowest_rmse_wins_when_clearly_better():
    s = pd.Series([1.0] * 80, index=pd.date_range(
        "2026-05-01", periods=80, freq="15s", tz="UTC"
    ))
    # Constant=1 is perfect for a constant series; Constant=10 is awful.
    report = evaluate_forecasters(
        s,
        factories={
            "perfect": lambda: _ConstantForecaster(value=1.0, params=1),
            "awful":   lambda: _ConstantForecaster(value=10.0, params=1),
        },
        n_splits=3,
        inference_latency_trials=3,
    )
    assert report.selected_name == "perfect"
    assert "RMSE" in report.selection_reason


def test_latency_breaks_rmse_tie():
    s = pd.Series([2.0] * 80, index=pd.date_range(
        "2026-05-01", periods=80, freq="15s", tz="UTC"
    ))
    report = evaluate_forecasters(
        s,
        factories={
            "fast": lambda: _ConstantForecaster(value=2.0, params=100, delay_s=0.0),
            "slow": lambda: _ConstantForecaster(value=2.0, params=1,   delay_s=0.05),
        },
        n_splits=3,
        inference_latency_trials=5,
    )
    assert report.selected_name == "fast"
    assert "latency" in report.selection_reason


def test_parsimony_breaks_remaining_tie():
    s = pd.Series([2.0] * 80, index=pd.date_range(
        "2026-05-01", periods=80, freq="15s", tz="UTC"
    ))
    report = evaluate_forecasters(
        s,
        factories={
            "big":   lambda: _ConstantForecaster(value=2.0, params=10_000, delay_s=0),
            "small": lambda: _ConstantForecaster(value=2.0, params=1, delay_s=0),
        },
        n_splits=3,
        inference_latency_trials=3,
    )
    assert report.selected_name == "small"
    assert "parsimony" in report.selection_reason


def test_failed_forecaster_recorded_but_doesnt_crash():
    class _AlwaysFails(Forecaster):
        name = "broken"
        def fit(self, history):
            raise ForecasterFaultError("nope")
        def predict(self, history, horizon_seconds):
            raise ForecasterFaultError("nope")

    s = _series(n=80)
    report = evaluate_forecasters(
        s,
        factories={
            "ok":     lambda: _PersistForecaster(),
            "broken": lambda: _AlwaysFails(),
        },
        n_splits=3,
        inference_latency_trials=3,
    )
    assert report.scores["broken"].failed_folds > 0
    assert report.selected_name == "ok"


def test_report_as_table_columns():
    s = _series(n=80)
    report = evaluate_forecasters(
        s,
        factories={"persist": lambda: _PersistForecaster()},
        n_splits=3, inference_latency_trials=3,
    )
    table = report.as_table()
    expected = {"name", "mean_rmse", "mean_mae", "inference_p95_seconds",
                "n_parameters", "failed_folds"}
    assert expected.issubset(set(table.columns))
