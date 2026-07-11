"""Tests for the XGBoost forecaster (§3.6)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from data.synthetic import generate, to_wide
from forecasting import XGBoostForecaster
from forecasting.base import ForecasterFaultError


def _periodic_series(n_samples: int = 200):
    df_long = generate(workload="periodic", duration_seconds=n_samples * 15, seed=0)
    wide = to_wide(df_long, service="frontend")
    return wide.set_index("timestamp")["cpu"]


def test_xgboost_fit_predict_smoke():
    s = _periodic_series(n_samples=300)
    # Trim the grid hard so the test runs in a second.
    fc = XGBoostForecaster(
        horizon_seconds=30,
        sample_interval_seconds=15,
        n_estimators_grid=[50],
        max_depth_grid=[3],
        learning_rate_grid=[0.1],
    )
    fc.fit(s)
    f = fc.predict(s, horizon_seconds=30)
    assert np.isfinite(f.point)
    assert f.sigma >= 0


def test_xgboost_predict_before_fit_raises():
    fc = XGBoostForecaster(
        n_estimators_grid=[50], max_depth_grid=[3], learning_rate_grid=[0.1],
    )
    s = _periodic_series(n_samples=300)
    with pytest.raises(ForecasterFaultError, match="before fit"):
        fc.predict(s, horizon_seconds=30)


def test_xgboost_horizon_mismatch_rejected():
    fc = XGBoostForecaster(
        horizon_seconds=30, n_estimators_grid=[50],
        max_depth_grid=[3], learning_rate_grid=[0.1],
    )
    s = _periodic_series(n_samples=300)
    fc.fit(s)
    with pytest.raises(ValueError, match="horizon"):
        fc.predict(s, horizon_seconds=60)


def test_xgboost_n_parameters_positive_after_fit():
    fc = XGBoostForecaster(
        n_estimators_grid=[50], max_depth_grid=[3], learning_rate_grid=[0.1],
    )
    s = _periodic_series(n_samples=300)
    fc.fit(s)
    assert fc.n_parameters() > 0


def test_xgboost_artifact_round_trip(tmp_path):
    fc = XGBoostForecaster(
        horizon_seconds=30,
        sample_interval_seconds=15,
        n_estimators_grid=[50],
        max_depth_grid=[3],
        learning_rate_grid=[0.1],
    )
    s = _periodic_series(n_samples=300)
    fc.fit(s)
    before = fc.predict(s, horizon_seconds=30)

    artifact_dir = fc.save_artifact(tmp_path / "frontend-xgboost")
    assert (artifact_dir / "model.json").exists()
    assert (artifact_dir / "metadata.json").exists()

    code = f"""
from data.synthetic import generate, to_wide
from forecasting import XGBoostForecaster

df_long = generate(workload="periodic", duration_seconds={300 * 15}, seed=0)
series = to_wide(df_long, service="frontend").set_index("timestamp")["cpu"]
loaded = XGBoostForecaster.load_artifact({str(artifact_dir)!r})
after = loaded.predict(series, horizon_seconds=30)
assert abs(after.point - {before.point!r}) < 1e-9
assert abs(after.sigma - {before.sigma!r}) < 1e-9
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
