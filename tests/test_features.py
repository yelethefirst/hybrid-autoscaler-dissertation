"""Tests for the §3.5 feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.features import add_upstream_request_rate, engineer_features
from data.features.engineer import (
    feature_schema_for,
)
from data.synthetic import generate, to_wide


def _toy_wide(values):
    ts = pd.date_range("2026-05-01T00:00:00Z", periods=len(values), freq="15s", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "cpu": values})


def test_lag_columns_match_shift():
    wide = _toy_wide([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    out = engineer_features(wide, base_column="cpu", lag_intervals=[1, 3, 5])
    # lag k → row i should equal value at i-k
    for k in [1, 3, 5]:
        col = f"cpu_lag{k}"
        for i in range(k, len(wide)):
            assert out.loc[i, col] == wide.loc[i - k, "cpu"]
        # Opening rows are NaN
        for i in range(k):
            assert pd.isna(out.loc[i, col])


def test_rolling_mean_is_past_only():
    wide = _toy_wide([10.0] * 4 + [20.0] * 4 + [30.0] * 4)
    out = engineer_features(
        wide, base_column="cpu", rolling_windows_seconds=[30],  # n=2 at 15s cadence
        lag_intervals=[], diff_first=False, diff_first_log=False,
    )
    # At row 1: rmean over rows [0, 1] = 10
    assert out.loc[1, "cpu_rmean30s"] == pytest.approx(10.0)
    # At row 4: rmean over rows [3, 4] = (10 + 20) / 2 = 15
    assert out.loc[4, "cpu_rmean30s"] == pytest.approx(15.0)
    # Opening: only one sample of history → NaN with min_periods=2
    assert pd.isna(out.loc[0, "cpu_rmean30s"])


def test_rolling_variance_is_nonnegative_where_defined():
    wide = _toy_wide(list(range(20)))
    out = engineer_features(
        wide, base_column="cpu", rolling_windows_seconds=[60],
        lag_intervals=[], diff_first=False, diff_first_log=False,
    )
    var = out["cpu_rvar60s"].dropna()
    assert (var >= 0).all()


def test_diff_first():
    wide = _toy_wide([10.0, 12.0, 9.0, 15.0])
    out = engineer_features(
        wide, base_column="cpu", lag_intervals=[],
        rolling_windows_seconds=[], diff_first=True, diff_first_log=False,
    )
    assert pd.isna(out.loc[0, "cpu_diff1"])
    assert out.loc[1, "cpu_diff1"] == pytest.approx(2.0)
    assert out.loc[2, "cpu_diff1"] == pytest.approx(-3.0)
    assert out.loc[3, "cpu_diff1"] == pytest.approx(6.0)


def test_log_diff_handles_zero_safely():
    wide = _toy_wide([0.0, 0.0, 1.0, 2.0])
    out = engineer_features(
        wide, base_column="cpu", lag_intervals=[],
        rolling_windows_seconds=[], diff_first=False, diff_first_log=True,
    )
    # No infs / NaNs (other than the opening row which is structurally NaN).
    vals = out["cpu_logdiff1"].iloc[1:].to_numpy()
    assert np.all(np.isfinite(vals))


def test_default_features_match_schema():
    wide = _toy_wide(list(range(40)))
    out = engineer_features(wide, base_column="cpu")
    schema = feature_schema_for(service="frontend", base_column="cpu")
    expected = schema.expected_feature_names()
    # Every name in the schema must be a column in the engineered frame
    # (excluding the upstream features which require upstream services).
    for name in expected:
        if name.endswith("_request_rate_t"):
            continue
        assert name in out.columns, f"missing feature column: {name}"


def test_upstream_request_rate_joins_correctly():
    df_long = generate(workload="periodic", duration_seconds=300, seed=0)
    target = to_wide(df_long, service="cartservice")
    enriched = add_upstream_request_rate(target, df_long, upstream_services=["frontend"])
    assert "frontend_request_rate_t" in enriched.columns
    # Values should be non-null after the first sample
    assert enriched["frontend_request_rate_t"].notna().sum() > 0


def test_rolling_window_too_small_rejected():
    wide = _toy_wide(list(range(20)))
    with pytest.raises(ValueError, match="below sample interval"):
        engineer_features(wide, base_column="cpu", rolling_windows_seconds=[5])  # < 15s sample


def test_negative_lag_rejected():
    wide = _toy_wide(list(range(20)))
    with pytest.raises(ValueError, match="lag interval"):
        engineer_features(wide, base_column="cpu", lag_intervals=[-1])
