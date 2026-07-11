"""Tests for the empirical anti-leakage validator (§3.5)."""

from __future__ import annotations

import pandas as pd
import pytest

from data.features import engineer_features
from data.leakage_check import assert_no_leakage, check_no_leakage
from data.synthetic import generate, to_wide


def _wide():
    df_long = generate(workload="periodic", duration_seconds=900, seed=0)
    return to_wide(df_long, service="frontend")


# ────────────────────────────────────────────────────────────────────────
# Causal pipelines should PASS
# ────────────────────────────────────────────────────────────────────────
def test_default_engineer_passes_leakage_check():
    wide = _wide()
    report = check_no_leakage(
        pipeline=lambda d: engineer_features(d, base_column="cpu"),
        wide=wide,
        base_columns=["cpu"],
        split_fraction=0.5,
        seed=0,
    )
    assert report.passed, report.summary()


def test_pipeline_with_only_lags_passes():
    wide = _wide()
    report = check_no_leakage(
        pipeline=lambda d: engineer_features(
            d, base_column="cpu",
            lag_intervals=[1, 2, 3], rolling_windows_seconds=[],
            diff_first=False, diff_first_log=False,
        ),
        wide=wide, base_columns=["cpu"], seed=0,
    )
    assert report.passed


# ────────────────────────────────────────────────────────────────────────
# Deliberately-leaky pipelines should FAIL
# ────────────────────────────────────────────────────────────────────────
def test_negative_shift_is_caught():
    """shift(-1) looks one step into the future — must be detected."""
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        out["cpu_lag_minus_1"] = out["cpu"].shift(-1)
        return out

    wide = _wide()
    report = check_no_leakage(leaky, wide, base_columns=["cpu"], seed=0)
    assert not report.passed
    assert "cpu_lag_minus_1" in report.leaked_columns


def test_centred_rolling_mean_is_caught():
    """A centred rolling window includes future samples in the calculation."""
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        # Centred window: at index i it uses [i-w//2, i+w//2]
        out["cpu_rmean_centred"] = out["cpu"].rolling(window=5, center=True).mean()
        return out

    wide = _wide()
    report = check_no_leakage(leaky, wide, base_columns=["cpu"], seed=0)
    assert not report.passed
    assert "cpu_rmean_centred" in report.leaked_columns


def test_future_diff_is_caught():
    """diff(-1) = x(t) − x(t+1): looks ahead."""
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        out["cpu_future_diff"] = out["cpu"].diff(-1)
        return out

    wide = _wide()
    report = check_no_leakage(leaky, wide, base_columns=["cpu"], seed=0)
    assert not report.passed


def test_assert_no_leakage_raises_on_leak():
    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        out["future"] = out["cpu"].shift(-2)
        return out

    wide = _wide()
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(leaky, wide, base_columns=["cpu"])


def test_assert_no_leakage_silent_when_clean():
    wide = _wide()
    # Should not raise.
    assert_no_leakage(
        lambda d: engineer_features(d, base_column="cpu"),
        wide,
        base_columns=["cpu"],
    )


def test_report_split_index_reported():
    wide = _wide()
    report = check_no_leakage(
        lambda d: engineer_features(d, base_column="cpu"),
        wide, base_columns=["cpu"], split_fraction=0.5,
    )
    assert report.n_rows_checked == report.split_index
    assert 0 < report.split_index < len(wide)
