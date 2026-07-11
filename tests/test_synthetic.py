"""Tests for the synthetic telemetry generator."""

from __future__ import annotations

import pandas as pd
import pytest

from data import METRIC_FAMILIES
from data.synthetic import generate, to_wide


def test_generate_produces_all_five_metric_families():
    df = generate(workload="periodic", duration_seconds=300, seed=1)
    fams_seen = set(df["metric_family"].unique())
    assert fams_seen == {f.value for f in METRIC_FAMILIES}


def test_generate_is_deterministic_for_same_seed():
    df_a = generate(workload="burst", duration_seconds=300, seed=42)
    df_b = generate(workload="burst", duration_seconds=300, seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_generate_differs_for_different_seeds():
    df_a = generate(workload="burst", duration_seconds=300, seed=1)
    df_b = generate(workload="burst", duration_seconds=300, seed=2)
    assert not df_a.equals(df_b)


def test_sample_count_matches_duration():
    # 600 s @ 15 s = 40 samples per (service, metric_name)
    df = generate(workload="periodic", duration_seconds=600, sample_interval_seconds=15, seed=0)
    sample_count = (
        df[df["service"] == "frontend"]
          .groupby("metric_name")["timestamp"]
          .count()
    )
    # Each metric_name should have exactly 40 samples
    assert (sample_count == 40).all(), sample_count.to_dict()


def test_to_wide_produces_expected_columns():
    df = generate(workload="periodic", duration_seconds=300, seed=0)
    wide = to_wide(df, service="frontend")
    expected = {"timestamp", "cpu", "memory", "pod_ready", "request_rate",
                "request_rate_errors", "response_time_p95"}
    assert expected.issubset(set(wide.columns))


def test_to_wide_chronologically_sorted():
    df = generate(workload="ramp", duration_seconds=600, seed=0)
    wide = to_wide(df, service="frontend")
    assert wide["timestamp"].is_monotonic_increasing


def test_downstream_responds_to_upstream():
    """The Hu et al. coupling: downstream CPU should rise when upstream RPS rises."""
    df = generate(workload="burst", duration_seconds=720, seed=7)
    front = to_wide(df, service="frontend")
    cart = to_wide(df, service="cartservice")
    # Pick the burst window (high frontend RPS).
    high_idx = front["request_rate"].nlargest(5).index
    low_idx = front["request_rate"].nsmallest(5).index
    # Use the same timestamps to look up cartservice CPU.
    cart_high = cart.loc[high_idx, "cpu"].mean()
    cart_low = cart.loc[low_idx, "cpu"].mean()
    assert cart_high > cart_low


@pytest.mark.parametrize("workload", ["burst", "ramp", "periodic", "trace_like", "steady"])
def test_all_workload_kinds_run(workload):
    df = generate(workload=workload, duration_seconds=300, seed=0)
    assert len(df) > 0
    # Schema-version pinned in attrs
    assert df.attrs["schema_version"]


def test_periodic_oscillates():
    df = generate(workload="periodic", duration_seconds=600, seed=0)
    front = to_wide(df, service="frontend")
    # RPS should swing — std should be a meaningful fraction of mean.
    assert front["request_rate"].std() > 0.1 * front["request_rate"].mean()
