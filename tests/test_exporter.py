"""Tests for the Prometheus → Parquet exporter, using a fake PromClient."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.collect import PrometheusExporter, build_queries, run_campaign
from data.collect.exporter import CampaignConfig
from data.schema import MetricFamily


# ────────────────────────────────────────────────────────────────────────
# Fakes
# ────────────────────────────────────────────────────────────────────────
class FakePromClient:
    """Returns fixed synthetic Prometheus range-query results."""

    def __init__(self, ts_start: datetime, n_samples: int = 10, step: int = 15):
        self._t0 = ts_start
        self._n = n_samples
        self._step = step

    def custom_query_range(self, query: str, start_time, end_time, step) -> list:
        # Return one series with n_samples points; values differ per query so
        # we can verify each query lands in the right row family.
        return [{
            "metric": {"pod": "test-pod-0", "namespace": "default"},
            "values": [
                [
                    (self._t0 + timedelta(seconds=i * self._step)).timestamp(),
                    str(0.1 + 0.01 * i),     # arbitrary value sequence
                ]
                for i in range(self._n)
            ],
        }]


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────
def test_build_queries_returns_all_five_families():
    qs = build_queries("frontend", "default")
    fams = {q[0] for q in qs}
    assert fams == set(MetricFamily)


def test_build_queries_includes_service_label():
    qs = build_queries("checkoutservice")
    for _, _, q in qs:
        assert 'pod=~"checkoutservice-.*"' in q


def test_fetch_returns_long_format():
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    client = FakePromClient(ts_start=t0, n_samples=4)
    exp = PrometheusExporter(client)
    df = exp.fetch("frontend", "default", t0, t0 + timedelta(seconds=60), step_seconds=15)
    assert {"timestamp", "service", "namespace", "metric_family",
            "metric_name", "value", "labels"} <= set(df.columns)
    assert (df["service"] == "frontend").all()
    # Every family should appear at least once.
    assert set(df["metric_family"].unique()) == {f.value for f in MetricFamily}


def test_fetch_value_types():
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    df = PrometheusExporter(FakePromClient(t0, n_samples=4)).fetch(
        "frontend", "default", t0, t0 + timedelta(seconds=60), step_seconds=15
    )
    assert pd.api.types.is_float_dtype(df["value"])
    assert all(isinstance(t, (pd.Timestamp, datetime)) for t in df["timestamp"])


def test_run_campaign_writes_parquet_batches(tmp_path: Path):
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = t0 + timedelta(minutes=12)        # → 3 batches of 5 min, last partial
    cfg = CampaignConfig(
        services=["frontend", "cartservice"],
        namespace="default",
        sample_interval_seconds=15,
        batch_seconds=300,
        output_dir=tmp_path,
    )
    written = run_campaign(
        exporter=PrometheusExporter(FakePromClient(t0, n_samples=20)),
        config=cfg,
        start=t0,
        end=end,
    )
    # Expect 3 files (5+5+2 minutes)
    assert len(written) == 3
    for p in written:
        df = pd.read_parquet(p)
        assert not df.empty
        assert set(df["service"].unique()) == {"frontend", "cartservice"}


def test_campaign_handles_empty_batch_gracefully(tmp_path: Path):
    """If a query returns no data, the campaign should skip that batch silently."""
    class EmptyClient:
        def custom_query_range(self, **kwargs):
            return []

    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    cfg = CampaignConfig(
        services=["frontend"], output_dir=tmp_path, batch_seconds=300,
    )
    written = run_campaign(
        exporter=PrometheusExporter(EmptyClient()),
        config=cfg,
        start=t0,
        end=t0 + timedelta(minutes=5),
    )
    assert written == []
