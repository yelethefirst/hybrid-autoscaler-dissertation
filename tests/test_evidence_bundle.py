"""Round-trip tests for the JSONL evidence-bundle writer (§3.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from controller.evidence_bundle import EvidenceBundleWriter
from controller.state import Decision, EngineState


def _decision(**overrides) -> Decision:
    defaults = dict(
        service="frontend",
        namespace="default",
        horizon_seconds=30,
        forecast_point=0.42,
        forecast_sigma=0.05,
        forecaster_name="seasonal_naive",
        current_replicas=2,
        observed_metric=0.30,
        recommended_replicas=3,
        new_replicas=3,
        state=EngineState.NOMINAL,
        rate_limited=False,
        fallback_engaged=False,
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_writer_creates_parent_dir(tmp_path: Path):
    path = tmp_path / "sub" / "evidence.jsonl"
    EvidenceBundleWriter(path)
    assert path.parent.exists()


def test_round_trip_single_decision(tmp_path: Path):
    path = tmp_path / "evidence.jsonl"
    w = EvidenceBundleWriter(path)
    w.write(_decision())
    rows = w.read_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["service"] == "frontend"
    assert row["state"] == "NOMINAL"
    assert row["new_replicas"] == 3
    # timestamp is ISO-formatted
    assert "T" in row["timestamp"]


def test_round_trip_with_history_summary(tmp_path: Path):
    path = tmp_path / "evidence.jsonl"
    w = EvidenceBundleWriter(path)
    idx = pd.date_range("2026-05-26T00:00:00Z", periods=4, freq="15s", tz="UTC")
    history = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    w.write(_decision(), history=history)
    row = w.read_all()[0]
    s = row["feature_window_summary"]
    assert s["n"] == 4
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["mean"] == pytest.approx(2.5)
    assert s["last"] == 4.0


def test_appends_multiple_lines(tmp_path: Path):
    path = tmp_path / "evidence.jsonl"
    w = EvidenceBundleWriter(path)
    for i in range(5):
        w.write(_decision(current_replicas=i, new_replicas=i + 1))
    rows = w.read_all()
    assert len(rows) == 5
    assert [r["current_replicas"] for r in rows] == [0, 1, 2, 3, 4]


def test_lines_are_valid_json_individually(tmp_path: Path):
    path = tmp_path / "evidence.jsonl"
    w = EvidenceBundleWriter(path)
    w.write(_decision())
    w.write(_decision(state=EngineState.SCALE_LIMITED, rate_limited=True))
    for line in path.read_text().splitlines():
        json.loads(line)
