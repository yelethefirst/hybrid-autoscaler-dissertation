"""Trace-replay workload — §3.9 specification (Alibaba cluster trace).

Replays a pre-processed Alibaba cluster trace as normalised user counts.
The trace file must be a two-column CSV with no header:
    timestamp_seconds, rps

The replay scales the RPS column linearly so that:
    min(rps) → FLOOR_USERS
    max(rps) → PEAK_USERS

The trace is replayed in wall-clock time: the supervisor must provide a
trace file whose total duration equals the desired trial duration.

Trace preprocessing (not in this file) is in experiments/traces/:
    bin/preprocess-alibaba-trace.sh  — downloads and preprocesses the CSV.

Override via environment variables:
    TRACE_FILE    — path to the preprocessed trace CSV (required)
    FLOOR_USERS   — min user count (default: 50)
    PEAK_USERS    — max user count (default: 500)

Usage:
    TRACE_FILE=experiments/traces/alibaba_rps_30m.csv \\
    locust -f experiments/workloads/trace_replay.py \\
        --host http://localhost:30080 \\
        --headless --users 500 --spawn-rate 50 -t 30m
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from locust import LoadTestShape

# NOTE: no `from .user import BoutiqueUser` here — Locust loads this file as a
# standalone module (no package context), so a relative import crashes at
# startup (DEV-009/DEV-017). The supervisor and run-campaign.sh pass user.py
# as a second -f argument; Locust discovers BoutiqueUser from there.

TRACE_FILE = os.getenv("TRACE_FILE", "")
FLOOR_USERS = int(os.getenv("FLOOR_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "500"))


def _load_trace(path: str) -> list[tuple[float, float]]:
    """Return list of (timestamp_s, rps) sorted by timestamp."""
    rows: list[tuple[float, float]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def _normalise(trace: list[tuple[float, float]]) -> list[tuple[float, int]]:
    """Scale rps to [FLOOR_USERS, PEAK_USERS] and return (timestamp_s, users)."""
    if not trace:
        return []
    rps_vals = [r for _, r in trace]
    lo, hi = min(rps_vals), max(rps_vals)
    span_rps = hi - lo if hi > lo else 1.0
    span_usr = PEAK_USERS - FLOOR_USERS

    result: list[tuple[float, int]] = []
    for ts, rps in trace:
        users = int(FLOOR_USERS + span_usr * (rps - lo) / span_rps)
        result.append((ts, users))
    return result


# Load trace at module import time so Locust has it before tick() is called.
_TRACE: list[tuple[float, int]] = []
if TRACE_FILE and Path(TRACE_FILE).is_file():
    _TRACE = _normalise(_load_trace(TRACE_FILE))


class TraceReplayShape(LoadTestShape):
    """Replay a normalised trace as Locust user counts.

    Ticks every second; binary-searches the trace for the current timestamp
    to find the target user count. If the trace file is absent (e.g., during
    unit tests), the shape returns None immediately and the test stops.
    """

    def tick(self):
        if not _TRACE:
            return None
        t = self.get_run_time()
        if t > _TRACE[-1][0]:
            return None
        # Binary search for the last trace point ≤ t.
        lo, hi = 0, len(_TRACE) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _TRACE[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        users = _TRACE[lo][1]
        return users, 50
