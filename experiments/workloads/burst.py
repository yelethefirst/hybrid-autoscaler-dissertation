"""Canonical burst workload — §3.9 specification.

Shape: low → peak → low
    Warm-up  : BASELINE_USERS for WARMUP_SECONDS  (default: 50 users, 5 min)
    Burst    : PEAK_USERS     for BURST_SECONDS   (default: 500 users, 2 min)
    Cool-down: BASELINE_USERS for COOLDOWN_SECONDS (default: 50 users, 5 min)
    Total    : 12 minutes

Override via environment variables:
    BASELINE_USERS  — floor user count (default: 50)
    PEAK_USERS      — burst peak (default: 500)
    WARMUP_SECONDS  — duration of warm-up phase (default: 300)
    BURST_SECONDS   — duration of burst phase (default: 120)
    COOLDOWN_SECONDS— duration of cool-down phase (default: 300)
    SPAWN_RATE      — users spawned/stopped per second (default: 50)

Usage:
    locust -f experiments/workloads/burst.py \\
        --host http://localhost:30080 \\
        --headless --users 500 --spawn-rate 50 -t 12m

Or via the Phase 5 supervisor which sets env vars automatically.
"""

from __future__ import annotations

import os

from locust import LoadTestShape

# NOTE: no `from .user import BoutiqueUser` here — Locust loads this file as a
# standalone module (no package context), so a relative import crashes at
# startup (DEV-009/DEV-017). The supervisor and run-campaign.sh pass user.py
# as a second -f argument; Locust discovers BoutiqueUser from there.

BASELINE_USERS = int(os.getenv("BASELINE_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "500"))
WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", "300"))
BURST_SECONDS = int(os.getenv("BURST_SECONDS", "120"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "300"))
SPAWN_RATE = int(os.getenv("SPAWN_RATE", "50"))

_T1 = WARMUP_SECONDS
_T2 = _T1 + BURST_SECONDS
_T3 = _T2 + COOLDOWN_SECONDS


class BurstShape(LoadTestShape):
    """Step function: BASELINE → PEAK → BASELINE.

    The step transition (rather than a ramp) is the distinguishing feature of
    the burst workload — it creates the maximum δ-load that tests a controller's
    ability to predict and pre-provision before the burst materialises.
    """

    def tick(self):
        t = self.get_run_time()
        if t < _T1:
            return BASELINE_USERS, SPAWN_RATE
        if t < _T2:
            return PEAK_USERS, SPAWN_RATE
        if t < _T3:
            return BASELINE_USERS, SPAWN_RATE
        return None
