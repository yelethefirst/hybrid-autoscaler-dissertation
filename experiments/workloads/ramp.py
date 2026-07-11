"""Canonical ramp workload — §3.9 specification.

Shape: linear ramp up → hold → linear ramp down
    Ramp-up  : FLOOR_USERS → PEAK_USERS over RAMP_SECONDS  (default: 50→500, 10 min)
    Hold     : PEAK_USERS  for HOLD_SECONDS                 (default: 500, 5 min)
    Ramp-down: PEAK_USERS → FLOOR_USERS over RAMP_SECONDS   (default: 500→50, 10 min)
    Total    : 25 minutes

Override via environment variables:
    FLOOR_USERS   — floor user count (default: 50)
    PEAK_USERS    — peak user count (default: 500)
    RAMP_SECONDS  — duration of ramp phases (default: 600)
    HOLD_SECONDS  — duration of hold phase (default: 300)
    SPAWN_RATE    — users spawned/stopped per second (default: 1; overridden by shape)

Usage:
    locust -f experiments/workloads/ramp.py \\
        --host http://localhost:30080 \\
        --headless --users 500 --spawn-rate 1 -t 25m
"""

from __future__ import annotations

import os

from locust import LoadTestShape

# NOTE: no `from .user import BoutiqueUser` here — Locust loads this file as a
# standalone module (no package context), so a relative import crashes at
# startup (DEV-009/DEV-017). The supervisor and run-campaign.sh pass user.py
# as a second -f argument; Locust discovers BoutiqueUser from there.

FLOOR_USERS = int(os.getenv("FLOOR_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "500"))
RAMP_SECONDS = int(os.getenv("RAMP_SECONDS", "600"))
HOLD_SECONDS = int(os.getenv("HOLD_SECONDS", "300"))

_T1 = RAMP_SECONDS
_T2 = _T1 + HOLD_SECONDS
_T3 = _T2 + RAMP_SECONDS
_RANGE = PEAK_USERS - FLOOR_USERS


class RampShape(LoadTestShape):
    """Triangular ramp: linear increase → plateau → linear decrease.

    The gradual ramp tests whether the predictive controller learns the trend
    and pre-provisions, whereas HPA (purely reactive) will lag behind the ramp.
    """

    def tick(self):
        t = self.get_run_time()
        if t < _T1:
            users = int(FLOOR_USERS + (_RANGE * t / _T1))
            rate = max(1, _RANGE // _T1)
            return users, rate
        if t < _T2:
            return PEAK_USERS, 50
        if t < _T3:
            elapsed = t - _T2
            users = int(PEAK_USERS - (_RANGE * elapsed / RAMP_SECONDS))
            rate = max(1, _RANGE // RAMP_SECONDS)
            return max(FLOOR_USERS, users), rate
        return None
