"""Canonical periodic workload — §3.9 specification.

Shape: sinusoidal oscillation between FLOOR_USERS and PEAK_USERS.
    Period   : PERIOD_SECONDS          (default: 600 s = 10 min)
    Duration : NUM_PERIODS × PERIOD_S  (default: 3 × 10 min = 30 min)
    Floor    : FLOOR_USERS             (default: 50)
    Peak     : PEAK_USERS              (default: 500)

The sinusoidal shape is: users(t) = mid + amplitude × sin(2π·t / period)
where mid = (PEAK + FLOOR) / 2, amplitude = (PEAK - FLOOR) / 2.

Override via environment variables:
    FLOOR_USERS   — minimum user count (default: 50)
    PEAK_USERS    — maximum user count (default: 500)
    PERIOD_SECONDS— length of one full cycle (default: 600)
    NUM_PERIODS   — number of cycles to run (default: 3)

Usage:
    locust -f experiments/workloads/periodic.py \\
        --host http://localhost:30080 \\
        --headless --users 500 --spawn-rate 50 -t 30m
"""

from __future__ import annotations

import math
import os

from locust import LoadTestShape

# NOTE: no `from .user import BoutiqueUser` here — Locust loads this file as a
# standalone module (no package context), so a relative import crashes at
# startup (DEV-009/DEV-017). The supervisor and run-campaign.sh pass user.py
# as a second -f argument; Locust discovers BoutiqueUser from there.

FLOOR_USERS = int(os.getenv("FLOOR_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "500"))
PERIOD_SECONDS = int(os.getenv("PERIOD_SECONDS", "600"))
NUM_PERIODS = int(os.getenv("NUM_PERIODS", "3"))

_MID = (PEAK_USERS + FLOOR_USERS) / 2
_AMP = (PEAK_USERS - FLOOR_USERS) / 2
_TOTAL = PERIOD_SECONDS * NUM_PERIODS


class PeriodicShape(LoadTestShape):
    """Sinusoidal load that repeats NUM_PERIODS times.

    Tests whether the predictive controller learns the seasonality and
    pre-provisions before each peak, compared to HPA which always lags.
    """

    def tick(self):
        t = self.get_run_time()
        if t >= _TOTAL:
            return None
        users = int(_MID + _AMP * math.sin(2 * math.pi * t / PERIOD_SECONDS))
        users = max(FLOOR_USERS, min(PEAK_USERS, users))
        return users, 50
