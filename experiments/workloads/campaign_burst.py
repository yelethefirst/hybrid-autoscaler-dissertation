"""Looping burst workload for 5-hour training data campaigns (§3.5).

Identical shape to burst.py but repeats indefinitely using modulo time so
Locust does not stop after the first cycle. The campaign runner controls total
duration via --run-time.

Environment variables: same as burst.py.
"""

from __future__ import annotations

import os

from locust import LoadTestShape

BASELINE_USERS = int(os.getenv("BASELINE_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "300"))
WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", "300"))
BURST_SECONDS = int(os.getenv("BURST_SECONDS", "120"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "300"))
SPAWN_RATE = int(os.getenv("SPAWN_RATE", "30"))

_CYCLE = WARMUP_SECONDS + BURST_SECONDS + COOLDOWN_SECONDS  # 720 s default


class CampaignBurstShape(LoadTestShape):
    """Step-function burst that repeats every _CYCLE seconds indefinitely.

    Each cycle: BASELINE → PEAK → BASELINE.
    25 cycles in 5 hours gives the forecasters rich repeating-burst structure.
    """

    def tick(self):
        t = self.get_run_time() % _CYCLE
        if t < WARMUP_SECONDS:
            return BASELINE_USERS, SPAWN_RATE
        if t < WARMUP_SECONDS + BURST_SECONDS:
            return PEAK_USERS, SPAWN_RATE
        return BASELINE_USERS, SPAWN_RATE
