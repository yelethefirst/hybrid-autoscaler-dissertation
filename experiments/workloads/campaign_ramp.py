"""Looping ramp workload for 5-hour training data campaigns (§3.5).

Ramps from FLOOR_USERS to PEAK_USERS over RAMP_UP_SECONDS, holds for
HOLD_SECONDS, ramps back down, then repeats. Locust duration controls
total run time.

Environment variables:
    FLOOR_USERS      default 50
    PEAK_USERS       default 300
    RAMP_UP_SECONDS  default 600
    HOLD_SECONDS     default 300
    RAMP_DOWN_SECONDS default 600
"""

from __future__ import annotations

import os

from locust import LoadTestShape

FLOOR_USERS = int(os.getenv("FLOOR_USERS", "50"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "300"))
RAMP_UP = int(os.getenv("RAMP_UP_SECONDS", "600"))
HOLD = int(os.getenv("HOLD_SECONDS", "300"))
RAMP_DOWN = int(os.getenv("RAMP_DOWN_SECONDS", "600"))
SPAWN_RATE = int(os.getenv("SPAWN_RATE", "10"))

_CYCLE = RAMP_UP + HOLD + RAMP_DOWN  # 1500 s default (25 min/cycle, ~12 cycles in 5h)


class CampaignRampShape(LoadTestShape):
    """Linear ramp up → hold → ramp down, repeating indefinitely."""

    def tick(self):
        t = self.get_run_time() % _CYCLE
        if t < RAMP_UP:
            frac = t / RAMP_UP
            users = int(FLOOR_USERS + (PEAK_USERS - FLOOR_USERS) * frac)
            return max(FLOOR_USERS, users), SPAWN_RATE
        if t < RAMP_UP + HOLD:
            return PEAK_USERS, SPAWN_RATE
        frac = (t - RAMP_UP - HOLD) / RAMP_DOWN
        users = int(PEAK_USERS - (PEAK_USERS - FLOOR_USERS) * frac)
        return max(FLOOR_USERS, users), SPAWN_RATE
