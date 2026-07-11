"""Minimal burst workload — Phase 1 vertical-slice demo.

This is NOT the §3.9 burst spec used for measured A/B trials. The full
§3.9 burst (50→500→50 users, 2 min each, with wrk2 for measurement and
coordinated-omission correction) is implemented in Phase 5
(experiments/workloads/burst.py). This file exists only to drive enough
load against the frontend during Phase 1 that we can watch the loop scale
up, watch CONFIDENCE_MARGIN_HIGH or SCALE_LIMITED appear in the evidence
bundle, and watch it scale back down.

Defaults are sized for a laptop kind cluster:
    BASELINE_USERS  =  10 (idle floor)
    PEAK_USERS      = 100 (burst)
    LOW_DURATION    =  60 seconds
    PEAK_DURATION   =  60 seconds

Override via environment variables, e.g.:
    BASELINE_USERS=50 PEAK_USERS=500 locust -f experiments/workloads/burst_minimal.py \
        --host http://localhost:30080 --headless -u 500 -r 50 -t 6m

Usage example (single-shot burst from low → peak → low, 3 minutes total):
    locust -f experiments/workloads/burst_minimal.py \
        --host http://localhost:30080 \
        --headless \
        -u 100 -r 100 \
        -t 3m
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, LoadTestShape, between, task

# ─── Tunables (override via env) ────────────────────────────────────────────
BASELINE_USERS = int(os.getenv("BASELINE_USERS", "10"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "100"))
LOW_DURATION = int(os.getenv("LOW_DURATION", "60"))
PEAK_DURATION = int(os.getenv("PEAK_DURATION", "60"))
SPAWN_RATE = int(os.getenv("SPAWN_RATE", "20"))


class BoutiqueUser(HttpUser):
    """A tiny synthetic user that exercises a handful of Online Boutique paths.

    These paths drive CPU on the `frontend` Deployment (the Phase 1 target).
    The product list is hardcoded — Online Boutique product IDs in v0.10.5.
    """

    wait_time = between(1, 3)

    PRODUCT_IDS = [
        "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O",
        "L9ECAV7KIM", "2ZYFJ3GM2N", "0PUK6V6EV0",
    ]

    @task(4)
    def browse_home(self) -> None:
        self.client.get("/")

    @task(3)
    def view_product(self) -> None:
        pid = random.choice(self.PRODUCT_IDS)
        self.client.get(f"/product/{pid}", name="/product/[id]")

    @task(1)
    def view_cart(self) -> None:
        self.client.get("/cart")


class BurstShape(LoadTestShape):
    """Step function: BASELINE → PEAK → BASELINE.

    LoadTestShape lets us script user counts over time, which is closer to
    what §3.9 describes than the `-u/-r` CLI flags alone. Phase 5 will
    replace this with the canonical 50→500→50 burst and add the ramp,
    periodic and Alibaba-trace shapes.
    """

    def tick(self):
        t = self.get_run_time()
        if t < LOW_DURATION:
            return BASELINE_USERS, SPAWN_RATE
        if t < LOW_DURATION + PEAK_DURATION:
            return PEAK_USERS, SPAWN_RATE
        if t < 2 * LOW_DURATION + PEAK_DURATION:
            return BASELINE_USERS, SPAWN_RATE
        return None       # signal Locust to stop
