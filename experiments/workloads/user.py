"""Shared Locust user class for all Online Boutique workloads.

All canonical §3.9 workload files import BoutiqueUser from this module so that
the user behaviour is identical across burst, ramp, periodic, and trace-replay
trials. The task mix targets the `frontend` Deployment (the primary autoscaling
target) while exercising a realistic cross-service call graph.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

# Reproducible request mix: the supervisor exports the per-trial seed
# (previously the seed existed only as result metadata — 2026-07-05 review).
# Gevent scheduling still interleaves users nondeterministically, so this
# makes the *choice sequence* reproducible, not exact request timing.
_seed = os.getenv("TRIAL_SEED")
if _seed is not None:
    random.seed(int(_seed))

PRODUCT_IDS = [
    "OLJCESPC7Z",
    "66VCHSJNUP",
    "1YMWWN1N4O",
    "L9ECAV7KIM",
    "2ZYFJ3GM2N",
    "0PUK6V6EV0",
    "LS4PSXUNUM",
    "9SIQT8TOJO",
    "6E92ZMYYFZ",
]


class BoutiqueUser(HttpUser):
    """Synthetic user exercising the Online Boutique v0.10.5 frontend paths.

    Task weights reflect a plausible e-commerce session:
      40% — browse home
      30% — view a product
      15% — view cart
      10% — checkout (add to cart, then POST /cart/checkout — exercises
            checkoutservice, paymentservice, emailservice, shippingservice)
       5% — currency selection (exercises currencyservice via frontend)

    The checkout task mirrors Online Boutique's own loadgenerator: the previous
    `GET /checkout` hit a nonexistent route (404) and generated no downstream
    load at all (DEV-017), leaving checkout/payment/email/shipping idle in the
    pilot training telemetry.
    """

    wait_time = between(1, 3)

    @task(40)
    def browse_home(self) -> None:
        self.client.get("/")

    @task(30)
    def view_product(self) -> None:
        pid = random.choice(PRODUCT_IDS)
        self.client.get(f"/product/{pid}", name="/product/[id]")

    @task(15)
    def view_cart(self) -> None:
        self.client.get("/cart")

    @task(10)
    def checkout(self) -> None:
        pid = random.choice(PRODUCT_IDS)
        self.client.post(
            "/cart",
            data={"product_id": pid, "quantity": random.choice([1, 2, 3])},
            name="/cart [add]",
        )
        self.client.post(
            "/cart/checkout",
            data={
                "email": "someone@example.com",
                "street_address": "1600 Amphitheatre Parkway",
                "zip_code": "94043",
                "city": "Mountain View",
                "state": "CA",
                "country": "United States",
                # digits only — the v0.10.5 frontend validator rejects dashes
                "credit_card_number": "4432801561520454",
                "credit_card_expiration_month": "1",
                "credit_card_expiration_year": "2039",
                "credit_card_cvv": "672",
            },
            name="/cart/checkout",
        )

    @task(5)
    def set_currency(self) -> None:
        self.client.post(
            "/setCurrency",
            data={"currency_code": "EUR"},
            name="/setCurrency",
        )
