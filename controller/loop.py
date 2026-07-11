"""Main control loop.

Ticks once every `tick_seconds` (15 s per §3.5). On each tick:
    1. Pull the metric history window from Prometheus.
    2. Ask the forecaster for f̂(t+h), σ̂(t+h).
       Catch ForecasterFaultError → engine takes the FALLBACK_FORECASTER_FAULT path.
    3. Read current replicas from the Kubernetes scale subresource.
    4. Read u_t (instantaneous utilisation) for the HPA-equivalent fallback path.
    5. Engine produces a Decision.
    6. Actuate (idempotent — no PATCH if replicas unchanged).
    7. Write the evidence-bundle row.

Loop is interruptible via SIGTERM/SIGINT; partial decision rows are not
left behind because the evidence writer flushes per line.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from forecasting.base import Forecaster, ForecasterFaultError

from .config import EngineConfig
from .decision_engine import DecisionEngine
from .evidence_bundle import EvidenceBundleWriter
from .k8s_actuator import K8sActuator
from .prometheus_client import PrometheusClient
from .state import Decision

LOGGER = logging.getLogger(__name__)


class ControlLoop:
    """Composes the engine with its I/O adapters and runs the tick loop."""

    def __init__(
        self,
        config: EngineConfig,
        forecaster: Forecaster,
        prometheus: PrometheusClient,
        actuator: K8sActuator,
        evidence: EvidenceBundleWriter,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.config = config
        self.forecaster = forecaster
        self.prom = prometheus
        self.actuator = actuator
        self.evidence = evidence
        self.engine = DecisionEngine(config)
        self._clock = clock
        self._wall_clock = wall_clock
        self._stop = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            LOGGER.info("received signal %s — stopping after current tick", signum)
            self._stop = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def stop(self) -> None:
        self._stop = True

    def run(self, max_ticks: Optional[int] = None) -> int:
        """Run the loop. Returns number of ticks executed."""
        tick = 0
        while not self._stop:
            t_start = self._clock()
            try:
                self._tick()
            except Exception:
                LOGGER.exception("tick %d failed", tick)
            tick += 1
            if max_ticks is not None and tick >= max_ticks:
                break
            elapsed = self._clock() - t_start
            sleep_for = max(0.0, self.config.tick_seconds - elapsed)
            if sleep_for == 0.0:
                LOGGER.warning(
                    "tick %d exceeded budget (%.2fs > %ds)",
                    tick,
                    elapsed,
                    self.config.tick_seconds,
                )
            time.sleep(sleep_for)
        return tick

    # ------------------------------------------------------------------ #
    # One tick                                                            #
    # ------------------------------------------------------------------ #
    def _tick(self) -> Decision:
        c = self.config

        # 1. History window from Prometheus
        history = self.prom.cpu_total_cores(
            service=c.service,
            namespace=c.namespace,
            lookback_seconds=c.history_seconds,
            step_seconds=c.tick_seconds,
        )

        # 2. Forecast
        fault_reason: Optional[str] = None
        point: Optional[float] = None
        sigma: Optional[float] = None
        shap_extra: dict = {}
        try:
            forecast = self.forecaster.predict(history, c.horizon_seconds)
            point, sigma = forecast.point, forecast.sigma
            # Phase 6: SHAP attribution — runs after a successful predict so
            # attribution failures can never affect the scaling decision.
            try:
                shap_extra = self.forecaster.shap_attribution(history, c.horizon_seconds)
            except Exception:
                pass  # attribution is best-effort
        except ForecasterFaultError as e:
            fault_reason = f"{type(e).__name__}: {e}"
            LOGGER.warning("forecaster fault: %s", fault_reason)
        except Exception as e:  # any unexpected exception → fault
            fault_reason = f"{type(e).__name__}: {e}"
            LOGGER.exception("forecaster raised unexpected exception")

        # 3. Current replicas (spec-side, source of truth for HPA semantics)
        current_replicas = self.actuator.get_replicas(c.service)

        # 4. u_t for the HPA-equivalent fallback path
        observed = self.prom.cpu_avg_utilisation(c.service, c.namespace)

        # 5. Decision
        decision = self.engine.decide(
            current_replicas=current_replicas,
            forecast_point=point,
            forecast_sigma=sigma,
            forecaster_name=self.forecaster.name,
            forecaster_fault_reason=fault_reason,
            observed_metric=observed,
            now_seconds=self._clock(),
        )

        # 6. Actuate
        self.actuator.set_replicas(c.service, decision.new_replicas)

        # 7. Persist evidence (include SHAP attribution if available)
        self.evidence.write(
            decision,
            history=history,
            extra={"shap": shap_extra} if shap_extra else None,
        )

        LOGGER.info(
            "tick service=%s state=%s replicas %d→%d forecast=%s sigma=%s",
            c.service,
            decision.state,
            decision.current_replicas,
            decision.new_replicas,
            "n/a" if point is None else f"{point:.3f}",
            "n/a" if sigma is None else f"{sigma:.3f}",
        )
        return decision
