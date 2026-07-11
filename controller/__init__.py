"""Hybrid predictive and explainable autoscaling controller.

This package implements the controller artefact described in Chapters 3 and 4
of the dissertation. The control loop is:

    Prometheus telemetry  →  Forecaster (§3.6)
                          →  Decision engine (§3.7)
                          →  Kubernetes scale subresource
                          →  Evidence bundle (§3.8)

The decision engine is pure (no I/O) and therefore unit-testable in
isolation. The I/O adapters live in `prometheus_client`, `k8s_actuator`,
and `evidence_bundle`. The runtime loop is in `loop`; the CLI entry point
in `main`.
"""

from .config import EngineConfig
from .decision_engine import DecisionEngine
from .state import Decision, EngineState

__all__ = ["DecisionEngine", "EngineConfig", "EngineState", "Decision"]
