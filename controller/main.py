"""CLI entry point for the controller.

Usage:
    uv run python -m controller.main --config controller/configs/frontend-local.yaml \
        --prometheus-url http://localhost:9090

Use `--forecaster` for explicit local selection, or `--model-registry` to
load the selected per-service forecaster from a registry file.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from .config import EngineConfig
from .evidence_bundle import EvidenceBundleWriter
from .forecaster_loader import (
    SUPPORTED_ONLINE_FORECASTERS,
    build_forecaster,
    build_forecaster_from_registry,
)
from .k8s_actuator import K8sActuator
from .loop import ControlLoop
from .prometheus_client import PrometheusClient


@click.command(name="hybrid-autoscaler")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the EngineConfig YAML.",
)
@click.option(
    "--prometheus-url",
    default="http://localhost:9090",
    show_default=True,
    help="Base URL of the Prometheus HTTP API.",
)
@click.option(
    "--forecaster",
    default="seasonal_naive",
    type=click.Choice(SUPPORTED_ONLINE_FORECASTERS, case_sensitive=False),
    show_default=True,
    help="Forecaster to construct directly when --model-registry is not supplied.",
)
@click.option(
    "--model-registry",
    "model_registry_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "YAML registry mapping service/namespace/target metric/horizon "
        "to the selected forecaster."
    ),
)
@click.option(
    "--kubeconfig",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to kubeconfig. Defaults to KUBECONFIG / ~/.kube/config.",
)
@click.option(
    "--in-cluster",
    is_flag=True,
    help="Use in-cluster Kubernetes service-account credentials.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=(
        "Log scaling decisions but skip kubectl scale. "
        "Metrics are still read from Prometheus and evidence is still written."
    ),
)
@click.option(
    "--evidence-path",
    "evidence_path_override",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Override the config's evidence_path (used by the experiment "
        "supervisor for per-trial evidence files). Falls back to the "
        "EVIDENCE_PATH environment variable, then to the config value."
    ),
)
@click.option(
    "--max-ticks",
    type=int,
    default=None,
    help="Stop after this many ticks. Default: run until SIGINT/SIGTERM.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
)
def main(
    config_path: Path,
    prometheus_url: str,
    forecaster: str,
    model_registry_path: Optional[Path],
    kubeconfig: Optional[Path],
    in_cluster: bool,
    dry_run: bool,
    evidence_path_override: Optional[Path],
    max_ticks: Optional[int],
    log_level: str,
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    cfg = EngineConfig.from_yaml(config_path)
    override = evidence_path_override or (
        Path(os.environ["EVIDENCE_PATH"]) if os.environ.get("EVIDENCE_PATH") else None
    )
    if override is not None:
        cfg = cfg.model_copy(update={"evidence_path": override})
        click.echo(f"▶  evidence path override: {override}")
    click.echo(f"▶  config loaded: service={cfg.service} namespace={cfg.namespace}")
    if dry_run:
        click.echo("▶  DRY-RUN mode: scaling decisions will be logged but not applied")

    if model_registry_path is not None:
        model = build_forecaster_from_registry(model_registry_path, cfg)
        click.echo(
            f"▶  model registry loaded: {model_registry_path.name} "
            f"→ {model.name} for {cfg.service}"
        )
    else:
        model = build_forecaster(forecaster.lower(), cfg)
    prom = PrometheusClient(prometheus_url)
    actuator = K8sActuator(
        namespace=cfg.namespace,
        kubeconfig=kubeconfig,
        in_cluster=in_cluster,
        dry_run=dry_run,
    )
    evidence = EvidenceBundleWriter(cfg.evidence_path)

    loop = ControlLoop(cfg, model, prom, actuator, evidence)
    loop.install_signal_handlers()

    click.echo(
        f"▶  starting control loop "
        f"(tick={cfg.tick_seconds}s, horizon={cfg.horizon_seconds}s, "
        f"forecaster={model.name})"
    )
    n = loop.run(max_ticks=max_ticks)
    click.echo(f"✅  stopped after {n} tick(s). Evidence: {cfg.evidence_path}")


if __name__ == "__main__":
    main(standalone_mode=False) if False else main()  # noqa: PLR1716
    sys.exit(0)
