"""Kubernetes scale-subresource client.

Reads and writes the replica count of a Deployment via the standard `/scale`
subresource. This is the same endpoint Kubernetes HPA uses, so we are
guaranteed identical actuation semantics to the baseline (a requirement for
the §3.9 A/B comparison to be fair).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from kubernetes import client, config

LOGGER = logging.getLogger(__name__)


class K8sActuator:
    """Patches `Deployment/scale` to change replica counts."""

    def __init__(
        self,
        namespace: str,
        kubeconfig: Optional[str | Path] = None,
        in_cluster: bool = False,
        dry_run: bool = False,
    ):
        self.namespace = namespace
        self.dry_run = dry_run
        if in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config(config_file=str(kubeconfig) if kubeconfig else None)
        self._apps = client.AppsV1Api()

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #
    def get_replicas(self, deployment: str) -> int:
        """Return the current spec.replicas of the Deployment."""
        scale = self._apps.read_namespaced_deployment_scale(
            name=deployment, namespace=self.namespace
        )
        return int(scale.spec.replicas or 0)

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #
    def set_replicas(self, deployment: str, replicas: int) -> int:
        """Set spec.replicas. Returns the new spec.replicas reported by the API.

        No-op if the requested count equals the current count (avoids
        no-op PATCHes that could otherwise inflate audit-log churn).
        """
        if replicas < 0:
            raise ValueError(f"replicas must be ≥ 0 (got {replicas})")

        current = self.get_replicas(deployment)
        if replicas == current:
            return current

        if self.dry_run:
            LOGGER.info(
                "DRY-RUN: would scale deployment=%s ns=%s from=%d to=%d",
                deployment,
                self.namespace,
                current,
                replicas,
            )
            return replicas

        body = {"spec": {"replicas": replicas}}
        result = self._apps.patch_namespaced_deployment_scale(
            name=deployment, namespace=self.namespace, body=body
        )
        new = int(result.spec.replicas or 0)
        LOGGER.info(
            "scaled deployment=%s ns=%s from=%d to=%d",
            deployment,
            self.namespace,
            current,
            new,
        )
        return new
