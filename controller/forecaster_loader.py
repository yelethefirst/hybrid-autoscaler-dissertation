"""Forecaster construction for the live controller.

Phase 1 hard-wired Seasonal Naive in `controller.main`. Phase 4 needs the
controller to load the selected per-service forecaster. This module provides
the conservative first step: CLI selection for online-safe forecasters and a
small registry format that can later point to trained XGBoost/LSTM artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from forecasting import (
    Forecaster,
    HoltWinters,
    LSTMForecaster,
    SARIMA,
    SeasonalNaive,
    XGBoostForecaster,
)

from .config import EngineConfig

ForecasterName = Literal["seasonal_naive", "holt_winters", "sarima"]
RegistryForecasterName = Literal[
    "seasonal_naive", "holt_winters", "sarima", "xgboost", "lstm"
]

SUPPORTED_ONLINE_FORECASTERS: tuple[str, ...] = (
    "seasonal_naive",
    "holt_winters",
    "sarima",
)


class ModelRegistryEntry(BaseModel):
    """One selected model entry for one service/horizon."""

    model_config = ConfigDict(extra="forbid")

    service: str
    namespace: str = "default"
    target_metric: str = "cpu"
    horizon_seconds: int
    forecaster: RegistryForecasterName
    params: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[Path] = None
    rho: Optional[float] = None
    sigma_max: Optional[float] = None
    validation: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistry(BaseModel):
    """Minimal model registry consumed by `controller.main`."""

    model_config = ConfigDict(extra="forbid")

    generated_at_utc: Optional[str] = None
    source_telemetry: Optional[Path] = None
    selection_csv: Optional[Path] = None
    h1_csv: Optional[Path] = None
    full_grid: Optional[bool] = None
    entries: List[ModelRegistryEntry]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelRegistry":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def select(self, config: EngineConfig) -> ModelRegistryEntry:
        """Return the entry matching the controller service/namespace/horizon."""
        matches = [
            entry
            for entry in self.entries
            if entry.service == config.service
            and entry.namespace == config.namespace
            and entry.target_metric == config.metric_source
            and entry.horizon_seconds == config.horizon_seconds
        ]
        if not matches:
            raise ValueError(
                "model registry has no entry for "
                f"{config.namespace}/{config.service} metric={config.metric_source} "
                f"h={config.horizon_seconds}s"
            )
        if len(matches) > 1:
            raise ValueError(
                "model registry has duplicate entries for "
                f"{config.namespace}/{config.service} metric={config.metric_source} "
                f"h={config.horizon_seconds}s"
            )
        return matches[0]


def build_forecaster(
    name: str,
    config: EngineConfig,
    params: Optional[Dict[str, Any]] = None,
) -> Forecaster:
    """Build an online-safe forecaster for the live control loop."""
    p = dict(params or {})
    period_seconds = int(p.pop("period_seconds", 60))
    sample_interval_seconds = int(p.pop("sample_interval_seconds", config.tick_seconds))

    if name == "seasonal_naive":
        return SeasonalNaive(
            period_seconds=period_seconds,
            sample_interval_seconds=sample_interval_seconds,
            **p,
        )
    if name == "holt_winters":
        return HoltWinters(
            period_seconds=period_seconds,
            sample_interval_seconds=sample_interval_seconds,
            **p,
        )
    if name == "sarima":
        p.setdefault("refit_each_predict", True)
        return SARIMA(
            period_seconds=period_seconds,
            sample_interval_seconds=sample_interval_seconds,
            **p,
        )
    if name in {"xgboost", "lstm"}:
        raise NotImplementedError(
            f"{name} requires a trained artefact loader and cannot be "
            "constructed from params only. Use --model-registry with an "
            "artifact_path entry for the selected trained model."
        )
    raise ValueError(
        f"unsupported forecaster '{name}'. Supported online forecasters: "
        f"{', '.join(SUPPORTED_ONLINE_FORECASTERS)}"
    )


def build_forecaster_from_registry(
    registry_path: str | Path,
    config: EngineConfig,
) -> Forecaster:
    """Build the selected forecaster for `config` from a registry YAML file."""
    registry_file = Path(registry_path)
    entry = ModelRegistry.from_yaml(registry_file).select(config)
    if entry.artifact_path is not None and entry.forecaster in SUPPORTED_ONLINE_FORECASTERS:
        raise ValueError(
            f"registry entry for {entry.service} supplies artifact_path, but "
            f"{entry.forecaster} is constructed from params only"
        )
    if entry.forecaster == "xgboost":
        if entry.artifact_path is None:
            raise ValueError("xgboost registry entries must supply artifact_path")
        return XGBoostForecaster.load_artifact(
            _resolve_artifact_path(registry_file, entry.artifact_path)
        )
    if entry.forecaster == "lstm":
        if entry.artifact_path is None:
            raise ValueError("lstm registry entries must supply artifact_path")
        return LSTMForecaster.load_artifact(
            _resolve_artifact_path(registry_file, entry.artifact_path)
        )
    return build_forecaster(entry.forecaster, config, entry.params)


def _resolve_artifact_path(registry_path: Path, artifact_path: Path) -> Path:
    if artifact_path.is_absolute():
        return artifact_path
    return registry_path.parent / artifact_path
