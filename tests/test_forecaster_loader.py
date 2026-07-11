"""Tests for controller forecaster selection and model-registry loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from controller.config import EngineConfig
from controller.forecaster_loader import (
    ModelRegistry,
    build_forecaster,
    build_forecaster_from_registry,
)
from forecasting import HoltWinters, SARIMA, SeasonalNaive


def _cfg(**overrides) -> EngineConfig:
    defaults = dict(
        service="frontend",
        namespace="default",
        r_min=1,
        r_max=10,
        delta_s=2,
        rho=0.3,
        k=1.5,
        sigma_max=0.2,
        horizon_seconds=30,
        tick_seconds=15,
        history_seconds=600,
        target_utilisation=0.5,
        hpa_stabilisation_window_seconds=300,
        metric_source="cpu",
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


def test_build_forecaster_supports_online_safe_families():
    cfg = _cfg()
    assert isinstance(build_forecaster("seasonal_naive", cfg), SeasonalNaive)
    assert isinstance(build_forecaster("holt_winters", cfg), HoltWinters)
    assert isinstance(build_forecaster("sarima", cfg), SARIMA)


def test_sarima_controller_default_refits_each_predict():
    model = build_forecaster("sarima", _cfg())
    assert isinstance(model, SARIMA)
    assert model.refit_each_predict is True


def test_build_forecaster_rejects_unloaded_ml_models():
    with pytest.raises(NotImplementedError, match="trained artefact loader"):
        build_forecaster("xgboost", _cfg())


def test_model_registry_selects_matching_service_namespace_horizon(tmp_path: Path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: frontend
    namespace: default
    horizon_seconds: 30
    forecaster: holt_winters
    params:
      period_seconds: 60
""".strip()
    )

    registry = ModelRegistry.from_yaml(path)
    entry = registry.select(_cfg())
    assert entry.forecaster == "holt_winters"

    model = build_forecaster_from_registry(path, _cfg())
    assert isinstance(model, HoltWinters)


def test_model_registry_rejects_missing_entry(tmp_path: Path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: checkoutservice
    namespace: default
    horizon_seconds: 30
    forecaster: seasonal_naive
""".strip()
    )

    with pytest.raises(ValueError, match="no entry"):
        ModelRegistry.from_yaml(path).select(_cfg())


def test_config_rejects_requests_metric_source():
    # The control loop only ever feeds CPU history to the forecaster, so a
    # "requests" metric_source would silently receive CPU inputs. It is
    # rejected at config validation until the loop supports it
    # (2026-07-05 code review).
    with pytest.raises(Exception, match="cpu"):
        _cfg(metric_source="requests")


def test_model_registry_rejects_mismatched_target_metric(tmp_path: Path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: frontend
    namespace: default
    target_metric: requests
    horizon_seconds: 30
    forecaster: seasonal_naive
""".strip()
    )

    registry = ModelRegistry.from_yaml(path)
    # A cpu-config controller must not silently accept a requests-trained model.
    with pytest.raises(ValueError, match="metric=cpu"):
        registry.select(_cfg(metric_source="cpu"))


def test_model_registry_rejects_artifact_path_for_param_only_forecaster(tmp_path: Path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: frontend
    namespace: default
    horizon_seconds: 30
    forecaster: seasonal_naive
    artifact_path: models/frontend.pkl
""".strip()
    )

    with pytest.raises(ValueError, match="params only"):
        build_forecaster_from_registry(path, _cfg())


def test_model_registry_loads_xgboost_artifact_path(tmp_path: Path, monkeypatch):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: frontend
    namespace: default
    horizon_seconds: 30
    forecaster: xgboost
    artifact_path: artifacts/frontend-xgboost
""".strip()
    )
    seen = {}

    class DummyForecaster:
        name = "xgboost"

    def fake_load_artifact(artifact_path):
        seen["path"] = artifact_path
        return DummyForecaster()

    monkeypatch.setattr(
        "controller.forecaster_loader.XGBoostForecaster.load_artifact",
        fake_load_artifact,
    )

    model = build_forecaster_from_registry(path, _cfg())
    assert model.name == "xgboost"
    assert seen["path"] == tmp_path / "artifacts/frontend-xgboost"


def test_model_registry_requires_ml_artifact_path(tmp_path: Path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
entries:
  - service: frontend
    namespace: default
    horizon_seconds: 30
    forecaster: lstm
""".strip()
    )

    with pytest.raises(ValueError, match="artifact_path"):
        build_forecaster_from_registry(path, _cfg())
