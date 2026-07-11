"""Tests for Phase 3 forecasting registry and artefact export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from controller.config import EngineConfig
from controller.forecaster_loader import ModelRegistry
from forecasting import Forecast, Forecaster
from forecasting.run_forecasting import (
    ForecastingRunConfig,
    build_registry_entry,
    make_factories,
    persist_selected_artifact,
    write_model_registry,
)
from forecasting.selection import ForecasterScore, SelectionReport


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


def _report(selected: str = "seasonal_naive") -> SelectionReport:
    score = ForecasterScore(
        name=selected,
        fold_rmses=[1.2, 1.0],
        fold_maes=[0.7, 0.6],
        fold_pinball=[0.3, 0.25],
        inference_p95_seconds=0.002,
        n_parameters=4,
    )
    return SelectionReport(
        service="frontend",
        horizon_seconds=30,
        scores={selected: score},
        selected_name=selected,
        selection_reason="lowest RMSE (1.1000)",
    )


def test_registry_export_is_loadable_by_controller(tmp_path: Path):
    config = ForecastingRunConfig(
        telemetry_dir=tmp_path,
        out_dir=tmp_path,
        namespace="default",
        target_metric="cpu",
        rho=0.3,
        sigma_max=0.2,
    )
    entry = build_registry_entry(_report("holt_winters"), config)
    registry_path = write_model_registry(
        tmp_path / "registry.yaml",
        entries=[entry],
        generated_at_utc="2026-06-24T00:00:00+00:00",
        source_telemetry=tmp_path / "telemetry.parquet",
        selection_csv=tmp_path / "selection.csv",
        h1_csv=tmp_path / "h1.csv",
        full_grid=False,
    )

    registry = ModelRegistry.from_yaml(registry_path)
    selected = registry.select(_cfg())

    assert registry.full_grid is False
    assert selected.forecaster == "holt_winters"
    assert selected.target_metric == "cpu"
    assert selected.rho == 0.3
    assert selected.sigma_max == 0.2
    assert selected.params["refit_each_predict"] is True
    assert selected.validation["mean_rmse"] == 1.1


def test_selected_ml_forecaster_is_saved_with_registry_relative_artifact_path(tmp_path: Path):
    class _PersistableForecaster(Forecaster):
        name = "xgboost"

        def __init__(self):
            self.fit_called = False

        def fit(self, history):
            self.fit_called = True
            return self

        def predict(self, history, horizon_seconds):
            return Forecast(point=1.0, sigma=0.1)

        def save_artifact(self, path):
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "marker.txt").write_text("saved")
            return out

    model = _PersistableForecaster()
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2026-06-24", periods=4, freq="15s", tz="UTC"),
    )

    artifact_path = persist_selected_artifact(
        _report("xgboost"),
        series,
        {"xgboost": lambda: model},
        tmp_path / "models",
        registry_dir=tmp_path,
    )

    assert model.fit_called is True
    assert artifact_path == Path("models/frontend-h30-xgboost")
    assert (tmp_path / artifact_path / "marker.txt").read_text() == "saved"


def test_full_grid_defaults_to_five_splits_and_dev_defaults_to_three(tmp_path: Path):
    dev = ForecastingRunConfig(telemetry_dir=tmp_path, out_dir=tmp_path)
    full = ForecastingRunConfig(telemetry_dir=tmp_path, out_dir=tmp_path, full_grid=True)
    explicit = ForecastingRunConfig(
        telemetry_dir=tmp_path,
        out_dir=tmp_path,
        full_grid=True,
        n_splits=2,
    )

    assert dev.resolved_n_splits == 3
    assert full.resolved_n_splits == 5
    assert explicit.resolved_n_splits == 2


def test_lstm_device_config_is_passed_to_factory(tmp_path: Path):
    config = ForecastingRunConfig(
        telemetry_dir=tmp_path,
        out_dir=tmp_path,
        lstm_device="cpu",
    )

    lstm = make_factories(config)["lstm"]()

    assert lstm.device_preference == "cpu"
