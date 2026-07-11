"""End-to-end Phase 3 forecasting run and model-registry export.

This module backs ``bin/run-forecasting.sh``. It keeps the shell wrapper small
and makes the dissertation-critical output path testable:

* per-service model-selection CSV;
* H1 comparison CSV;
* selected-model artefacts for XGBoost/LSTM;
* a controller-loadable model registry YAML.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from data.collect.exporter import load_campaign
from data.synthetic import to_wide

from . import HoltWinters, LSTMForecaster, SARIMA, SeasonalNaive, XGBoostForecaster
from .base import Forecaster
from .comparison import h1_test
from .selection import SelectionReport, evaluate_forecasters


ForecasterFactory = Callable[[], Forecaster]

ARTIFACT_FORECASTERS = {"xgboost", "lstm"}
ONLINE_PARAM_FORECASTERS = {"seasonal_naive", "holt_winters", "sarima"}


@dataclass(frozen=True)
class ForecastingRunConfig:
    """Configuration for one Phase 3 forecasting run."""

    telemetry_dir: Path
    out_dir: Path
    artifact_dir: Optional[Path] = None
    namespace: str = "default"
    target_metric: str = "cpu"
    horizon_seconds: int = 30
    sample_interval_seconds: int = 15
    period_seconds: int = 60
    full_grid: bool = False
    skip_lstm: bool = False
    skip_sarima: bool = False
    lstm_device: str = "auto"
    n_splits: Optional[int] = None
    inference_latency_trials: int = 5
    seed: int = 0
    rho: Optional[float] = None
    sigma_max: Optional[float] = None
    fail_on_h1_not_supported: bool = True

    @property
    def resolved_n_splits(self) -> int:
        return self.n_splits if self.n_splits is not None else (5 if self.full_grid else 3)

    @property
    def resolved_artifact_dir(self) -> Path:
        if self.artifact_dir is not None:
            return self.artifact_dir
        return self.out_dir / "models"


@dataclass(frozen=True)
class ForecastingRunResult:
    """Paths and verdict produced by a Phase 3 forecasting run."""

    selection_path: Optional[Path]
    h1_path: Optional[Path]
    registry_path: Optional[Path]
    artifact_dir: Path
    h1_supported: bool
    services_evaluated: list[str]


def make_factories(config: ForecastingRunConfig) -> Dict[str, ForecasterFactory]:
    """Build the five §3.6 forecaster factories for this run."""

    horizon = config.horizon_seconds
    sample_interval = config.sample_interval_seconds
    period = config.period_seconds

    if config.full_grid:
        xgb_factory = lambda: XGBoostForecaster(  # noqa: E731
            horizon_seconds=horizon,
            sample_interval_seconds=sample_interval,
            random_state=config.seed,
        )
        lstm_factory = lambda: LSTMForecaster(  # noqa: E731
            horizon_seconds=horizon,
            sample_interval_seconds=sample_interval,
            seed=config.seed,
            device=config.lstm_device,
        )
    else:
        xgb_factory = lambda: XGBoostForecaster(  # noqa: E731
            horizon_seconds=horizon,
            sample_interval_seconds=sample_interval,
            n_estimators_grid=[100],
            max_depth_grid=[3, 5],
            learning_rate_grid=[0.1],
            random_state=config.seed,
        )
        lstm_factory = lambda: LSTMForecaster(  # noqa: E731
            horizon_seconds=horizon,
            sample_interval_seconds=sample_interval,
            hidden_sizes=[32],
            seq_lens=[30],
            dropouts=[0.0],
            num_layers_grid=[1],
            max_epochs=20,
            patience=5,
            seed=config.seed,
            device=config.lstm_device,
        )

    factories = {
        "seasonal_naive": lambda: SeasonalNaive(
            period_seconds=period,
            sample_interval_seconds=sample_interval,
        ),
        # refit_each_predict=True matches how the live controller runs these
        # models (see _registry_params): without it, rolling-origin CV scores a
        # stale model while deployment refits every tick — selection would then
        # rank behaviour that never ships (2026-07-05 code review).
        "holt_winters": lambda: HoltWinters(
            period_seconds=period,
            sample_interval_seconds=sample_interval,
            refit_each_predict=True,
        ),
        "sarima": lambda: SARIMA(
            period_seconds=period,
            sample_interval_seconds=sample_interval,
            refit_each_predict=True,
        ),
        "xgboost": xgb_factory,
        "lstm": lstm_factory,
    }
    if config.skip_lstm:
        del factories["lstm"]
        print("SKIP_LSTM: excluding LSTM from this run")
    if config.skip_sarima:
        del factories["sarima"]
        print("SKIP_SARIMA: excluding SARIMA from this run (DEV-010: hangs on long series)")
    return factories


def persist_selected_artifact(
    report: SelectionReport,
    history: pd.Series,
    factories: Dict[str, ForecasterFactory],
    artifact_dir: Path,
    *,
    registry_dir: Path,
) -> Optional[Path]:
    """Train and save the selected ML model, returning a registry-relative path.

    Statistical forecasters are reconstructed by controller params, so only
    XGBoost and LSTM write trained artefacts.
    """

    selected = report.selected_name
    if selected not in ARTIFACT_FORECASTERS:
        return None
    if selected not in factories:
        raise ValueError(f"no factory registered for selected forecaster '{selected}'")

    model = factories[selected]()
    model.fit(history)
    save_artifact = getattr(model, "save_artifact", None)
    if save_artifact is None:
        raise TypeError(f"selected forecaster '{selected}' does not support artefact saving")

    destination = artifact_dir / f"{_slug(report.service)}-h{report.horizon_seconds}-{selected}"
    saved_path = Path(save_artifact(destination))
    return _relative_path(saved_path, registry_dir)


def build_registry_entry(
    report: SelectionReport,
    config: ForecastingRunConfig,
    *,
    artifact_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build one controller-loadable model-registry entry."""

    if not report.selected_name:
        raise ValueError(f"service {report.service} has no selected forecaster")

    entry: Dict[str, Any] = {
        "service": report.service,
        "namespace": config.namespace,
        "target_metric": config.target_metric,
        "horizon_seconds": report.horizon_seconds,
        "forecaster": report.selected_name,
        "params": _registry_params(report.selected_name, config),
        "validation": _selected_validation(report),
    }
    if artifact_path is not None:
        entry["artifact_path"] = str(artifact_path)
    if config.rho is not None:
        entry["rho"] = config.rho
    if config.sigma_max is not None:
        entry["sigma_max"] = config.sigma_max
    return entry


def write_model_registry(
    path: Path,
    *,
    entries: Iterable[Dict[str, Any]],
    generated_at_utc: str,
    source_telemetry: Path,
    selection_csv: Optional[Path],
    h1_csv: Optional[Path],
    full_grid: bool,
) -> Path:
    """Write the YAML registry consumed by ``controller.forecaster_loader``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_at_utc": generated_at_utc,
        "source_telemetry": str(source_telemetry),
        "full_grid": bool(full_grid),
        "entries": list(entries),
    }
    if selection_csv is not None:
        payload["selection_csv"] = str(selection_csv)
    if h1_csv is not None:
        payload["h1_csv"] = str(h1_csv)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def run(config: ForecastingRunConfig) -> ForecastingRunResult:
    """Execute the Phase 3 forecasting pipeline."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Support both multi-batch campaign dirs (telemetry_*.parquet) and single files.
    batch_files = sorted(config.telemetry_dir.glob("telemetry_*.parquet"))
    if batch_files:
        print(f"source telemetry: {config.telemetry_dir} ({len(batch_files)} batch files)")
        df_long = load_campaign(config.telemetry_dir)
        source_telemetry_path = config.telemetry_dir
    else:
        source_telemetry_path = _latest_parquet(config.telemetry_dir)
        print(f"source telemetry: {source_telemetry_path}")
        df_long = pd.read_parquet(source_telemetry_path)
    if "service" not in df_long.columns:
        raise ValueError("telemetry parquet must contain a 'service' column")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    factories = make_factories(config)

    all_selection_rows: list[pd.DataFrame] = []
    all_h1_rows: list[pd.DataFrame] = []
    registry_entries: list[Dict[str, Any]] = []
    services_evaluated: list[str] = []
    h1_any_supported = False

    services_in_data = sorted(df_long["service"].dropna().unique())
    print(f"services in telemetry: {services_in_data}")

    for service in services_in_data:
        wide = to_wide(df_long, service=service)
        if config.target_metric not in wide.columns or wide.empty or len(wide) < 80:
            print(f"skip {service}: insufficient {config.target_metric} data ({len(wide)} rows)")
            continue

        series = wide.set_index("timestamp")[config.target_metric]
        # Impute NaN gaps (collection artefacts such as port-forward restarts)
        # before passing to any forecaster.  linear interpolation by time is
        # appropriate for CPU/request-rate signals; ffill/bfill handles edges.
        n_nan = int(series.isna().sum())
        if n_nan:
            series = series.interpolate(method="time").ffill().bfill()
            print(f"  imputed {n_nan} NaN values in {service} series")
        print(f"evaluating {service} ({len(series)} samples)")
        try:
            report = evaluate_forecasters(
                series,
                factories=factories,
                service=service,
                horizon_seconds=config.horizon_seconds,
                n_splits=config.resolved_n_splits,
                inference_latency_trials=config.inference_latency_trials,
                heartbeat_path=config.out_dir / "forecasting_progress.json",
            )
        except Exception as exc:
            print(f"evaluation failed for {service}: {exc}")
            continue

        print(f"selected {service}: {report.selected_name} ({report.selection_reason})")
        services_evaluated.append(service)

        selection_table = report.as_table()
        selection_table["service"] = service
        all_selection_rows.append(selection_table)

        try:
            h1 = h1_test(report)
        except Exception as exc:
            print(f"H1 test failed for {service}: {exc}")
            h1 = None
        if h1 is not None:
            h1_any_supported = h1_any_supported or h1.h1_supported
            status = "supported" if h1.h1_supported else "not supported"
            print(f"H1 {status} for {service}")
            h1_table = h1.as_table()
            h1_table["service"] = service
            all_h1_rows.append(h1_table)

        try:
            artifact_path = persist_selected_artifact(
                report,
                series,
                factories,
                config.resolved_artifact_dir,
                registry_dir=config.out_dir,
            )
        except Exception as exc:
            print(f"selected artefact save failed for {service}: {exc}")
            continue
        registry_entries.append(
            build_registry_entry(report, config, artifact_path=artifact_path)
        )

    selection_path: Optional[Path] = None
    h1_path: Optional[Path] = None
    registry_path: Optional[Path] = None

    if all_selection_rows:
        selection_path = config.out_dir / f"phase3_{stamp}_selection.csv"
        pd.concat(all_selection_rows, ignore_index=True).to_csv(selection_path, index=False)
        print(f"selection table: {selection_path}")

    if all_h1_rows:
        h1_path = config.out_dir / f"phase3_{stamp}_h1.csv"
        pd.concat(all_h1_rows, ignore_index=True).to_csv(h1_path, index=False)
        print(f"H1 table: {h1_path}")

    if registry_entries:
        registry_path = config.out_dir / f"phase3_{stamp}_model_registry.yaml"
        write_model_registry(
            registry_path,
            entries=registry_entries,
            generated_at_utc=generated_at_utc,
            source_telemetry=source_telemetry_path,
            selection_csv=selection_path,
            h1_csv=h1_path,
            full_grid=config.full_grid,
        )
        print(f"model registry: {registry_path}")

    if not services_evaluated:
        raise RuntimeError("no services were evaluated")

    if config.fail_on_h1_not_supported and not h1_any_supported:
        print("Phase 3 exit criterion NOT MET: no candidate beat Seasonal Naive")
        raise SystemExit(1)

    print(
        "Phase 3 exit criterion "
        + ("MET: H1 supported in at least one service" if h1_any_supported else "not enforced")
    )
    return ForecastingRunResult(
        selection_path=selection_path,
        h1_path=h1_path,
        registry_path=registry_path,
        artifact_dir=config.resolved_artifact_dir,
        h1_supported=h1_any_supported,
        services_evaluated=services_evaluated,
    )


def main(argv: Optional[list[str]] = None) -> int:
    # Timestamped per-fold/per-model progress from forecasting.selection —
    # a FULL_GRID run must never be silent for hours (DEV-010 lesson).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args(argv)
    config = ForecastingRunConfig(
        telemetry_dir=args.telemetry_dir,
        out_dir=args.out_dir,
        artifact_dir=args.artifact_dir,
        namespace=args.namespace,
        target_metric=args.target_metric,
        horizon_seconds=args.horizon_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        period_seconds=args.period_seconds,
        full_grid=args.full_grid,
        skip_lstm=args.skip_lstm,
        skip_sarima=args.skip_sarima,
        lstm_device=args.lstm_device,
        n_splits=args.n_splits,
        inference_latency_trials=args.inference_latency_trials,
        seed=args.seed,
        rho=args.rho,
        sigma_max=args.sigma_max,
        fail_on_h1_not_supported=not args.allow_h1_failure,
    )
    try:
        run(config)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"forecasting run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 forecasting selection.")
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--target-metric", default="cpu")
    parser.add_argument("--horizon-seconds", type=int, default=30)
    parser.add_argument("--sample-interval-seconds", type=int, default=15)
    parser.add_argument("--period-seconds", type=int, default=60)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--inference-latency-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument(
        "--allow-h1-failure",
        action="store_true",
        help="Write outputs even when no candidate beats Seasonal Naive and exit 0.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Exclude LSTM from evaluation (use when training time is prohibitive).",
    )
    parser.add_argument(
        "--skip-sarima",
        action="store_true",
        help="Exclude SARIMA from evaluation (DEV-010: hangs on series > ~1000 rows).",
    )
    parser.add_argument(
        "--lstm-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help=(
            "PyTorch device for LSTM. Use cpu on Apple Silicon if MPS hangs; "
            "use cuda on a Linux GPU host."
        ),
    )
    return parser.parse_args(argv)


def _registry_params(name: str, config: ForecastingRunConfig) -> Dict[str, Any]:
    if name in {"seasonal_naive", "holt_winters", "sarima"}:
        params: Dict[str, Any] = {
            "period_seconds": config.period_seconds,
            "sample_interval_seconds": config.sample_interval_seconds,
        }
        if name in {"holt_winters", "sarima"}:
            params["refit_each_predict"] = True
        return params
    return {}


def _selected_validation(report: SelectionReport) -> Dict[str, Any]:
    score = report.scores.get(report.selected_name)
    if score is None:
        return {"selection_reason": report.selection_reason}

    return _to_builtin(
        {
            "selection_reason": report.selection_reason,
            "rmse_vs_mae_selection_agrees": report.rmse_vs_mae_selection_agrees,
            "mean_rmse": score.mean_rmse,
            "mean_mae": score.mean_mae,
            "mean_pinball_q70": score.mean_pinball,
            "inference_p95_seconds": score.inference_p95_seconds,
            "n_parameters": score.n_parameters,
            "failed_folds": score.failed_folds,
            "fold_rmses": score.fold_rmses,
            "fold_maes": score.fold_maes,
        }
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _latest_parquet(telemetry_dir: Path) -> Path:
    files = sorted(telemetry_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no parquet files in {telemetry_dir}")
    return files[-1]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")


def _relative_path(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
