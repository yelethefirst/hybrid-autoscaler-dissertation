"""Model selection harness (§3.6).

§3.6: "Model selection per service uses a multi-criteria decision. Primary
criterion: lowest RMSE on the held-out validation fold averaged across
walk-forward splits. Tiebreaker one: lowest inference latency measured at
the ninety-fifth percentile (relevant because the forecaster runs in the
scaling-decision critical path). Tiebreaker two: parsimony (fewest
parameters)."

The harness:
    1. Walk-forward CV across `walk_forward_splits` from data.splits.
    2. For each fold, fit each forecaster on the train portion, predict
       the next horizon step from each rolling origin in the val portion.
    3. Aggregate RMSE and MAE per (forecaster, fold), and overall mean.
    4. Time inference latency over a small benchmark of predict calls.
    5. Apply the three-criterion selection rule.

The result is a `SelectionReport` that feeds Chapter 4 §4.2.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.splits import walk_forward_splits

from .base import Forecaster, ForecasterFaultError
from .metrics import mae, pinball_loss, rmse

LOGGER = logging.getLogger(__name__)

# A single fit+predict taking longer than this logs a loud warning — the
# DEV-010 lesson: a hung fit must be visible in the log, not discoverable only
# by sampling CPU counters hours later.
SLOW_FIT_WARN_SECONDS = 600


def _write_heartbeat(path: Optional[Path], **fields) -> None:
    """Best-effort progress heartbeat (JSON, atomically replaced each update)."""
    if path is None:
        return
    try:
        payload = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except OSError:
        pass  # progress reporting must never break the run


class _SlowFitAlarm:
    """Logs an escalating warning if the wrapped block runs too long."""

    def __init__(self, label: str, warn_seconds: int = SLOW_FIT_WARN_SECONDS):
        self.label = label
        self.warn_seconds = warn_seconds
        self._timer: Optional[threading.Timer] = None

    def _warn(self):
        LOGGER.warning(
            "%s still running after %ds — possible hang (DEV-010 class); "
            "check CPU activity before assuming progress",
            self.label, self.warn_seconds,
        )
        # re-arm so long hangs keep shouting
        self._timer = threading.Timer(self.warn_seconds, self._warn)
        self._timer.daemon = True
        self._timer.start()

    def __enter__(self):
        self._timer = threading.Timer(self.warn_seconds, self._warn)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *exc):
        if self._timer is not None:
            self._timer.cancel()
        return False


# ---------------------------------------------------------------------- #
# Results                                                                 #
# ---------------------------------------------------------------------- #
@dataclass
class ForecasterScore:
    """Per-forecaster aggregated results."""

    name: str
    fold_rmses: List[float] = field(default_factory=list)
    fold_maes: List[float] = field(default_factory=list)
    fold_pinball: List[float] = field(default_factory=list)
    inference_p95_seconds: float = float("inf")
    n_parameters: int = 0
    failed_folds: int = 0

    @property
    def mean_rmse(self) -> float:
        return float(np.mean(self.fold_rmses)) if self.fold_rmses else float("inf")

    @property
    def mean_mae(self) -> float:
        return float(np.mean(self.fold_maes)) if self.fold_maes else float("inf")

    @property
    def mean_pinball(self) -> float:
        return float(np.mean(self.fold_pinball)) if self.fold_pinball else float("inf")

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "mean_rmse": self.mean_rmse,
            "mean_mae": self.mean_mae,
            "mean_pinball_q70": self.mean_pinball,
            "inference_p95_seconds": self.inference_p95_seconds,
            "n_parameters": self.n_parameters,
            "failed_folds": self.failed_folds,
            "fold_rmses": list(self.fold_rmses),
            "fold_maes": list(self.fold_maes),
        }


@dataclass
class SelectionReport:
    """Aggregated walk-forward CV report for one service / horizon."""

    service: str
    horizon_seconds: int
    scores: Dict[str, ForecasterScore]
    selected_name: str
    selection_reason: str
    rmse_vs_mae_selection_agrees: bool = True

    def as_table(self) -> pd.DataFrame:
        rows = [s.as_dict() for s in self.scores.values()]
        return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)


# ---------------------------------------------------------------------- #
# Public API                                                              #
# ---------------------------------------------------------------------- #
ForecasterFactory = Callable[[], Forecaster]


def evaluate_forecasters(
    history: pd.Series,
    factories: Dict[str, ForecasterFactory],
    *,
    service: str = "unknown",
    horizon_seconds: int = 30,
    n_splits: int = 5,
    inference_latency_trials: int = 10,
    pinball_quantile: float = 0.7,
    heartbeat_path: Optional[Path] = None,
) -> SelectionReport:
    """Run walk-forward CV across all forecasters and return a SelectionReport.

    Parameters
    ----------
    history : pd.Series
        The univariate target series (e.g. frontend CPU rate). Must be sorted
        chronologically by index.
    factories : dict[name → callable]
        Maps forecaster name → no-arg factory that returns a fresh
        `Forecaster` instance for each fold. (Each fold trains a fresh
        model — the standard walk-forward pattern.)
    service : str
        Service label (recorded in the report).
    horizon_seconds : int
        Forecast horizon. Must match what the candidates expect.
    n_splits : int
        Number of walk-forward folds (§3.5: 5).
    inference_latency_trials : int
        Number of predict calls used to estimate inference p95 latency
        (§3.6 first tiebreaker).
    """
    n = len(history)
    scores: Dict[str, ForecasterScore] = {
        name: ForecasterScore(name=name) for name in factories
    }

    n_models = len(factories)
    for fold_i, (train_idx, val_idx) in enumerate(
        walk_forward_splits(n, n_splits=n_splits), start=1
    ):
        train_series = history.iloc[train_idx]
        val_series = history.iloc[val_idx]

        for model_i, (name, make) in enumerate(factories.items(), start=1):
            label = f"{service} fold {fold_i}/{n_splits} model {model_i}/{n_models} ({name})"
            LOGGER.info("start  %s  train=%d val=%d", label, len(train_idx), len(val_idx))
            _write_heartbeat(
                heartbeat_path,
                service=service, fold=fold_i, n_folds=n_splits,
                model=name, model_index=model_i, n_models=n_models,
                phase="fitting",
            )
            t0 = time.perf_counter()
            model = make()
            try:
                with _SlowFitAlarm(label):
                    model.fit(train_series)
                    preds, truths = _rolling_origin_predict(
                        model, train_series, val_series, horizon_seconds=horizon_seconds
                    )
            except Exception as exc:
                scores[name].failed_folds += 1
                LOGGER.warning("failed %s after %.1fs: %s: %s",
                               label, time.perf_counter() - t0, type(exc).__name__, exc)
                continue
            elapsed = time.perf_counter() - t0
            if len(preds) == 0:
                scores[name].failed_folds += 1
                LOGGER.warning("failed %s after %.1fs: no predictions produced", label, elapsed)
                continue
            fold_rmse = rmse(truths, preds)
            scores[name].fold_rmses.append(fold_rmse)
            scores[name].fold_maes.append(mae(truths, preds))
            scores[name].fold_pinball.append(pinball_loss(truths, preds, pinball_quantile))
            LOGGER.info("done   %s  rmse=%.5f  %.1fs", label, fold_rmse, elapsed)

    # Time inference latency on each candidate using the full series.
    for name, make in factories.items():
        model = make()
        try:
            model.fit(history.iloc[: int(0.8 * n)])
        except Exception:
            scores[name].inference_p95_seconds = float("inf")
            scores[name].n_parameters = 0
            continue
        scores[name].n_parameters = model.n_parameters()
        scores[name].inference_p95_seconds = _measure_inference_p95(
            model, history, horizon_seconds, trials=inference_latency_trials
        )

    # Apply §3.6 three-criterion selection.
    selected_name, reason = _select_best(scores)

    # §3.6 sensitivity check: does MAE-based selection agree with RMSE?
    mae_winner = _winner_by("mean_mae", scores)
    return SelectionReport(
        service=service,
        horizon_seconds=horizon_seconds,
        scores=scores,
        selected_name=selected_name,
        selection_reason=reason,
        rmse_vs_mae_selection_agrees=(mae_winner == selected_name),
    )


# ---------------------------------------------------------------------- #
# Selection rule (§3.6)                                                   #
# ---------------------------------------------------------------------- #
def _select_best(scores: Dict[str, ForecasterScore]) -> Tuple[str, str]:
    """Apply §3.6 three-criterion selection. Returns (winner_name, reason)."""
    viable = {n: s for n, s in scores.items() if s.fold_rmses}
    if not viable:
        return "", "no viable forecaster (all folds failed)"

    # Primary: lowest mean RMSE on val folds.
    min_rmse = min(s.mean_rmse for s in viable.values())
    # Tolerance for tiebreaker: within 1% of best RMSE.
    # Guard against NaN mean_rmse (can occur if truths contain NaN despite
    # imputation): fall back to all viable forecasters so selection continues.
    rmse_tied = {n: s for n, s in viable.items() if s.mean_rmse <= min_rmse * 1.01}
    if not rmse_tied:
        rmse_tied = viable
    if len(rmse_tied) == 1:
        winner = next(iter(rmse_tied))
        return winner, f"lowest RMSE ({viable[winner].mean_rmse:.4f})"

    # Tiebreaker 1: lowest inference p95 latency.
    # Tolerance is the larger of 10% relative and 1 ms absolute — sub-ms
    # differences are not operationally meaningful at a 15 s tick and
    # should fall through to the parsimony tiebreaker.
    min_lat = min(s.inference_p95_seconds for s in rmse_tied.values())
    lat_threshold = max(min_lat * 1.10, min_lat + 0.001)
    lat_tied = {n: s for n, s in rmse_tied.items()
                if s.inference_p95_seconds <= lat_threshold}
    if len(lat_tied) == 1:
        winner = next(iter(lat_tied))
        return winner, (
            f"RMSE tied; broken by lowest p95 inference latency "
            f"({viable[winner].inference_p95_seconds * 1000:.1f} ms)"
        )

    # Tiebreaker 2: parsimony (fewest parameters).
    winner = min(lat_tied, key=lambda n: lat_tied[n].n_parameters)
    return winner, (
        f"RMSE + latency tied; broken by parsimony "
        f"({viable[winner].n_parameters} parameters)"
    )


def _winner_by(field_name: str, scores: Dict[str, ForecasterScore]) -> str:
    viable = {n: s for n, s in scores.items() if s.fold_rmses}
    if not viable:
        return ""
    return min(viable, key=lambda n: getattr(viable[n], field_name))


# ---------------------------------------------------------------------- #
# Rolling-origin prediction inside one walk-forward fold                  #
# ---------------------------------------------------------------------- #
def _rolling_origin_predict(
    model: Forecaster,
    train_series: pd.Series,
    val_series: pd.Series,
    *,
    horizon_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each origin t in val: predict horizon-ahead, compare with truth."""
    history = train_series.copy()
    preds: List[float] = []
    truths: List[float] = []

    # Step through val_series one sample at a time.
    sample_interval = _infer_interval_seconds(history)
    h_samples = max(1, round(horizon_seconds / sample_interval))

    for i in range(len(val_series) - h_samples + 1):
        # Use history-so-far to predict the value h steps ahead in val.
        try:
            f = model.predict(history, horizon_seconds=horizon_seconds)
        except ForecasterFaultError:
            # Append the latest val sample to history and continue.
            history = pd.concat([history, val_series.iloc[i : i + 1]])
            continue
        preds.append(f.point)
        truths.append(float(val_series.iloc[i + h_samples - 1]))
        # Advance the rolling origin by one sample.
        history = pd.concat([history, val_series.iloc[i : i + 1]])
    return np.asarray(preds), np.asarray(truths)


def _measure_inference_p95(
    model: Forecaster, history: pd.Series, horizon_seconds: int, *, trials: int
) -> float:
    """Return the p95 of `trials` predict() durations on the latest history."""
    times: List[float] = []
    n = len(history)
    for i in range(trials):
        cutoff = max(int(n * 0.5), n - i - 1)
        sub = history.iloc[: cutoff + 1]
        t0 = time.perf_counter()
        try:
            model.predict(sub, horizon_seconds=horizon_seconds)
        except Exception:
            return float("inf")
        times.append(time.perf_counter() - t0)
    return float(np.quantile(times, 0.95))


def _infer_interval_seconds(series: pd.Series) -> int:
    if isinstance(series.index, pd.DatetimeIndex) and len(series) >= 2:
        delta = series.index[1] - series.index[0]
        return max(1, int(delta.total_seconds()))
    return 15
