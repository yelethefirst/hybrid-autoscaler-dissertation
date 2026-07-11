"""XGBoost forecaster (§3.6).

§3.6: "XGBoost (Chen and Guestrin, 2016) with the lagged-feature
representation specified in Section 3.5… grid over number of estimators
in {100, 200, 400}, max depth in {3, 5, 7}, learning rate in
{0.05, 0.1, 0.2}, early stopping on validation MAE with patience of
twenty rounds, objective reg:squarederror."

Feature representation
----------------------
Reuses `data.features.engineer.engineer_features` so the lag / rolling /
diff features are identical to those declared by the §3.5 feature schema.
At training time:
    1. Wrap history into a one-column wide DataFrame.
    2. Engineer features.
    3. Build (X, y) where y[i] = value at i+h (the horizon-ahead target).
    4. Drop NaN-bearing rows (opening window of the lag/rolling features).
At prediction time the same engineering is applied to the *tail* of the
incoming history, and the last row's features are fed to the model.

Prediction interval
-------------------
Quantile-residual σ̂ from validation residuals (§3.6 for ML models).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data.features.engineer import (
    DEFAULT_LAG_INTERVALS,
    DEFAULT_ROLLING_WINDOWS_SEC,
    engineer_features,
)

from .base import Forecast, Forecaster, ForecasterFaultError
from .intervals import quantile_residual_sigma


# §3.6 hyperparameter grid (single source of truth)
DEFAULT_N_ESTIMATORS = [100, 200, 400]
DEFAULT_MAX_DEPTH = [3, 5, 7]
DEFAULT_LEARNING_RATE = [0.05, 0.1, 0.2]
DEFAULT_EARLY_STOPPING_ROUNDS = 20


@dataclass(frozen=True)
class XGBHyperparams:
    n_estimators: int
    max_depth: int
    learning_rate: float


class XGBoostForecaster(Forecaster):
    """Gradient-boosted trees forecasting with the §3.6 hyperparameter grid."""

    name = "xgboost"

    def __init__(
        self,
        horizon_seconds: int = 30,
        sample_interval_seconds: int = 15,
        *,
        lag_intervals: Optional[List[int]] = None,
        rolling_windows_seconds: Optional[List[int]] = None,
        n_estimators_grid: Optional[List[int]] = None,
        max_depth_grid: Optional[List[int]] = None,
        learning_rate_grid: Optional[List[float]] = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        random_state: int = 0,
    ):
        self.horizon_seconds = horizon_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.horizon_samples = max(1, round(horizon_seconds / sample_interval_seconds))
        self.lag_intervals = list(lag_intervals or DEFAULT_LAG_INTERVALS)
        self.rolling_windows_seconds = list(rolling_windows_seconds or DEFAULT_ROLLING_WINDOWS_SEC)
        self.n_estimators_grid = list(n_estimators_grid or DEFAULT_N_ESTIMATORS)
        self.max_depth_grid = list(max_depth_grid or DEFAULT_MAX_DEPTH)
        self.learning_rate_grid = list(learning_rate_grid or DEFAULT_LEARNING_RATE)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.random_state = int(random_state)

        self._best_model = None
        self._best_hp: Optional[XGBHyperparams] = None
        self._best_val_mae: float = float("inf")
        self._val_residuals: np.ndarray = np.array([])
        self._feature_columns: List[str] = []
        self._fit_seconds: float = 0.0
        self._shap_explainer = None  # cached TreeExplainer, built on first use

    # ------------------------------------------------------------------ #
    # Forecaster interface                                                #
    # ------------------------------------------------------------------ #
    def fit(self, history: pd.Series) -> "XGBoostForecaster":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(
                "XGBoostForecaster requires xgboost. Install via uv: `uv sync`."
            ) from e

        X_train, y_train, X_val, y_val = self._build_training_matrix(history)

        from sklearn.metrics import mean_absolute_error

        best_mae = float("inf")
        best_model = None
        best_hp = None

        t0 = time.perf_counter()
        for n_est in self.n_estimators_grid:
            for max_d in self.max_depth_grid:
                for lr in self.learning_rate_grid:
                    model = xgb.XGBRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        learning_rate=lr,
                        objective="reg:squarederror",
                        early_stopping_rounds=self.early_stopping_rounds,
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=0,
                    )
                    try:
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                        )
                        preds = model.predict(X_val)
                        val_mae = mean_absolute_error(y_val, preds)
                    except Exception:
                        continue
                    if val_mae < best_mae:
                        best_mae = val_mae
                        best_model = model
                        best_hp = XGBHyperparams(n_est, max_d, lr)

        if best_model is None:
            raise ForecasterFaultError("XGBoost: no candidate fit succeeded")

        self._best_model = best_model
        self._best_hp = best_hp
        self._best_val_mae = float(best_mae)
        val_preds = best_model.predict(X_val)
        self._val_residuals = np.asarray(y_val) - np.asarray(val_preds)
        self._fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        if self._best_model is None:
            raise ForecasterFaultError("XGBoost: predict called before fit")
        if horizon_seconds != self.horizon_seconds:
            raise ValueError(
                f"XGBoost was trained for horizon {self.horizon_seconds}s; "
                f"asked for {horizon_seconds}s. Train a separate model per horizon."
            )

        clean = history.dropna().astype(float)
        if len(clean) < max(self.lag_intervals) + 2:
            raise ForecasterFaultError("not enough history to compute features")

        # Build a one-row feature vector from the tail of the history.
        wide = pd.DataFrame({"timestamp": clean.index, "cpu": clean.values})
        feats = engineer_features(
            wide,
            base_column="cpu",
            sample_interval_seconds=self.sample_interval_seconds,
            lag_intervals=self.lag_intervals,
            rolling_windows_seconds=self.rolling_windows_seconds,
        )
        feats = feats.drop(columns=["timestamp", "cpu"])
        feats = feats[self._feature_columns]   # consistent column order
        last_row = feats.iloc[[-1]]
        if last_row.isna().any(axis=None):
            raise ForecasterFaultError("feature row has NaNs (history too short)")

        try:
            point = float(self._best_model.predict(last_row.values)[0])
        except Exception as e:
            raise ForecasterFaultError(f"XGBoost prediction failed: {e}") from e

        if not math.isfinite(point):
            raise ForecasterFaultError(f"XGBoost produced non-finite point: {point}")

        if len(self._val_residuals) < 4:
            raise ForecasterFaultError("not enough validation residuals for σ")
        sigma = quantile_residual_sigma(self._val_residuals)
        return Forecast(point=point, sigma=sigma)

    def shap_attribution(
        self, history: pd.Series, horizon_seconds: int, top_k: int = 10
    ) -> dict:
        """Return exact SHAP values for the most recent prediction feature row.

        Uses a cached shap.TreeExplainer (built once per model load). Returns
        {} on any error so attribution failures never affect the control loop.
        """
        if self._best_model is None:
            return {}
        try:
            import shap

            clean = history.dropna().astype(float)
            if len(clean) < max(self.lag_intervals) + 2:
                return {}

            wide = pd.DataFrame({"timestamp": clean.index, "cpu": clean.values})
            feats = engineer_features(
                wide,
                base_column="cpu",
                sample_interval_seconds=self.sample_interval_seconds,
                lag_intervals=self.lag_intervals,
                rolling_windows_seconds=self.rolling_windows_seconds,
            )
            feats = feats.drop(columns=["timestamp", "cpu"])
            feats = feats[self._feature_columns]
            last_row = feats.iloc[[-1]]
            if last_row.isna().any(axis=None):
                return {}

            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self._best_model)

            sv = self._shap_explainer.shap_values(last_row.values)
            base_val = float(self._shap_explainer.expected_value)

            pairs = sorted(
                zip(self._feature_columns, sv[0]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_k]

            return {
                "method": "shap_tree",
                "base_value": round(base_val, 6),
                "top_features": {name: round(float(val), 6) for name, val in pairs},
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Artefact persistence                                                #
    # ------------------------------------------------------------------ #
    def save_artifact(self, path: str | Path) -> Path:
        """Persist a trained XGBoost forecaster for live-controller loading."""
        if self._best_model is None or self._best_hp is None:
            raise ForecasterFaultError("XGBoost: cannot save before fit")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        model_path = out / "model.json"
        metadata_path = out / "metadata.json"
        self._best_model.save_model(str(model_path))

        metadata = {
            "forecaster": self.name,
            "horizon_seconds": self.horizon_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "horizon_samples": self.horizon_samples,
            "lag_intervals": self.lag_intervals,
            "rolling_windows_seconds": self.rolling_windows_seconds,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
            "best_hp": asdict(self._best_hp),
            "best_val_mae": self._best_val_mae,
            "val_residuals": self._val_residuals.tolist(),
            "feature_columns": self._feature_columns,
            "fit_seconds": self._fit_seconds,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return out

    @classmethod
    def load_artifact(cls, path: str | Path) -> "XGBoostForecaster":
        """Load a trained XGBoost forecaster saved by `save_artifact`."""
        src = Path(path)
        metadata = json.loads((src / "metadata.json").read_text())

        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(
                "XGBoostForecaster requires xgboost. Install via uv: `uv sync`."
            ) from e

        obj = cls(
            horizon_seconds=int(metadata["horizon_seconds"]),
            sample_interval_seconds=int(metadata["sample_interval_seconds"]),
            lag_intervals=list(metadata["lag_intervals"]),
            rolling_windows_seconds=list(metadata["rolling_windows_seconds"]),
            n_estimators_grid=[int(metadata["best_hp"]["n_estimators"])],
            max_depth_grid=[int(metadata["best_hp"]["max_depth"])],
            learning_rate_grid=[float(metadata["best_hp"]["learning_rate"])],
            early_stopping_rounds=int(metadata["early_stopping_rounds"]),
            random_state=int(metadata["random_state"]),
        )
        model = xgb.XGBRegressor()
        model.load_model(str(src / "model.json"))
        obj._best_model = model
        obj._best_hp = XGBHyperparams(**metadata["best_hp"])
        obj._best_val_mae = float(metadata["best_val_mae"])
        obj._val_residuals = np.asarray(metadata["val_residuals"], dtype=float)
        obj._feature_columns = list(metadata["feature_columns"])
        obj._fit_seconds = float(metadata.get("fit_seconds", 0.0))
        return obj

    # ------------------------------------------------------------------ #
    # Parsimony tiebreaker                                                #
    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        if self._best_hp is None:
            return 0
        # Rough estimate: trees × nodes-per-tree (~2^depth)
        return int(self._best_hp.n_estimators * (2 ** self._best_hp.max_depth))

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _build_training_matrix(
        self, history: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Build (X_train, y_train, X_val, y_val) honouring §3.5 features + 80/20 split."""
        clean = history.dropna().astype(float)
        if len(clean) < max(self.lag_intervals) + self.horizon_samples + 10:
            raise ForecasterFaultError(
                f"XGBoost needs at least "
                f"{max(self.lag_intervals) + self.horizon_samples + 10} clean samples"
            )

        wide = pd.DataFrame({"timestamp": clean.index, "cpu": clean.values})
        feats = engineer_features(
            wide,
            base_column="cpu",
            sample_interval_seconds=self.sample_interval_seconds,
            lag_intervals=self.lag_intervals,
            rolling_windows_seconds=self.rolling_windows_seconds,
        )

        # Target: value at i + horizon_samples
        feats["target"] = feats["cpu"].shift(-self.horizon_samples)
        feats = feats.dropna().reset_index(drop=True)
        if len(feats) < 10:
            raise ForecasterFaultError("XGBoost: not enough rows after feature engineering")

        feature_cols = [c for c in feats.columns
                        if c not in ("timestamp", "cpu", "target")]
        self._feature_columns = feature_cols
        X = feats[feature_cols]
        y = feats["target"].values

        # Time-ordered 80/20 split for early-stopping signal.
        split = int(len(feats) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y[:split], y[split:]
        if len(X_val) < 4:
            raise ForecasterFaultError("XGBoost: validation slice has < 4 rows")
        return X_train, y_train, X_val, y_val
