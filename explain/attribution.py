"""SHAP attribution dispatcher (§3.8, §3.11).

Maps forecaster type → attribution method:
    XGBoostForecaster  → tree_shap.explain()
    LSTMForecaster     → timeshap_backend.explain()
    statistical        → perturbation.explain()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (avoid heavy deps at module load)
    from forecasting.lstm_model import LSTMForecaster
    from forecasting.xgboost_model import XGBoostForecaster

import numpy as np
import pandas as pd

from forecasting.base import Forecaster


@dataclass
class Attribution:
    """Per-decision SHAP attribution vector.

    Attributes
    ----------
    top_features:
        (feature_name, shap_value) pairs ranked by |shap_value|, descending.
        At most `top_k` entries.
    method:
        Name of the attribution method used (e.g. "tree_shap", "timeshap",
        "perturbation").
    expected_value:
        The baseline prediction (SHAP φ₀) around which attributions sum.
        None when the method does not provide a natural baseline.
    raw_shap:
        Full dict {feature_name: shap_value} before top-k truncation.
    feature_row:
        The one-row DataFrame of features used for prediction (XGBoost only).
        Used by faithfulness.py for feature-level insertion/deletion AUC.
    error:
        Non-None if attribution failed; top_features will be empty.
    """

    top_features: List[Tuple[str, float]] = field(default_factory=list)
    method: str = "unknown"
    expected_value: Optional[float] = None
    raw_shap: Dict[str, float] = field(default_factory=dict)
    feature_row: Optional[Any] = field(default=None, repr=False)  # pd.DataFrame, not JSON-serialised
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "expected_value": self.expected_value,
            "top_features": [{"name": n, "shap": v} for n, v in self.top_features],
            "error": self.error,
        }

    @classmethod
    def failure(cls, method: str, error: str) -> "Attribution":
        return cls(method=method, error=error)


class Attributor:
    """Dispatches per-decision SHAP attribution to the correct backend.

    Parameters
    ----------
    top_k:
        Number of top features to include in `Attribution.top_features`.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def explain(
        self,
        history: pd.Series,
        forecaster: Forecaster,
        horizon_seconds: int,
    ) -> Attribution:
        """Compute SHAP attribution for a single prediction.

        Parameters
        ----------
        history:
            The same pandas Series that would be passed to forecaster.predict().
        forecaster:
            A fitted Forecaster instance.
        horizon_seconds:
            Forecast horizon in seconds.

        Returns
        -------
        Attribution with top_k features and SHAP values.
        """
        # Import lazily so tests that don't need SHAP don't pay the import cost.
        from forecasting.lstm_model import LSTMForecaster
        from forecasting.xgboost_model import XGBoostForecaster

        if isinstance(forecaster, XGBoostForecaster):
            return self._tree_shap(history, forecaster, horizon_seconds)
        if isinstance(forecaster, LSTMForecaster):
            return self._timeshap(history, forecaster, horizon_seconds)
        return self._perturbation(history, forecaster, horizon_seconds)

    # ------------------------------------------------------------------ #
    # TreeSHAP for XGBoost                                                #
    # ------------------------------------------------------------------ #

    def _tree_shap(
        self,
        history: pd.Series,
        forecaster: "XGBoostForecaster",
        horizon_seconds: int,
    ) -> Attribution:
        try:
            import shap

            from data.features.engineer import engineer_features

            clean = history.dropna().astype(float)
            wide = pd.DataFrame({"timestamp": clean.index, "cpu": clean.values})
            feats = engineer_features(
                wide,
                base_column="cpu",
                sample_interval_seconds=forecaster.sample_interval_seconds,
                lag_intervals=forecaster.lag_intervals,
                rolling_windows_seconds=forecaster.rolling_windows_seconds,
            )
            feats = feats.drop(columns=["timestamp", "cpu"])
            feats = feats[forecaster._feature_columns]
            last_row = feats.iloc[[-1]]

            explainer = shap.TreeExplainer(forecaster._best_model)
            sv = explainer.shap_values(last_row)
            if isinstance(sv, list):
                sv = sv[0]
            shap_vals = sv[0].tolist()
            names = forecaster._feature_columns
            raw = dict(zip(names, shap_vals))
            top = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[: self.top_k]
            return Attribution(
                top_features=top,
                method="tree_shap",
                expected_value=float(explainer.expected_value)
                if np.isscalar(explainer.expected_value)
                else float(explainer.expected_value[0]),
                raw_shap=raw,
                feature_row=last_row,
            )
        except Exception as exc:
            return Attribution.failure("tree_shap", str(exc))

    # ------------------------------------------------------------------ #
    # TimeSHAP for LSTM                                                   #
    # ------------------------------------------------------------------ #

    def _timeshap(
        self,
        history: pd.Series,
        forecaster: "LSTMForecaster",
        horizon_seconds: int,
    ) -> Attribution:
        try:
            import torch
            from timeshap.explainer import local_report

            clean = history.dropna().astype(float).values
            # TimeSHAP expects a model function f(x) → scalar and a 3-D numpy array.
            # LSTMForecaster stores the fitted network as _best_model and the
            # sequence length inside _best_hp (2026-07-05 code review: the
            # previous _model/seq_len attributes never existed, so LSTM
            # attribution failed silently on every call).
            if forecaster._best_model is None or forecaster._best_hp is None:
                return Attribution.failure("timeshap", "LSTM forecaster not fitted")
            model = forecaster._best_model
            seq_len = int(forecaster._best_hp.seq_len)
            if len(clean) < seq_len:
                return Attribution.failure("timeshap", "not enough history for TimeSHAP")

            seq = clean[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)

            def model_fn(x: np.ndarray) -> np.ndarray:
                t = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    out = model(t)
                return out.numpy().reshape(-1)

            # TimeSHAP local_report returns per-timestep Shapley values.
            report = local_report(
                f=model_fn,
                data=seq,
                baseline=np.zeros_like(seq),
                feature_names=[f"t-{seq_len - i}" for i in range(seq_len)],
                plot=False,
            )
            sv_dict: Dict[str, float] = {}
            if "shapley_values" in report:
                for i, v in enumerate(report["shapley_values"][0]):
                    sv_dict[f"t-{seq_len - i}"] = float(v)
            top = sorted(sv_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[: self.top_k]
            return Attribution(top_features=top, method="timeshap", raw_shap=sv_dict)
        except Exception as exc:
            return Attribution.failure("timeshap", str(exc))

    # ------------------------------------------------------------------ #
    # Perturbation SHAP for statistical models                            #
    # ------------------------------------------------------------------ #

    def _perturbation(
        self,
        history: pd.Series,
        forecaster: Forecaster,
        horizon_seconds: int,
    ) -> Attribution:
        """Remove-one-feature ablation over lag positions.

        For statistical models that do not expose internal features, we
        approximate Shapley values by replacing each lag position with the
        series mean and measuring the change in the point forecast.
        """
        try:
            clean = history.dropna().astype(float)
            if len(clean) < 2:
                return Attribution.failure("perturbation", "insufficient history")

            baseline_forecast = forecaster.predict(history, horizon_seconds).point
            mean_val = float(clean.mean())

            # Ablate each of the last min(len, 10) lag positions.
            n_lags = min(len(clean), 10)
            raw: Dict[str, float] = {}
            for i in range(n_lags):
                ablated = clean.copy()
                ablated.iloc[-(i + 1)] = mean_val
                try:
                    ablated_forecast = forecaster.predict(
                        ablated.rename(history.name), horizon_seconds
                    ).point
                    raw[f"lag_{i+1}"] = baseline_forecast - ablated_forecast
                except Exception:
                    raw[f"lag_{i+1}"] = 0.0

            top = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[: self.top_k]
            return Attribution(
                top_features=top,
                method="perturbation",
                expected_value=mean_val,
                raw_shap=raw,
            )
        except Exception as exc:
            return Attribution.failure("perturbation", str(exc))
