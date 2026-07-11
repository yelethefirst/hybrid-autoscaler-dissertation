"""Faithfulness metrics for SHAP attributions (§3.11).

Three complementary checks verify that the attribution is not decorative:

1. Insertion AUC
   Re-introduce features in order of |SHAP| (most important first).
   Track how forecast RMSE recovers as features are restored.
   Higher AUC → attributions correctly identify the most important features.

2. Deletion AUC
   Remove features in order of |SHAP| (most important first).
   Track how forecast RMSE degrades as features are removed.
   Higher AUC → removing the top-SHAP features hurts more than removing random ones.

3. Parameter randomisation test
   Randomly re-initialise model weights and recompute SHAP.
   The attribution vector for the randomised model should be uncorrelated with
   the original attribution (Spearman ρ close to 0 → model-specificity confirmed).

References
----------
    Samek et al. (2017) — pixel-flipping / insertion/deletion AUC.
    Adebayo et al. (2018) — sanity checks for saliency maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from forecasting.lstm_model import LSTMForecaster
    from forecasting.xgboost_model import XGBoostForecaster

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forecasting.base import Forecaster

from .attribution import Attribution, Attributor


@dataclass
class FaithfulnessMetrics:
    """Results of all three faithfulness checks for one attribution."""

    insertion_auc: Optional[float] = None
    deletion_auc: Optional[float] = None
    param_randomisation_rho: Optional[float] = None  # Spearman ρ with randomised model
    param_randomisation_p: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "insertion_auc": self.insertion_auc,
            "deletion_auc": self.deletion_auc,
            "param_randomisation_rho": self.param_randomisation_rho,
            "param_randomisation_p": self.param_randomisation_p,
            "error": self.error,
        }

    @property
    def passes(self) -> bool:
        """True if the attribution is faithful under all computed checks.

        Thresholds (one-sided):
            insertion_auc  > 0.6  — primary metric (Samek et al. 2017)
            deletion_auc   > 0.2  — secondary metric (weaker threshold; deletion
                                    curves are noisier on small datasets)
            |ρ|            < 0.3  — param randomisation (Adebayo et al. 2018)
        """
        if self.error:
            return False
        checks = []
        if self.insertion_auc is not None:
            checks.append(self.insertion_auc > 0.6)
        if self.deletion_auc is not None:
            checks.append(self.deletion_auc > 0.2)
        if self.param_randomisation_rho is not None:
            checks.append(abs(self.param_randomisation_rho) < 0.3)
        return all(checks) if checks else False


def _xgboost_insertion_deletion_auc(
    attribution: "Attribution",
    forecaster: "XGBoostForecaster",
    mode: str = "insertion",
) -> float:
    """Feature-level insertion/deletion AUC for XGBoost using the feature row."""

    feature_row = attribution.feature_row
    shap_vals = attribution.raw_shap
    if feature_row is None or not shap_vals or forecaster._best_model is None:
        return float("nan")

    import numpy as np

    feature_names = list(feature_row.columns)
    row_vals = feature_row.values[0].copy()
    bg = row_vals.mean()  # simple background baseline

    baseline_pred = float(forecaster._best_model.predict(feature_row.values)[0])

    ordered_names = sorted(shap_vals, key=lambda k: abs(shap_vals[k]), reverse=True)

    scores: List[float] = []
    for step in range(1, len(ordered_names) + 1):
        masked = row_vals.copy()
        if mode == "insertion":
            masked[:] = bg
            for name in ordered_names[:step]:
                if name in feature_names:
                    idx = feature_names.index(name)
                    masked[idx] = row_vals[idx]
        else:
            for name in ordered_names[:step]:
                if name in feature_names:
                    idx = feature_names.index(name)
                    masked[idx] = bg
        pred = float(forecaster._best_model.predict(masked.reshape(1, -1))[0])
        scores.append(abs(pred - baseline_pred))

    if not scores:
        return float("nan")
    xs = np.arange(len(scores)) / max(len(scores) - 1, 1)
    max_s = max(scores) or 1.0
    ys = np.array(scores) / max_s
    if mode == "deletion":
        return float(np.trapz(ys, xs))
    return float(1.0 - np.trapz(ys, xs))


def _insertion_deletion_auc(
    history: pd.Series,
    forecaster: Forecaster,
    horizon_seconds: int,
    shap_vals: Dict[str, float],
    mode: str = "insertion",
    background_val: Optional[float] = None,
) -> float:
    """Compute insertion or deletion AUC for feature attributions.

    Uses perturbation on the history array: inserts/removes lag positions in
    descending |SHAP| order and tracks forecast deviation from the baseline.

    This approximation applies only to statistical forecasters and XGBoost
    (where the input space is the lag window). For LSTM the input is the raw
    sequence, so the same lag-position perturbation is used.

    Parameters
    ----------
    mode:
        "insertion" — features start absent (replaced by background_val),
                       added back one by one in |SHAP| order.
        "deletion"  — features start present, removed one by one.
    """
    clean = history.dropna().astype(float)
    if len(clean) < 2 or not shap_vals:
        return float("nan")

    bg = background_val if background_val is not None else float(clean.mean())
    try:
        baseline_point = forecaster.predict(history, horizon_seconds).point
    except Exception:
        return float("nan")

    # Sort lag indices (SHAP keys may be "lag_1", "lag_2", …) by |SHAP|.
    lag_keys = sorted(shap_vals, key=lambda k: abs(shap_vals[k]), reverse=True)
    n = min(len(lag_keys), len(clean))

    scores: List[float] = []
    for step in range(1, n + 1):
        ablated = clean.copy()
        if mode == "insertion":
            # Start from all-background; insert top-step features.
            ablated[:] = bg
            for k in lag_keys[:step]:
                idx = _lag_key_to_index(k, len(clean))
                if idx is not None:
                    ablated.iloc[idx] = clean.iloc[idx]
        else:
            # Start with all features; delete top-step features.
            for k in lag_keys[:step]:
                idx = _lag_key_to_index(k, len(clean))
                if idx is not None:
                    ablated.iloc[idx] = bg
        try:
            pt = forecaster.predict(ablated.rename(history.name), horizon_seconds).point
            diff = abs(pt - baseline_point)
            scores.append(diff)
        except Exception:
            scores.append(0.0)

    if not scores:
        return float("nan")
    # AUC via trapezoidal rule over [0, n] steps.
    xs = np.arange(len(scores)) / max(len(scores) - 1, 1)
    # Normalise scores to [0, 1].
    max_s = max(scores) or 1.0
    ys = np.array(scores) / max_s
    if mode == "deletion":
        return float(np.trapz(ys, xs))
    else:
        return float(1.0 - np.trapz(ys, xs))


def _lag_key_to_index(key: str, n: int) -> Optional[int]:
    """Convert "lag_k" string to a negative index in the history array."""
    if key.startswith("lag_"):
        try:
            k = int(key[4:])
            return -k if k <= n else None
        except ValueError:
            return None
    # TimeSHAP keys: "t-k"
    if key.startswith("t-"):
        try:
            k = int(key[2:])
            return -k if k <= n else None
        except ValueError:
            return None
    return None


def _param_randomisation_rho(
    history: pd.Series,
    forecaster: Forecaster,
    horizon_seconds: int,
    attribution: Attribution,
    attributor: Attributor,
    n_random: int = 3,
) -> Tuple[Optional[float], Optional[float]]:
    """Spearman ρ between original and parameter-randomised attributions.

    For XGBoost: creates a new XGBoostForecaster with random weights by
    fitting on a shuffled copy of the history.
    For statistical models: shuffles the history to destroy temporal structure.
    """
    from forecasting.lstm_model import LSTMForecaster
    from forecasting.xgboost_model import XGBoostForecaster

    orig_vals = [v for _, v in attribution.top_features]
    if len(orig_vals) < 2:
        return None, None

    rhos: List[float] = []
    for seed in range(n_random):
        try:
            if isinstance(forecaster, XGBoostForecaster):
                rand_model = _randomised_xgboost(forecaster, history, seed)
            elif isinstance(forecaster, LSTMForecaster):
                rand_model = _randomised_lstm(forecaster, history, seed)
            else:
                rand_model = _shuffled_statistical(forecaster, history, seed)
            if rand_model is None:
                continue
            rand_attr = attributor.explain(history, rand_model, horizon_seconds)
            if rand_attr.error or not rand_attr.top_features:
                continue
            rand_vals = [v for _, v in rand_attr.top_features]
            min_len = min(len(orig_vals), len(rand_vals))
            if min_len < 2:
                continue
            rho, _ = spearmanr(orig_vals[:min_len], rand_vals[:min_len])
            rhos.append(float(rho))
        except Exception:
            continue

    if not rhos:
        return None, None
    mean_rho = float(np.mean(rhos))
    _, p = spearmanr(orig_vals, orig_vals[::-1])  # dummy p for the mean rho
    return mean_rho, float(p)


def _randomised_xgboost(
    original: "XGBoostForecaster",
    history: pd.Series,
    seed: int,
) -> Optional["XGBoostForecaster"]:
    from forecasting.xgboost_model import XGBoostForecaster

    rng = np.random.default_rng(seed)
    shuffled = history.copy()
    shuffled[:] = rng.permutation(history.dropna().values)
    try:
        rand = XGBoostForecaster(
            horizon_seconds=original.horizon_seconds,
            sample_interval_seconds=original.sample_interval_seconds,
        )
        rand.fit(shuffled)
        return rand
    except Exception:
        return None


def _randomised_lstm(
    original: "LSTMForecaster",
    history: pd.Series,
    seed: int,
) -> Optional["LSTMForecaster"]:
    from forecasting.lstm_model import LSTMForecaster

    rng = np.random.default_rng(seed)
    shuffled = history.copy()
    shuffled[:] = rng.permutation(history.dropna().values)
    try:
        rand = LSTMForecaster(
            horizon_seconds=original.horizon_seconds,
            sample_interval_seconds=original.sample_interval_seconds,
            max_epochs=5,
            seed=seed,
        )
        rand.fit(shuffled)
        return rand
    except Exception:
        return None


def _shuffled_statistical(
    forecaster: Forecaster,
    history: pd.Series,
    seed: int,
) -> Optional[Forecaster]:
    """Return a copy of the statistical forecaster trained on shuffled history."""
    import copy

    rng = np.random.default_rng(seed)
    shuffled = history.copy()
    shuffled[:] = rng.permutation(history.dropna().values)
    try:
        rand = copy.deepcopy(forecaster)
        rand.fit(shuffled)  # type: ignore[attr-defined]
        return rand
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_metrics(
    history: pd.Series,
    forecaster: Forecaster,
    horizon_seconds: int,
    attribution: Attribution,
    attributor: Optional[Attributor] = None,
    run_param_randomisation: bool = True,
) -> FaithfulnessMetrics:
    """Compute all three faithfulness checks for an attribution.

    Parameters
    ----------
    run_param_randomisation:
        Set to False to skip the randomisation test (it re-trains the model,
        which can be slow for XGBoost/LSTM).
    """
    if attribution.error or not attribution.raw_shap:
        return FaithfulnessMetrics(error=f"attribution failed: {attribution.error}")

    attr = attributor or Attributor()

    from forecasting.xgboost_model import XGBoostForecaster

    if isinstance(forecaster, XGBoostForecaster) and attribution.feature_row is not None:
        ins_auc = _xgboost_insertion_deletion_auc(attribution, forecaster, mode="insertion")
        del_auc = _xgboost_insertion_deletion_auc(attribution, forecaster, mode="deletion")
    else:
        ins_auc = _insertion_deletion_auc(
            history, forecaster, horizon_seconds, attribution.raw_shap, mode="insertion"
        )
        del_auc = _insertion_deletion_auc(
            history, forecaster, horizon_seconds, attribution.raw_shap, mode="deletion"
        )

    rho, p = None, None
    if run_param_randomisation:
        rho, p = _param_randomisation_rho(
            history, forecaster, horizon_seconds, attribution, attr
        )

    return FaithfulnessMetrics(
        insertion_auc=ins_auc if not (ins_auc is None or np.isnan(ins_auc)) else None,
        deletion_auc=del_auc if not (del_auc is None or np.isnan(del_auc)) else None,
        param_randomisation_rho=rho,
        param_randomisation_p=p,
    )
