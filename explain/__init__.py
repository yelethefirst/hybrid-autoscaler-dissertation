"""Phase 6 explainability package (§3.8, §3.11).

Provides per-decision SHAP attribution vectors for each forecaster family and
faithfulness metrics that verify the attributions are not decorative.

Public interface
----------------
    from explain import Attributor, Attribution, FaithfulnessMetrics
    attr = Attributor().explain(history, forecaster, horizon_seconds)
    # attr.top_features: list[tuple[str, float]]  (name, shap_value)
    # attr.method: str

Attributor dispatches to the correct backend:
    XGBoostForecaster  → TreeSHAP (exact, via shap.TreeExplainer)
    LSTMForecaster     → TimeSHAP  (via timeshap library)
    SARIMA / HW / SNaive → Perturbation SHAP (remove-one-feature ablation)
"""

from .attribution import Attribution, Attributor
from .faithfulness import FaithfulnessMetrics, faithfulness_metrics

__all__ = ["Attribution", "Attributor", "FaithfulnessMetrics", "faithfulness_metrics"]
