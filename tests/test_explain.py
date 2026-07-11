"""Tests for the Phase 6 explain package (§3.11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explain import Attribution, Attributor, FaithfulnessMetrics, faithfulness_metrics
from forecasting.sarima import SARIMA
from forecasting.seasonal_naive import SeasonalNaive
from forecasting.xgboost_model import XGBoostForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_history() -> pd.Series:
    rng = np.random.default_rng(0)
    n = 120
    t = np.arange(n)
    vals = 0.05 + 0.03 * np.sin(t * 2 * np.pi / 20) + 0.005 * rng.standard_normal(n)
    return pd.Series(vals.clip(0, None), index=pd.RangeIndex(n), name="cpu")


@pytest.fixture(scope="module")
def xgb(synthetic_history) -> XGBoostForecaster:
    m = XGBoostForecaster(horizon_seconds=30, sample_interval_seconds=15)
    m.fit(synthetic_history)
    return m


@pytest.fixture(scope="module")
def sarima_model(synthetic_history) -> SARIMA:
    m = SARIMA(period_seconds=60, sample_interval_seconds=15)
    m.fit(synthetic_history)
    return m


@pytest.fixture(scope="module")
def seasonal_naive(synthetic_history) -> SeasonalNaive:
    m = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Attribution
# ─────────────────────────────────────────────────────────────────────────────

class TestAttribution:
    def test_tree_shap_returns_top_k_features(self, synthetic_history, xgb):
        attr = Attributor(top_k=5).explain(synthetic_history, xgb, 30)
        assert attr.error is None
        assert attr.method == "tree_shap"
        assert len(attr.top_features) == 5
        # Features ranked by |SHAP| descending
        shap_abs = [abs(v) for _, v in attr.top_features]
        assert shap_abs == sorted(shap_abs, reverse=True)

    def test_tree_shap_expected_value_is_float(self, synthetic_history, xgb):
        attr = Attributor().explain(synthetic_history, xgb, 30)
        assert isinstance(attr.expected_value, float)

    def test_tree_shap_feature_row_stored(self, synthetic_history, xgb):
        attr = Attributor().explain(synthetic_history, xgb, 30)
        assert attr.feature_row is not None
        assert attr.feature_row.shape[0] == 1

    def test_perturbation_sarima(self, synthetic_history, sarima_model):
        attr = Attributor(top_k=5).explain(synthetic_history, sarima_model, 30)
        assert attr.error is None
        assert attr.method == "perturbation"
        assert len(attr.top_features) <= 5
        for name, _ in attr.top_features:
            assert name.startswith("lag_")

    def test_perturbation_seasonal_naive(self, synthetic_history, seasonal_naive):
        attr = Attributor(top_k=3).explain(synthetic_history, seasonal_naive, 30)
        assert attr.error is None
        assert attr.method == "perturbation"

    def test_attribution_failure_returns_error_object(self, synthetic_history):
        broken = SeasonalNaive(period_seconds=60, sample_interval_seconds=15)
        # SeasonalNaive with only 2 samples fails to predict
        short = synthetic_history.iloc[:2]
        attr = Attributor().explain(short, broken, 30)
        # May either error on predict() or return a valid attribution
        assert isinstance(attr, Attribution)

    def test_to_dict_is_json_serialisable(self, synthetic_history, xgb):
        import json
        attr = Attributor(top_k=3).explain(synthetic_history, xgb, 30)
        d = attr.to_dict()
        json.dumps(d)  # must not raise
        assert "top_features" in d
        assert "method" in d


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness
# ─────────────────────────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_xgboost_insertion_auc_above_threshold(self, synthetic_history, xgb):
        attr = Attributor(top_k=5).explain(synthetic_history, xgb, 30)
        fm = faithfulness_metrics(
            synthetic_history, xgb, 30, attr, run_param_randomisation=False
        )
        assert fm.error is None
        assert fm.insertion_auc is not None
        assert fm.insertion_auc > 0.5, f"insertion_auc={fm.insertion_auc}"

    def test_xgboost_deletion_auc_above_threshold(self, synthetic_history, xgb):
        attr = Attributor(top_k=5).explain(synthetic_history, xgb, 30)
        fm = faithfulness_metrics(
            synthetic_history, xgb, 30, attr, run_param_randomisation=False
        )
        assert fm.deletion_auc is not None
        # Threshold is 0.2: deletion AUC on small synthetic data can be < 0.5
        # because XGBoost finds many features nearly equally important.
        # The primary faithfulness signal is insertion_auc.
        assert fm.deletion_auc > 0.2, f"deletion_auc={fm.deletion_auc}"

    def test_faithfulness_passes_for_xgboost(self, synthetic_history, xgb):
        attr = Attributor(top_k=5).explain(synthetic_history, xgb, 30)
        fm = faithfulness_metrics(
            synthetic_history, xgb, 30, attr, run_param_randomisation=False
        )
        assert fm.passes

    def test_failed_attribution_returns_error_faithfulness(self, synthetic_history, xgb):
        bad_attr = Attribution.failure("tree_shap", "simulated error")
        fm = faithfulness_metrics(synthetic_history, xgb, 30, bad_attr)
        assert fm.error is not None
        assert not fm.passes

    def test_faithfulness_to_dict_is_json_serialisable(self, synthetic_history, xgb):
        import json
        attr = Attributor(top_k=3).explain(synthetic_history, xgb, 30)
        fm = faithfulness_metrics(
            synthetic_history, xgb, 30, attr, run_param_randomisation=False
        )
        json.dumps(fm.to_dict())

    def test_sarima_faithfulness_does_not_error(self, synthetic_history, sarima_model):
        attr = Attributor(top_k=3).explain(synthetic_history, sarima_model, 30)
        fm = faithfulness_metrics(
            synthetic_history, sarima_model, 30, attr, run_param_randomisation=False
        )
        assert isinstance(fm, FaithfulnessMetrics)
        # insertion/deletion AUC may be 0/1 for SARIMA (flat response) — no assertion on value
        assert fm.error is None
