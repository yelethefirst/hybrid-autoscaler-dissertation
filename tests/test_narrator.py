"""Tests for Phase 7 LLM narrative generation (§3.12).

These tests cover the offline path only (no API key required).
The online path (LLM call) is covered by the integration test notebook and
the n=30 evaluation harness in experiments/evaluate_narratives.py.
"""

from __future__ import annotations

import json

import pytest

from explain.attribution import Attribution
from narrate import Narrator, NarrativeResult
from narrate.factscore import ClaimVerdict, FActScoreResult, generate_likert_workbook


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def nominal_decision() -> dict:
    return {
        "service": "frontend",
        "namespace": "default",
        "timestamp": "2026-07-02T16:30:00Z",
        "state": "NOMINAL",
        "current_replicas": 1,
        "new_replicas": 2,
        "recommended_replicas": 2,
        "rate_limited": False,
        "fallback_engaged": False,
        "forecaster_fault_reason": None,
        "horizon_seconds": 30,
        "forecast_point": 0.1021,
        "forecast_sigma": 0.0295,
        "forecaster_name": "sarima",
        "observed_metric": 0.072,
    }


@pytest.fixture
def fallback_decision() -> dict:
    return {
        "service": "frontend",
        "namespace": "default",
        "timestamp": "2026-07-02T16:31:00Z",
        "state": "FALLBACK_FORECASTER_FAULT",
        "current_replicas": 1,
        "new_replicas": 1,
        "recommended_replicas": 1,
        "rate_limited": False,
        "fallback_engaged": True,
        "forecaster_fault_reason": "ForecasterFaultError: SARIMA needs ≥ 12 clean samples (have 5)",
        "horizon_seconds": 30,
        "forecast_point": None,
        "forecast_sigma": None,
        "forecaster_name": "sarima",
        "observed_metric": 0.003,
    }


@pytest.fixture
def attribution() -> Attribution:
    return Attribution(
        top_features=[
            ("cpu_lag1", 0.045),
            ("cpu_rmean60s", 0.021),
            ("cpu_lag5", -0.012),
        ],
        method="perturbation",
        expected_value=0.05,
        raw_shap={"cpu_lag1": 0.045, "cpu_rmean60s": 0.021, "cpu_lag5": -0.012},
    )


@pytest.fixture
def fallback_attribution() -> Attribution:
    return Attribution(
        top_features=[],
        method="perturbation",
        error="forecaster faulted — no attribution",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Narrator offline mode
# ─────────────────────────────────────────────────────────────────────────────

class TestNarratorOffline:
    def test_offline_returns_narrative_result(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert isinstance(result, NarrativeResult)
        assert result.error is None

    def test_context_contains_forecast_point(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert "0.1021" in result.context

    def test_context_contains_state(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert "NOMINAL" in result.context

    def test_context_contains_replicas(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert "1 to 2" in result.context

    def test_context_contains_top_shap_feature(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert "cpu_lag1" in result.context
        assert "0.0450" in result.context

    def test_fallback_context_mentions_fallback(self, fallback_decision, fallback_attribution):
        result = Narrator(client=None).narrate_offline(fallback_decision, fallback_attribution)
        assert "FALLBACK_FORECASTER_FAULT" in result.context
        assert "Fallback active: True" in result.context

    def test_fallback_reason_included(self, fallback_decision, fallback_attribution):
        result = Narrator(client=None).narrate_offline(fallback_decision, fallback_attribution)
        assert "ForecasterFaultError" in result.context

    def test_to_dict_is_json_serialisable(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        d = result.to_dict()
        json.dumps(d)
        assert "narrative" in d
        assert "model" in d

    def test_model_is_offline(self, nominal_decision, attribution):
        result = Narrator(client=None).narrate_offline(nominal_decision, attribution)
        assert result.model == "offline"

    def test_narrate_raises_without_client(self, nominal_decision, attribution):
        # The online path requires a real client — should not silently succeed.
        result = Narrator(client=None).narrate(nominal_decision, attribution)
        assert result.error is not None


# ─────────────────────────────────────────────────────────────────────────────
# FActScore (offline — no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestFActScoreOffline:
    def test_score_empty_narrative_returns_error(self):
        from narrate.factscore import FActScoreEvaluator

        evaluator = FActScoreEvaluator(client=None)
        result = evaluator.score("", "some source")
        assert result.error is not None

    def test_factscore_result_to_dict_json_serialisable(self):
        result = FActScoreResult(
            narrative="The system scaled up.",
            claims=[
                ClaimVerdict("The system scaled up.", "supported"),
                ClaimVerdict("The CPU was 90%.", "not_supported"),
            ],
            factscore=0.5,
        )
        d = result.to_dict()
        json.dumps(d)
        assert d["factscore"] == 0.5
        assert d["n_claims"] == 2
        assert d["n_supported"] == 1

    def test_generate_likert_workbook(self, tmp_path):
        narratives = [
            {"trial_id": "burst-hybrid-001", "narrative": "The system scaled up to 2 replicas."},
            {"trial_id": "burst-hpa-001", "narrative": "HPA added 1 replica due to CPU load."},
        ]
        out = tmp_path / "workbook.csv"
        generate_likert_workbook(narratives, out)
        assert out.is_file()
        content = out.read_text()
        assert "row_id" in content
        assert "accuracy" in content
        assert "The system scaled up" in content
