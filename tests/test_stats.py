"""Tests for Phase 8 statistical analysis (§3.13).

Uses synthetic trial data to verify that hypothesis tests, BCa bootstrap,
and Holm-Bonferroni correction behave correctly before real experiment data
is available.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.stats import (
    HypothesisResult,
    _bca_ci,
    cohens_d_z,
    holm_bonferroni,
    render_table,
    run_h0_latency,
    run_h2_replica_seconds,
    run_h3_oscillation,
    run_h4_factscore,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_results(rng) -> pd.DataFrame:
    """10-pair A/B synthetic data: hybrid p95 is ~20ms lower than hpa."""
    n = 10
    workloads = ["burst", "ramp", "periodic", "trace_replay"]
    rows = []
    for wl in workloads:
        for i in range(n):
            # HPA baseline
            hpa_p95 = 200 + rng.standard_normal() * 20
            # Hybrid: 20 ms lower with some noise
            hybrid_p95 = hpa_p95 - 20 + rng.standard_normal() * 15
            rows.append({
                "trial_id": f"{wl}-hpa-{i:03d}",
                "autoscaler": "hpa",
                "workload": wl,
                "seed": i,
                "p95_latency_ms": max(0, hpa_p95),
                "replica_seconds": 1200 + rng.standard_normal() * 100,
                "oscillation_count": max(0, int(rng.integers(2, 6))),
            })
            rows.append({
                "trial_id": f"{wl}-hybrid-{i:03d}",
                "autoscaler": "hybrid",
                "workload": wl,
                "seed": i,
                "p95_latency_ms": max(0, hybrid_p95),
                "replica_seconds": 1000 + rng.standard_normal() * 80,
                "oscillation_count": max(0, int(rng.integers(0, 3))),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BCa bootstrap
# ─────────────────────────────────────────────────────────────────────────────

class TestBCaBootstrap:
    def test_ci_contains_true_mean(self, rng):
        data = rng.standard_normal(50) + 5.0
        lo, hi = _bca_ci(data, np.mean, n_bootstrap=5000, seed=0)
        assert lo < 5.0 < hi, f"CI [{lo:.3f}, {hi:.3f}] should contain 5.0"

    def test_ci_is_ordered(self, rng):
        data = rng.standard_normal(30)
        lo, hi = _bca_ci(data, np.mean, n_bootstrap=2000, seed=1)
        assert lo < hi

    def test_ci_width_decreases_with_n(self, rng):
        d_small = rng.standard_normal(20)
        d_large = rng.standard_normal(200)
        lo_s, hi_s = _bca_ci(d_small, np.mean, n_bootstrap=2000, seed=2)
        lo_l, hi_l = _bca_ci(d_large, np.mean, n_bootstrap=2000, seed=2)
        assert (hi_l - lo_l) < (hi_s - lo_s)


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's d_z
# ─────────────────────────────────────────────────────────────────────────────

class TestCohensD:
    def test_zero_difference_gives_zero(self):
        diff = np.zeros(10)
        assert cohens_d_z(diff) == pytest.approx(0.0)

    def test_known_d(self):
        # Identical values give SD=0 → guarded by epsilon, so use varied data.
        rng = np.random.default_rng(5)
        base = rng.standard_normal(30) + 2.0
        d = cohens_d_z(base)
        assert d > 0.5   # positive direction


# ─────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni
# ─────────────────────────────────────────────────────────────────────────────

class TestHolmBonferroni:
    def test_all_significant(self):
        p_vals = [0.001, 0.002, 0.003]
        results = holm_bonferroni(p_vals, alpha=0.05)
        assert all(sig for _, _, sig in results)

    def test_none_significant(self):
        p_vals = [0.4, 0.5, 0.6]
        results = holm_bonferroni(p_vals, alpha=0.05)
        assert not any(sig for _, _, sig in results)

    def test_returns_same_length(self):
        p_vals = [0.01, 0.04, 0.08, 0.20]
        results = holm_bonferroni(p_vals, alpha=0.05)
        assert len(results) == 4

    def test_adjusted_p_is_bounded(self):
        p_vals = [0.001, 0.01, 0.5]
        results = holm_bonferroni(p_vals, alpha=0.05)
        for adj_p, _, _ in results:
            assert 0.0 <= adj_p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# H0: latency
# ─────────────────────────────────────────────────────────────────────────────

class TestH0Latency:
    def test_returns_one_result_per_workload(self, synthetic_results):
        results = run_h0_latency(synthetic_results)
        assert len(results) == 4

    def test_all_results_are_hypothesis_results(self, synthetic_results):
        for r in run_h0_latency(synthetic_results):
            assert isinstance(r, HypothesisResult)

    def test_mean_diff_is_hybrid_minus_hpa(self, synthetic_results):
        for r in run_h0_latency(synthetic_results):
            # Hybrid is ~20ms lower → mean_diff should be negative
            assert r.mean_diff < 0, f"{r.workload}: mean_diff={r.mean_diff}"

    def test_significant_when_large_effect(self, rng):
        # Construct data with a very large effect
        n = 20
        rows = []
        for wl in ["burst"]:
            for i in range(n):
                hpa = 300 + rng.standard_normal() * 5
                hybrid = 100 + rng.standard_normal() * 5
                rows.append({"autoscaler": "hpa", "workload": wl, "p95_latency_ms": hpa,
                              "replica_seconds": 1200, "oscillation_count": 3})
                rows.append({"autoscaler": "hybrid", "workload": wl, "p95_latency_ms": hybrid,
                              "replica_seconds": 1000, "oscillation_count": 1})
        df = pd.DataFrame(rows)
        results = run_h0_latency(df)
        assert results[0].significant

    def test_not_significant_when_no_effect(self, rng):
        n = 10
        rows = []
        for wl in ["burst"]:
            for i in range(n):
                # Independent draws from the same distribution — expected mean diff ≈ 0
                hpa_val = 200 + rng.standard_normal() * 20
                hybrid_val = 200 + rng.standard_normal() * 20
                rows.append({"autoscaler": "hpa", "workload": wl, "p95_latency_ms": hpa_val,
                              "replica_seconds": 1200, "oscillation_count": 3})
                rows.append({"autoscaler": "hybrid", "workload": wl, "p95_latency_ms": hybrid_val,
                              "replica_seconds": 1200, "oscillation_count": 3})
        df = pd.DataFrame(rows)
        results = run_h0_latency(df)
        # With n=10 pairs from the same distribution, should rarely be significant
        # (This is a probabilistic test; seed 0 + rng fixture gives a stable outcome)
        assert not results[0].significant

    def test_to_dict_is_json_serialisable(self, synthetic_results):
        for r in run_h0_latency(synthetic_results):
            json.dumps(r.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# H2: replica-seconds
# ─────────────────────────────────────────────────────────────────────────────

class TestH2ReplicaSeconds:
    def test_burst_and_ramp_only(self, synthetic_results):
        results = run_h2_replica_seconds(synthetic_results)
        workloads = {r.workload for r in results}
        assert workloads == {"burst", "ramp"}

    def test_hybrid_uses_fewer_replica_seconds(self, synthetic_results):
        for r in run_h2_replica_seconds(synthetic_results):
            # Synthetic data: hybrid ~1000, hpa ~1200
            assert r.mean_diff < 0


# ─────────────────────────────────────────────────────────────────────────────
# H3: oscillation
# ─────────────────────────────────────────────────────────────────────────────

class TestH3Oscillation:
    def test_returns_hypothesis_result(self, synthetic_results):
        r = run_h3_oscillation(synthetic_results)
        assert isinstance(r, HypothesisResult)
        assert r.hypothesis == "H3"

    def test_mean_diff_is_negative(self, synthetic_results):
        r = run_h3_oscillation(synthetic_results)
        # Hybrid: integers 0-2; HPA: integers 2-5 → hybrid fewer
        assert r.mean_diff < 0


# ─────────────────────────────────────────────────────────────────────────────
# H4: FActScore
# ─────────────────────────────────────────────────────────────────────────────

class TestH4FActScore:
    def test_passes_when_scores_high(self):
        scores = [0.95] * 30
        r = run_h4_factscore(scores, mu0=0.8)
        assert r.significant

    def test_fails_when_scores_low(self):
        scores = [0.50] * 30
        r = run_h4_factscore(scores, mu0=0.8)
        assert not r.significant

    def test_mean_is_correct(self):
        scores = [0.9] * 20 + [0.7] * 10
        r = run_h4_factscore(scores)
        assert r.mean_hybrid == pytest.approx(sum(scores) / len(scores), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Table rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderTable:
    def test_markdown_table_has_header(self, synthetic_results):
        results = run_h0_latency(synthetic_results)
        table = render_table(results, fmt="markdown")
        assert "| hypothesis |" in table
        assert "---" in table

    def test_latex_table_has_begin_tabular(self, synthetic_results):
        results = run_h0_latency(synthetic_results)
        table = render_table(results, fmt="latex")
        assert r"\begin{tabular}" in table
        assert r"\end{tabular}" in table

    def test_empty_input_returns_empty_string(self):
        assert render_table([]) == ""

    def test_invalid_format_raises(self, synthetic_results):
        results = run_h0_latency(synthetic_results)
        with pytest.raises(ValueError):
            render_table(results, fmt="csv")
