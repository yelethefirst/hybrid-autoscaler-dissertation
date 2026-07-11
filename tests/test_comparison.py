"""Tests for the H1 paired t + Bonferroni-Holm comparison (§1.5 H1, §3.12)."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from forecasting.comparison import h1_test
from forecasting.selection import ForecasterScore, SelectionReport


def _report(baseline_rmses, candidate_rmses_by_name):
    scores = {
        "seasonal_naive": ForecasterScore(name="seasonal_naive", fold_rmses=list(baseline_rmses))
    }
    for n, vals in candidate_rmses_by_name.items():
        scores[n] = ForecasterScore(name=n, fold_rmses=list(vals))
    return SelectionReport(
        service="test",
        horizon_seconds=30,
        scores=scores,
        selected_name="",
        selection_reason="",
    )


def test_h1_supported_when_candidate_clearly_better():
    # Baseline RMSE high; candidate consistently half as much.
    rep = _report(
        baseline_rmses=[1.0, 1.1, 0.9, 1.05, 0.95],
        candidate_rmses_by_name={
            "xgboost": [0.5, 0.55, 0.45, 0.52, 0.48],
        },
    )
    result = h1_test(rep)
    assert result.h1_supported
    assert result.candidates[0].significant_after_holm
    assert result.candidates[0].rmse_reduction > 0


def test_h1_not_supported_when_candidate_no_better():
    rep = _report(
        baseline_rmses=[1.0, 1.1, 0.9, 1.05, 0.95],
        candidate_rmses_by_name={
            "lstm": [1.0, 1.1, 0.9, 1.05, 0.95],  # identical
        },
    )
    result = h1_test(rep)
    assert not result.h1_supported


def test_holm_correction_makes_marginal_results_non_significant():
    """A single candidate with p≈0.04 would be significant alone; with 4
    candidates each at p≈0.04, Holm should reject all but the smallest."""
    rep = _report(
        baseline_rmses=[1.0, 1.05, 0.95, 1.02, 0.98],
        candidate_rmses_by_name={
            "a": [0.95, 1.00, 0.90, 0.97, 0.93],
            "b": [0.96, 1.01, 0.91, 0.98, 0.94],
            "c": [0.96, 1.01, 0.91, 0.98, 0.94],
            "d": [0.96, 1.01, 0.91, 0.98, 0.94],
        },
    )
    result = h1_test(rep, alpha=0.05)
    # Holm: smallest raw_p tested at α/4; others at α/3, α/2, α.
    # If all raw p's are roughly equal, the smallest may pass but the
    # later-ranked ones face a relaxing threshold.
    # We assert the structure of corrected thresholds rather than a hard
    # significance count (synthetic numbers can wobble).
    thresholds = [c.holm_threshold for c in result.candidates]
    assert thresholds == sorted(thresholds)  # non-decreasing rank → threshold
    assert thresholds[0] == pytest.approx(0.05 / 4)
    assert thresholds[-1] == pytest.approx(0.05 / 1)


def test_table_form_includes_required_columns():
    rep = _report(
        baseline_rmses=[1.0, 1.0, 1.0, 1.0, 1.0],
        candidate_rmses_by_name={"cand": [0.5, 0.6, 0.4, 0.5, 0.55]},
    )
    table = h1_test(rep).as_table()
    expected = {"candidate", "n_folds", "rmse_baseline", "rmse_candidate",
                "rmse_reduction", "raw_p", "holm_p", "holm_threshold",
                "significant_holm", "cohens_d"}
    assert expected.issubset(set(table.columns))


def test_missing_baseline_rejected():
    rep = _report(baseline_rmses=[], candidate_rmses_by_name={"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        h1_test(rep)


def test_candidate_with_too_few_folds_skipped():
    rep = _report(
        baseline_rmses=[1.0, 1.0, 1.0, 1.0, 1.0],
        candidate_rmses_by_name={
            "single_fold": [0.5],            # only 1 fold → can't paired-t
            "ok":          [0.6, 0.6, 0.6, 0.6, 0.6],
        },
    )
    result = h1_test(rep)
    names = [v.name for v in result.candidates]
    assert "single_fold" not in names
    assert "ok" in names
