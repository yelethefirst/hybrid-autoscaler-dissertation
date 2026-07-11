"""H1 hypothesis test: each candidate vs Seasonal Naive (§1.5 / §3.12).

§1.5 H1: "At least one candidate model achieves a statistically significant
reduction in RMSE versus the Seasonal Naive baseline on held-out workload
windows (paired t-test, p < 0.05, with Bonferroni–Holm correction across
the four model families)."

The test is paired across walk-forward folds: for each fold f we have
RMSE_baseline(f) and RMSE_candidate(f), and we test
    H0: mean(baseline − candidate) ≤ 0  (candidate no better than baseline)
    H1: mean(baseline − candidate) >  0  (candidate strictly better)

with a one-sided paired t-test. Bonferroni–Holm corrects across the four
candidate families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats

from .selection import SelectionReport


@dataclass(frozen=True)
class CandidateVerdict:
    """One candidate's verdict against the Seasonal Naive baseline."""

    name: str
    n_paired_folds: int
    mean_rmse_baseline: float
    mean_rmse_candidate: float
    rmse_reduction: float                   # positive = candidate is better
    t_statistic: float
    raw_p_value: float
    holm_p_value: float
    holm_threshold: float                   # the Holm-corrected α for this rank
    significant_after_holm: bool

    @property
    def cohens_d(self) -> float:
        """Cohen's d on the paired RMSE differences (effect size)."""
        return self._cohens_d


    _cohens_d: float = 0.0


@dataclass(frozen=True)
class H1Result:
    """Aggregated H1 verdict — does at least one candidate beat Seasonal Naive?"""

    candidates: List[CandidateVerdict]
    h1_supported: bool
    alpha: float = 0.05
    baseline_name: str = "seasonal_naive"

    def as_table(self):
        import pandas as pd
        rows = []
        for v in self.candidates:
            rows.append(
                {
                    "candidate": v.name,
                    "n_folds": v.n_paired_folds,
                    "rmse_baseline": v.mean_rmse_baseline,
                    "rmse_candidate": v.mean_rmse_candidate,
                    "rmse_reduction": v.rmse_reduction,
                    "t_stat": v.t_statistic,
                    "raw_p": v.raw_p_value,
                    "holm_p": v.holm_p_value,
                    "holm_threshold": v.holm_threshold,
                    "significant_holm": v.significant_after_holm,
                    "cohens_d": v.cohens_d,
                }
            )
        return pd.DataFrame(rows).sort_values("raw_p").reset_index(drop=True)


# ---------------------------------------------------------------------- #
# Public API                                                              #
# ---------------------------------------------------------------------- #
def h1_test(
    report: SelectionReport,
    *,
    baseline_name: str = "seasonal_naive",
    alpha: float = 0.05,
) -> H1Result:
    """Run the H1 paired t-test with Bonferroni–Holm across all candidates.

    Parameters
    ----------
    report : SelectionReport
        Output of `evaluate_forecasters`. Must contain `baseline_name` and
        at least one other candidate, both with non-empty `fold_rmses`.
    baseline_name : str
        Name of the baseline forecaster in `report.scores`. Default
        "seasonal_naive" matches §1.5.
    alpha : float
        Family-wise α. Default 0.05 per §1.5.
    """
    if baseline_name not in report.scores:
        raise ValueError(f"baseline '{baseline_name}' not in report scores")
    base_score = report.scores[baseline_name]
    if not base_score.fold_rmses:
        raise ValueError(f"baseline '{baseline_name}' has no fold RMSEs")

    candidates_raw: List[Dict] = []
    for name, score in report.scores.items():
        if name == baseline_name:
            continue
        if not score.fold_rmses:
            continue
        n_paired = min(len(base_score.fold_rmses), len(score.fold_rmses))
        if n_paired < 2:
            continue
        base_arr = np.asarray(base_score.fold_rmses[:n_paired], dtype=float)
        cand_arr = np.asarray(score.fold_rmses[:n_paired], dtype=float)
        diff = base_arr - cand_arr            # positive = candidate is better
        # One-sided paired t-test: H1 mean(diff) > 0
        t_stat, p_two = stats.ttest_rel(base_arr, cand_arr)
        p_one_sided = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        cohens = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
        candidates_raw.append(
            dict(
                name=name,
                n=n_paired,
                base_mean=float(np.mean(base_arr)),
                cand_mean=float(np.mean(cand_arr)),
                reduction=float(np.mean(diff)),
                t=float(t_stat),
                raw_p=float(p_one_sided),
                cohens=cohens,
            )
        )

    # Bonferroni-Holm correction: sort by raw_p ascending; threshold for
    # rank i (1-indexed) is α / (m - i + 1) where m = number of candidates.
    candidates_raw.sort(key=lambda x: x["raw_p"])
    m = len(candidates_raw)

    verdicts: List[CandidateVerdict] = []
    any_significant = False
    for i, c in enumerate(candidates_raw, start=1):
        threshold = alpha / (m - i + 1) if m > 0 else alpha
        # Holm-adjusted p = max-so-far of raw_p * (m - i + 1), capped at 1.
        holm_p = min(1.0, c["raw_p"] * (m - i + 1))
        if verdicts:
            holm_p = max(holm_p, verdicts[-1].holm_p_value)
        sig = c["raw_p"] < threshold
        if sig:
            any_significant = True
        v = CandidateVerdict(
            name=c["name"],
            n_paired_folds=c["n"],
            mean_rmse_baseline=c["base_mean"],
            mean_rmse_candidate=c["cand_mean"],
            rmse_reduction=c["reduction"],
            t_statistic=c["t"],
            raw_p_value=c["raw_p"],
            holm_p_value=holm_p,
            holm_threshold=threshold,
            significant_after_holm=sig,
            _cohens_d=c["cohens"],
        )
        verdicts.append(v)

    return H1Result(
        candidates=verdicts,
        h1_supported=any_significant,
        alpha=alpha,
        baseline_name=baseline_name,
    )
