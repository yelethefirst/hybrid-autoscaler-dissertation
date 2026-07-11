"""Pre-registered statistical tests for the A/B experiment (§3.13).

Each function corresponds to one hypothesis in preregistration/hypotheses.yaml.
Tests are implemented exactly as specified: no post-hoc modifications.

All functions return HypothesisResult, which is JSON-serialisable and can be
rendered directly as a Chapter 4 table row.

Bootstrap method: BCa (bias-corrected and accelerated) via scipy.
Multiple comparisons: Holm-Bonferroni via statsmodels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(result_jsonl: str | Path) -> pd.DataFrame:
    """Load a trial result JSONL into a DataFrame.

    Returns columns including: trial_id, autoscaler, workload, seed,
    p95_latency_ms, replica_seconds, oscillation_count, etc.
    """
    rows = []
    with open(result_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BCa bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _bca_ci(
    data: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    data:
        1-D array of observed values.
    statistic_fn:
        Function (data → scalar) to bootstrap.
    n_bootstrap:
        Number of bootstrap samples.
    confidence:
        Nominal coverage (default 0.95).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    (lower, upper) BCa CI.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    obs = statistic_fn(data)

    # Bootstrap distribution
    boot = np.array([
        statistic_fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Degenerate case: all bootstrap samples identical
    if np.std(boot) < 1e-12:
        return float(obs), float(obs)

    # Bias correction z₀
    z0 = stats.norm.ppf(np.mean(boot < obs))
    if not math.isfinite(z0):
        z0 = 0.0

    # Jackknife acceleration a
    jack = np.array([statistic_fn(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if abs(den) > 1e-12 else 0.0

    alpha = 1 - confidence
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def _pct(z):
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        return stats.norm.cdf(adj)

    lo_pct = _pct(z_alpha_lo)
    hi_pct = _pct(z_alpha_hi)
    lo, hi = np.percentile(boot, [lo_pct * 100, hi_pct * 100])
    return float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_z(differences: np.ndarray) -> float:
    """Cohen's d_z for paired comparisons (standardised mean difference)."""
    return float(np.mean(differences) / (np.std(differences, ddof=1) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, float, bool]]:
    """Apply Holm-Bonferroni step-down correction.

    Returns a list of (adjusted_p, threshold, significant) tuples
    in the same order as the input p_values.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = [p_values[i] for i in order]

    results = [(0.0, 0.0, False)] * n
    rejected_all = True
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        adjusted_p = sorted_p[rank] * (n - rank)
        adjusted_p = min(adjusted_p, 1.0)
        if sorted_p[rank] > threshold:
            rejected_all = False
        significant = rejected_all and (sorted_p[rank] <= threshold)
        results[idx] = (adjusted_p, threshold, significant)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HypothesisResult:
    """Result of one pre-registered hypothesis test."""

    hypothesis: str
    outcome: str           # metric name
    workload: Optional[str]

    n_pairs: int
    mean_hybrid: float
    mean_hpa: float
    mean_diff: float       # hybrid − hpa (negative = hybrid better for latency/resource)

    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    holm_p: Optional[float] = None
    holm_threshold: Optional[float] = None
    significant: bool = False

    cohens_d_z: Optional[float] = None
    bca_ci_lo: Optional[float] = None
    bca_ci_hi: Optional[float] = None
    bca_coverage: float = 0.95

    verdict: str = ""      # "reject H_null" or "fail to reject H_null"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "outcome": self.outcome,
            "workload": self.workload,
            "n_pairs": self.n_pairs,
            "mean_hybrid": round(self.mean_hybrid, 4),
            "mean_hpa": round(self.mean_hpa, 4),
            "mean_diff": round(self.mean_diff, 4),
            "t_stat": round(self.t_stat, 4) if self.t_stat is not None else None,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "holm_p": round(self.holm_p, 6) if self.holm_p is not None else None,
            "holm_threshold": round(self.holm_threshold, 6) if self.holm_threshold is not None else None,
            "significant": bool(self.significant),
            "cohens_d_z": round(self.cohens_d_z, 4) if self.cohens_d_z is not None else None,
            "bca_ci": [round(self.bca_ci_lo, 4), round(self.bca_ci_hi, 4)]
            if self.bca_ci_lo is not None else None,
            "verdict": self.verdict,
        }


# ─────────────────────────────────────────────────────────────────────────────
# H0: latency difference across workloads
# ─────────────────────────────────────────────────────────────────────────────

def run_h0_latency(
    df: pd.DataFrame,
    outcome_col: str = "p95_latency_ms",
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 42,
) -> List[HypothesisResult]:
    """H0: no p95-latency difference (Hybrid vs HPA), per workload, Holm-corrected.

    Returns one HypothesisResult per workload.
    """
    workloads = sorted(df["workload"].unique())
    raw_results = []

    for wl in workloads:
        wl_df = df[df["workload"] == wl].copy()
        hybrid = wl_df[wl_df["autoscaler"] == "hybrid"][outcome_col].dropna()
        hpa = wl_df[wl_df["autoscaler"] == "hpa"][outcome_col].dropna()

        n = min(len(hybrid), len(hpa))
        if n < 2:
            raw_results.append(None)
            continue

        h = hybrid.values[:n]
        b = hpa.values[:n]
        diff = h - b

        t, p = stats.ttest_rel(h, b)
        d = cohens_d_z(diff)
        lo, hi = _bca_ci(diff, np.mean, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

        raw_results.append((wl, n, h, b, diff, t, p, d, lo, hi))

    # Holm-Bonferroni across workloads
    valid = [(i, r) for i, r in enumerate(raw_results) if r is not None]
    p_vals = [r[6] for _, r in valid]
    holm = holm_bonferroni(p_vals, alpha)

    results = []
    holm_idx = 0
    for i, r in enumerate(raw_results):
        if r is None:
            continue
        wl, n, h, b, diff, t, p, d, lo, hi = r
        adj_p, threshold, sig = holm[holm_idx]
        holm_idx += 1
        verdict = "reject H_null" if sig else "fail to reject H_null"
        results.append(HypothesisResult(
            hypothesis="H0",
            outcome=outcome_col,
            workload=wl,
            n_pairs=n,
            mean_hybrid=float(np.mean(h)),
            mean_hpa=float(np.mean(b)),
            mean_diff=float(np.mean(diff)),
            t_stat=float(t),
            p_value=float(p),
            holm_p=float(adj_p),
            holm_threshold=float(threshold),
            significant=sig,
            cohens_d_z=float(d),
            bca_ci_lo=float(lo),
            bca_ci_hi=float(hi),
            verdict=verdict,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H2: replica-seconds efficiency
# ─────────────────────────────────────────────────────────────────────────────

def run_h2_replica_seconds(
    df: pd.DataFrame,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 42,
) -> List[HypothesisResult]:
    """H2: Hybrid uses fewer replica-seconds than HPA (burst + ramp only, one-sided)."""
    results = []
    target_workloads = ["burst", "ramp"]

    for wl in target_workloads:
        wl_df = df[df["workload"] == wl].copy()
        hybrid = wl_df[wl_df["autoscaler"] == "hybrid"]["replica_seconds"].dropna()
        hpa = wl_df[wl_df["autoscaler"] == "hpa"]["replica_seconds"].dropna()

        n = min(len(hybrid), len(hpa))
        if n < 2:
            continue

        h, b = hybrid.values[:n], hpa.values[:n]
        diff = h - b   # negative = hybrid uses fewer replica-seconds

        # One-sided test: H1: mu < 0 (hybrid < hpa)
        t, p_two = stats.ttest_rel(h, b)
        p = p_two / 2 if t < 0 else 1 - p_two / 2
        d = cohens_d_z(diff)
        lo, hi = _bca_ci(diff, np.mean, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        sig = p < alpha
        results.append(HypothesisResult(
            hypothesis="H2",
            outcome="replica_seconds",
            workload=wl,
            n_pairs=n,
            mean_hybrid=float(np.mean(h)),
            mean_hpa=float(np.mean(b)),
            mean_diff=float(np.mean(diff)),
            t_stat=float(t),
            p_value=float(p),
            significant=sig,
            cohens_d_z=float(d),
            bca_ci_lo=float(lo),
            bca_ci_hi=float(hi),
            verdict="reject H_null" if sig else "fail to reject H_null",
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H3: oscillation count (non-parametric)
# ─────────────────────────────────────────────────────────────────────────────

def run_h3_oscillation(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> HypothesisResult:
    """H3: Hybrid produces fewer oscillations than HPA (Wilcoxon signed-rank)."""
    hybrid = df[df["autoscaler"] == "hybrid"]["oscillation_count"].dropna()
    hpa = df[df["autoscaler"] == "hpa"]["oscillation_count"].dropna()
    n = min(len(hybrid), len(hpa))
    h, b = hybrid.values[:n], hpa.values[:n]
    diff = h - b  # negative = hybrid fewer oscillations

    stat, p_two = stats.wilcoxon(h, b, alternative="two-sided")
    p = p_two / 2 if np.mean(diff) < 0 else 1 - p_two / 2
    sig = p < alpha
    return HypothesisResult(
        hypothesis="H3",
        outcome="oscillation_count",
        workload=None,
        n_pairs=n,
        mean_hybrid=float(np.mean(h)),
        mean_hpa=float(np.mean(b)),
        mean_diff=float(np.mean(diff)),
        p_value=float(p),
        significant=sig,
        verdict="reject H_null" if sig else "fail to reject H_null",
    )


# ─────────────────────────────────────────────────────────────────────────────
# H4: LLM FActScore >= 0.8
# ─────────────────────────────────────────────────────────────────────────────

def run_h4_factscore(
    factscores: List[float],
    mu0: float = 0.8,
    alpha: float = 0.05,
) -> HypothesisResult:
    """H4: FActScore >= mu0 (one-sample Wilcoxon signed-rank vs mu0).

    Parameters
    ----------
    factscores:
        List of per-narrative FActScore values (n=30 target).
    mu0:
        Hypothesised minimum (default 0.8, per preregistration).
    """
    arr = np.array(factscores)
    n = len(arr)
    stat, p = stats.wilcoxon(arr - mu0, alternative="greater")
    mean_fs = float(np.mean(arr))
    sig = p < alpha
    return HypothesisResult(
        hypothesis="H4",
        outcome="factscore",
        workload=None,
        n_pairs=n,
        mean_hybrid=mean_fs,
        mean_hpa=mu0,
        mean_diff=mean_fs - mu0,
        p_value=float(p),
        significant=sig,
        verdict="reject H_null (FActScore >= 0.8)" if sig else "fail to reject H_null",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Table rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_table(results: List[HypothesisResult], fmt: str = "markdown") -> str:
    """Render a list of HypothesisResult as a table string.

    Parameters
    ----------
    fmt:
        "markdown" (GitHub-flavoured) or "latex".
    """
    if not results:
        return ""

    rows = [r.to_dict() for r in results]
    cols = [
        "hypothesis", "outcome", "workload", "n_pairs",
        "mean_hybrid", "mean_hpa", "mean_diff",
        "t_stat", "p_value", "holm_p", "significant",
        "cohens_d_z", "bca_ci", "verdict",
    ]

    if fmt == "markdown":
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for r in rows:
            vals = [str(r.get(c, "")) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    if fmt == "latex":
        col_spec = "l" * len(cols)
        lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\hline",
            " & ".join(cols) + r" \\",
            r"\hline",
        ]
        for r in rows:
            vals = [str(r.get(c, "")).replace("_", r"\_") for c in cols]
            lines.append(" & ".join(vals) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)

    raise ValueError(f"unsupported format: {fmt}")
