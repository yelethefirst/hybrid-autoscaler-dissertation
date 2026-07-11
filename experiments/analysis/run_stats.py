"""H0–H4 statistical analysis — §3.11 / §3.12.

Reads the JSONL trial-result log from Phase 5 and runs every pre-registered
statistical test from preregistration/hypotheses.yaml.

Hypothesis summary
------------------
H0  p95 latency: paired-t, two-sided, Holm-Bonferroni across 4 workloads
H1  Forecasting vs Seasonal Naive: already decided in Phase 3 h1.csv
H2  Replica-seconds efficiency: one-sided paired-t, Holm-Bonferroni
H3  Oscillation count: one-sided Wilcoxon signed-rank
H4  LLM narrative FActScore ≥ 0.8: one-sample Wilcoxon (placeholder)

All tests, effect sizes (Cohen's d_z), and BCa bootstrap CIs follow
preregistration/hypotheses.yaml exactly (α=0.05, bootstrap n=10 000, seed=42).

Usage
-----
    uv run python experiments/analysis/run_stats.py \\
        --results experiments/results/results_canonical-ab-10-trials.jsonl \\
        [--h1-csv experiments/results/phase3_20260703T090249Z_h1.csv] \\
        [--factscore-csv experiments/results/factscore.csv] \\
        [--out experiments/results/stats_report.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ALPHA = 0.05
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
WORKLOADS = ["burst", "ramp", "periodic", "trace_replay"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Normalise workload name (trace-* → trace_replay)
    df["workload"] = df["workload"].str.replace("-", "_")
    return df


def build_pairs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a DataFrame with columns: workload, pair_id, hybrid, hpa, diff."""
    records = []
    for wl in WORKLOADS:
        hyb = df[(df["workload"] == wl) & (df["autoscaler"] == "hybrid")][metric].dropna().values
        hpa = df[(df["workload"] == wl) & (df["autoscaler"] == "hpa")][metric].dropna().values
        n = min(len(hyb), len(hpa))
        for i in range(n):
            records.append({
                "workload": wl,
                "pair_id": i + 1,
                "hybrid": hyb[i],
                "hpa": hpa[i],
                "diff": hyb[i] - hpa[i],
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical primitives
# ─────────────────────────────────────────────────────────────────────────────

def cohens_dz(diffs: np.ndarray) -> float:
    """Cohen's d_z for paired data: mean(d) / std(d, ddof=1)."""
    if len(diffs) < 2:
        return float("nan")
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def bca_ci(
    diffs: np.ndarray,
    statistic=np.mean,
    confidence: float = 0.95,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """BCa 95% bootstrap confidence interval (scipy ≥ 1.7)."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    result = stats.bootstrap(
        (diffs,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="BCa",
        random_state=seed,
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (same length as input)."""
    k = len(p_values)
    if k == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.array(p_values, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (k - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def fmt(x, decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{decimals}f}"


def verdict(reject: bool) -> str:
    return "REJECT H_null ✓" if reject else "fail to reject H_null"


# ─────────────────────────────────────────────────────────────────────────────
# H0 — p95 latency (two-sided paired-t, Holm across workloads)
# ─────────────────────────────────────────────────────────────────────────────

def run_h0(df: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pairs(df, "p95_latency_ms")
    if pairs.empty:
        print("H0: no p95_latency_ms data — skipping")
        return pd.DataFrame()

    rows = []
    raw_p = []
    for wl in WORKLOADS:
        d = pairs[pairs["workload"] == wl]["diff"].values
        if len(d) < 2:
            rows.append({"workload": wl, "n_pairs": len(d), "mean_diff_ms": float("nan"),
                         "t_stat": float("nan"), "p_raw": float("nan"), "d_z": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan")})
            raw_p.append(1.0)
            continue
        t, p = stats.ttest_rel(
            pairs[pairs["workload"] == wl]["hybrid"].values,
            pairs[pairs["workload"] == wl]["hpa"].values,
        )
        dz = cohens_dz(d)
        ci_lo, ci_hi = bca_ci(d)
        rows.append({
            "workload": wl, "n_pairs": len(d), "mean_diff_ms": float(np.mean(d)),
            "t_stat": float(t), "p_raw": float(p), "d_z": dz,
            "ci_low": ci_lo, "ci_high": ci_hi,
        })
        raw_p.append(float(p))

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_bonferroni(raw_p)
    result["reject"] = result["p_holm"] < ALPHA

    print("\n── H0: p95 Latency (two-sided paired-t, Holm-Bonferroni) ──────────────")
    for _, r in result.iterrows():
        print(
            f"  {r['workload']:15s}  n={int(r['n_pairs'])}  "
            f"mean_diff={fmt(r['mean_diff_ms'])}ms  "
            f"t={fmt(r['t_stat'])}  p={fmt(r['p_raw'])}  p_holm={fmt(r['p_holm'])}  "
            f"d_z={fmt(r['d_z'])}  "
            f"BCa95%=[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]  "
            f"→ {verdict(r['reject'])}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# H1 — Forecasting vs Seasonal Naive (from Phase 3 h1.csv)
# ─────────────────────────────────────────────────────────────────────────────

def run_h1(h1_csv: Optional[Path]) -> pd.DataFrame:
    """Parse Phase 3 H1 CSV.

    Actual columns: candidate, n_folds, rmse_baseline, rmse_candidate,
    rmse_reduction, t_stat, raw_p, holm_p, holm_threshold, significant_holm,
    cohens_d, service

    H1 is confirmed per-service if ANY candidate has significant_holm=True.
    Overall H1 is confirmed if at least one service confirms.
    """
    print("\n── H1: Forecasting > Seasonal Naive (Phase 3 result) ─────────────────")
    if h1_csv is None or not h1_csv.exists():
        print("  H1 CSV not provided or not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(h1_csv)

    # Best candidate per service: lowest holm_p among significant, else lowest raw_p
    summary_rows = []
    for svc, grp in df.groupby("service"):
        sig = grp[grp["significant_holm"].astype(bool)]
        if not sig.empty:
            best = sig.loc[sig["holm_p"].idxmin()]
            reject = True
        else:
            best = grp.loc[grp["raw_p"].idxmin()]
            reject = False
        summary_rows.append({
            "service": svc,
            "best_candidate": best["candidate"],
            "rmse_reduction": best["rmse_reduction"],
            "holm_p": best["holm_p"],
            "cohens_d": best["cohens_d"],
            "reject_h1": reject,
        })
    summary = pd.DataFrame(summary_rows)
    n_reject = int(summary["reject_h1"].sum())

    print(f"  Services with a candidate beating Seasonal Naive (holm_p < threshold): "
          f"{n_reject} / {len(summary)}")
    for _, r in summary.iterrows():
        print(
            f"  {str(r['service']):25s}  best={r['best_candidate']}  "
            f"rmse_red={fmt(r['rmse_reduction'])}  holm_p={fmt(r['holm_p'])}  "
            f"d={fmt(r['cohens_d'])}  → {verdict(r['reject_h1'])}"
        )
    overall = n_reject >= 1
    print(f"  H1 overall verdict: {'CONFIRMED' if overall else 'not confirmed'} "
          f"({n_reject}/{len(summary)} services reject H_null)")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Replica-seconds efficiency (one-sided paired-t, Holm)
# ─────────────────────────────────────────────────────────────────────────────

def run_h2(df: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pairs(df, "replica_seconds")
    if pairs.empty:
        print("H2: no replica_seconds data — skipping")
        return pd.DataFrame()

    rows = []
    raw_p = []
    for wl in ["burst", "ramp"]:  # H2 pre-registered only for burst + ramp
        d = pairs[pairs["workload"] == wl]["diff"].values
        if len(d) < 2:
            rows.append({"workload": wl, "n_pairs": len(d), "mean_diff": float("nan"),
                         "t_stat": float("nan"), "p_raw": float("nan"), "d_z": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan")})
            raw_p.append(1.0)
            continue
        hyb = pairs[pairs["workload"] == wl]["hybrid"].values
        hpa = pairs[pairs["workload"] == wl]["hpa"].values
        # One-sided: alternative = hybrid < hpa  ↔  alternative="less"
        t, p = stats.ttest_rel(hyb, hpa, alternative="less")
        dz = cohens_dz(d)
        ci_lo, ci_hi = bca_ci(d)
        rows.append({
            "workload": wl, "n_pairs": len(d), "mean_diff": float(np.mean(d)),
            "t_stat": float(t), "p_raw": float(p), "d_z": dz,
            "ci_low": ci_lo, "ci_high": ci_hi,
        })
        raw_p.append(float(p))

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_bonferroni(raw_p)
    result["reject"] = result["p_holm"] < ALPHA

    print("\n── H2: Replica-Seconds Efficiency (one-sided paired-t, Holm) ──────────")
    for _, r in result.iterrows():
        print(
            f"  {r['workload']:15s}  n={int(r['n_pairs'])}  "
            f"mean_diff={fmt(r['mean_diff'])}  "
            f"t={fmt(r['t_stat'])}  p={fmt(r['p_raw'])}  p_holm={fmt(r['p_holm'])}  "
            f"d_z={fmt(r['d_z'])}  "
            f"BCa95%=[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]  "
            f"→ {verdict(r['reject'])}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Oscillation count (one-sided Wilcoxon signed-rank)
# ─────────────────────────────────────────────────────────────────────────────

def run_h3(df: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pairs(df, "oscillation_count")
    if pairs.empty or pairs["diff"].isna().all():
        print("\nH3: no oscillation_count data — skipping")
        return pd.DataFrame()

    d = pairs["diff"].dropna().values
    if len(d) < 2:
        print("\nH3: insufficient pairs for Wilcoxon — skipping")
        return pd.DataFrame()

    # One-sided: hybrid oscillates less → differences (hybrid - hpa) < 0
    stat, p = stats.wilcoxon(d, alternative="less")
    reject = p < ALPHA

    print("\n── H3: Oscillation Count (one-sided Wilcoxon signed-rank) ─────────────")
    print(f"  n_pairs={len(d)}  mean_diff={fmt(np.mean(d))}  "
          f"W={fmt(stat, 1)}  p={fmt(p)}  → {verdict(reject)}")

    return pd.DataFrame([{
        "n_pairs": len(d), "mean_diff": float(np.mean(d)),
        "W_stat": float(stat), "p_raw": float(p),
        "reject": reject,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# H4 — LLM narrative FActScore (one-sample Wilcoxon vs μ₀ = 0.8)
# ─────────────────────────────────────────────────────────────────────────────

def run_h4(factscore_csv: Optional[Path]) -> pd.DataFrame:
    MU0 = 0.8
    print("\n── H4: LLM Narrative FActScore ≥ 0.8 (one-sample Wilcoxon) ────────────")
    if factscore_csv is None or not factscore_csv.exists():
        print("  FActScore CSV not available yet — Phase 7 not run. Skipping.")
        print("  Expected columns: trial_id, factscore (float in [0,1])")
        return pd.DataFrame()

    fs = pd.read_csv(factscore_csv)
    scores = fs["factscore"].dropna().values
    if len(scores) < 2:
        print("  Insufficient FActScore samples — skipping")
        return pd.DataFrame()

    # One-sided: H_alt: median > 0.8  → test scores - mu0 > 0 → alternative="greater"
    stat, p = stats.wilcoxon(scores - MU0, alternative="greater")
    reject = p < ALPHA

    print(f"  n={len(scores)}  mean={fmt(np.mean(scores))}  median={fmt(np.median(scores))}  "
          f"W={fmt(stat, 1)}  p={fmt(p)}  → {verdict(reject)}")

    return pd.DataFrame([{
        "n": len(scores), "mean_factscore": float(np.mean(scores)),
        "median_factscore": float(np.median(scores)),
        "W_stat": float(stat), "p_raw": float(p), "reject": reject,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path("experiments/results/results_canonical-ab-10-trials.jsonl"),
        help="JSONL trial result log from Phase 5 supervisor",
    )
    parser.add_argument(
        "--h1-csv", type=Path, default=None,
        help="Phase 3 H1 verdict CSV (phase3_*_h1.csv). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--factscore-csv", type=Path, default=None,
        help="FActScore CSV from Phase 7 (trial_id, factscore). Optional.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("experiments/results/stats_report.csv"),
        help="Output CSV with all verdicts (one row per hypothesis/workload).",
    )
    args = parser.parse_args()

    # Auto-detect H1 CSV
    h1_csv = args.h1_csv
    if h1_csv is None:
        candidates = sorted(
            Path("experiments/results").glob("phase3_*_h1.csv"), reverse=True
        )
        if candidates:
            h1_csv = candidates[0]

    print(f"Loading results: {args.results}")
    if not args.results.exists():
        print(f"ERROR: result file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    df = load_results(args.results)
    print(f"  {len(df)} trial rows loaded")
    print(f"  autoscalers : {sorted(df['autoscaler'].unique())}")
    print(f"  workloads   : {sorted(df['workload'].unique())}")

    # Check for dry-run data (all zeros → warn)
    if "p95_latency_ms" in df.columns and (df["p95_latency_ms"] == 0).all():
        print("\n  ⚠  WARNING: all p95_latency_ms are 0 — this looks like dry-run data.")
        print("     Run the real A/B experiment before interpreting these results.\n")

    # Run all hypotheses
    h0 = run_h0(df)
    h1 = run_h1(h1_csv)
    h2 = run_h2(df)
    h3 = run_h3(df)
    h4 = run_h4(args.factscore_csv)

    # Summary table
    print("\n══════════════════════════════════════════════════════")
    print("HYPOTHESIS VERDICT SUMMARY")
    print("══════════════════════════════════════════════════════")
    _print_verdict("H0", h0, col="reject")
    _print_verdict_h1(h1)
    _print_verdict("H2", h2, col="reject")
    _print_verdict("H3", h3, col="reject")
    _print_verdict("H4", h4, col="reject")

    # Save combined CSV
    frames = []
    for label, frame, cols in [
        ("H0", h0, ["workload", "n_pairs", "mean_diff_ms", "t_stat", "p_raw", "p_holm", "d_z", "ci_low", "ci_high", "reject"]),
        ("H2", h2, ["workload", "n_pairs", "mean_diff", "t_stat", "p_raw", "p_holm", "d_z", "ci_low", "ci_high", "reject"]),
        ("H3", h3, ["n_pairs", "mean_diff", "W_stat", "p_raw", "reject"]),
        ("H4", h4, ["n", "mean_factscore", "median_factscore", "W_stat", "p_raw", "reject"]),
    ]:
        if frame is not None and not frame.empty:
            sub = frame[[c for c in cols if c in frame.columns]].copy()
            sub.insert(0, "hypothesis", label)
            frames.append(sub)

    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nVerdicts written to: {args.out}")


def _print_verdict(label: str, df: pd.DataFrame, col: str = "reject") -> None:
    if df is None or df.empty or col not in df.columns:
        print(f"  {label}: no data")
        return
    n_reject = int(df[col].sum())
    n_total = len(df)
    status = "SUPPORTED" if n_reject > 0 else "NOT SUPPORTED"
    print(f"  {label}: {status} ({n_reject}/{n_total} comparisons reject H_null at α=0.05)")


def _print_verdict_h1(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("  H1: no data")
        return
    col = "reject_h1" if "reject_h1" in df.columns else None
    if col:
        n_reject = int(df[col].sum())
        status = "CONFIRMED" if n_reject >= 1 else "NOT CONFIRMED"
        print(f"  H1: {status} ({n_reject}/{len(df)} services reject H_null at Phase 3)")
    else:
        print("  H1: (Phase 3 data loaded, reject_h1 column not found)")


if __name__ == "__main__":
    main()
