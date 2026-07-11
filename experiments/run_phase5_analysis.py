"""Phase 5 statistical analysis — H0, H2, H3 pre-registered tests (§3.13).

Reads the canonical Phase 5 JSONL, pairs HPA vs Hybrid arms by trial number,
runs the pre-registered tests, and prints a dissertation-ready results table.

Usage (on server or Mac after rsync):
    uv run python -m experiments.run_phase5_analysis \\
        --result-jsonl experiments/results/results_canonical-ab-v2.jsonl \\
        --out experiments/results/phase5_stats.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import pandas as pd

from analysis.stats import (
    load_results,
    run_h0_latency,
    run_h2_replica_seconds,
    run_h3_oscillation,
    render_table,
)


def _trial_number(trial_id: str) -> str:
    """Extract the numeric suffix from a trial_id like 'burst-hpa-007' → '007'."""
    m = re.search(r"-(\d+)$", trial_id)
    return m.group(1) if m else trial_id


def _sort_for_pairing(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each arm within a workload by trial number so pairs align."""
    df = df.copy()
    df["_trial_num"] = df["trial_id"].apply(_trial_number)
    return df.sort_values(["workload", "autoscaler", "_trial_num"]).reset_index(drop=True)


@click.command()
@click.option(
    "--result-jsonl",
    default="experiments/results/results_canonical-ab-v2.jsonl",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--out",
    default="experiments/results/phase5_stats.json",
    type=click.Path(path_type=Path),
)
@click.option("--fmt", default="markdown", type=click.Choice(["markdown", "latex"]))
def main(result_jsonl: Path, out: Path, fmt: str) -> None:
    df = load_results(result_jsonl)
    df = _sort_for_pairing(df)

    # Exclude A/A trials from H0/H2/H3 (those have autoscaler = "hpaA" / "hpaB")
    df = df[df["autoscaler"].isin(["hpa", "hybrid"])].copy()

    click.echo(f"\nLoaded {len(df)} trials across workloads: {sorted(df['workload'].unique())}")
    for wl in sorted(df["workload"].unique()):
        wl_df = df[df["workload"] == wl]
        n_hpa = len(wl_df[wl_df["autoscaler"] == "hpa"])
        n_hyb = len(wl_df[wl_df["autoscaler"] == "hybrid"])
        click.echo(f"  {wl}: HPA n={n_hpa}, Hybrid n={n_hyb}")

    click.echo("\n── H0: p95 latency (Hybrid vs HPA) ─────────────────────────────")
    h0 = run_h0_latency(df)
    click.echo(render_table(h0, fmt=fmt))

    click.echo("\n── H0: p99 latency ──────────────────────────────────────────────")
    h0_p99 = run_h0_latency(df, outcome_col="p99_latency_ms")
    click.echo(render_table(h0_p99, fmt=fmt))

    click.echo("\n── H2: replica-seconds ──────────────────────────────────────────")
    h2 = run_h2_replica_seconds(df)
    click.echo(render_table(h2, fmt=fmt))

    click.echo("\n── H3: oscillation rate ─────────────────────────────────────────")
    h3 = run_h3_oscillation(df)
    click.echo(render_table([h3], fmt=fmt))

    all_results = [r.to_dict() for r in h0 + h0_p99 + h2 + [h3]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    click.echo(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
