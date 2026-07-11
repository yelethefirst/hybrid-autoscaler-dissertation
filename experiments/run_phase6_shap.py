"""Phase 6: Extract pre-computed SHAP attributions from Hybrid evidence bundles (§3.11).

The Hybrid controller computes and logs SHAP attributions for every scaling
decision in real time (see forecasting/holt_winters.py:shap_attribution and
controller/main.py). This script reads those logged attributions from the
evidence JSONL files and re-packages them into the format expected by Phase 7.

Faithfulness metrics (insertion/deletion AUC, param randomisation) require the
raw prediction history, which is not stored in evidence files; they are skipped
here and documented as a limitation.

Usage (on server):
    uv run python -m experiments.run_phase6_shap \\
        --result-dir experiments/results \\
        --out-dir experiments/results/shap

Output:
    experiments/results/shap/<trial_id>_shap.jsonl   (one row per decision)
    experiments/results/shap/summary.json             (aggregate counts)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import click


def _evidence_to_attribution(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a controller evidence row to the Phase 7 attribution dict format."""
    shap = row.get("shap", {})
    raw_features = shap.get("top_features", {})
    # raw_features is a dict {name: value}; convert to list of {name, shap}
    top_features = sorted(
        [{"name": k, "shap": float(v)} for k, v in raw_features.items()],
        key=lambda x: abs(x["shap"]),
        reverse=True,
    )
    return {
        "method": shap.get("method", "hw_components"),
        "expected_value": None,
        "top_features": top_features,
        "error": None,
    }


def _load_evidence(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@click.command()
@click.option(
    "--result-dir",
    default="experiments/results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out-dir",
    default="experiments/results/shap",
    type=click.Path(path_type=Path),
)
@click.option("--n-samples", default=30, help="Max decisions to extract per trial (for H4).")
def main(result_dir: Path, out_dir: Path, n_samples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_files = sorted(result_dir.glob("*-hybrid-*-evidence.jsonl"))
    if not evidence_files:
        click.echo("No evidence JSONL files found. Hybrid trials must complete first.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(evidence_files)} evidence files.")
    total_attributions = 0
    total_with_shap = 0

    for ev_path in evidence_files:
        trial_id = ev_path.stem.replace("-evidence", "")
        decisions = _load_evidence(ev_path)
        if not decisions:
            click.echo(f"  {trial_id}: empty, skipping")
            continue

        sample = decisions[:n_samples]
        out_rows = []

        for decision in sample:
            shap_present = bool(decision.get("shap", {}).get("top_features"))
            if not shap_present:
                out_rows.append({
                    "trial_id": trial_id,
                    "service": decision.get("service", "frontend"),
                    "decision_ts": decision.get("timestamp"),
                    "error": "no shap in evidence row",
                })
                continue

            attribution = _evidence_to_attribution(decision)
            out_rows.append({
                "trial_id": trial_id,
                "service": decision.get("service", "frontend"),
                "decision_ts": decision.get("timestamp"),
                "attribution": attribution,
                "faithfulness": None,
                "decision": decision,  # full evidence row for Phase 7 narrator
            })
            total_with_shap += 1

        total_attributions += len(out_rows)
        out_path = out_dir / f"{trial_id}_shap.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
        click.echo(f"  {trial_id}: {len(out_rows)} decisions → {out_path.name}")

    summary = {
        "n_evidence_files": len(evidence_files),
        "n_attributions_total": total_attributions,
        "n_attributions_with_shap": total_with_shap,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    click.echo(f"\nSummary: {summary}")
    click.echo(f"SHAP output → {out_dir}/")


if __name__ == "__main__":
    main()
