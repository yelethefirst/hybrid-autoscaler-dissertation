"""Phase 7: LLM narrator + FActScore evaluation (§3.12, §3.13 H4).

Reads SHAP attribution JSONL files from Phase 6, generates 2–3 sentence
natural-language narratives via the OpenAI API, evaluates grounding with
FActScore, and writes results.

Usage:
    OPENAI_API_KEY=sk-... uv run python -m experiments.run_phase7_narrator \\
        --shap-dir experiments/results/shap \\
        --out experiments/results/phase7_narratives.jsonl \\
        --n-samples 30 \\
        --model gpt-4o-mini

Output:
    experiments/results/phase7_narratives.jsonl  (one row per narrative)
    experiments/results/phase7_h4.json           (H4 FActScore test result)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import click

from explain.attribution import Attribution
from narrate.narrator import Narrator
from narrate.factscore import FActScoreEvaluator
from analysis.stats import run_h4_factscore, render_table


@click.command()
@click.option(
    "--shap-dir",
    default="experiments/results/shap",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    default="experiments/results/phase7_narratives.jsonl",
    type=click.Path(path_type=Path),
)
@click.option("--n-samples", default=30, help="Number of narratives to generate (H4 target: 30).")
@click.option("--model", default="gpt-4o-mini", help="OpenAI model name.")
def main(shap_dir: Path, out: Path, n_samples: int, model: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        click.echo("ERROR: OPENAI_API_KEY not set.", err=True)
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    shap_files = sorted(shap_dir.glob("*_shap.jsonl"))
    if not shap_files:
        click.echo("No SHAP JSONL files found. Run Phase 6 first.", err=True)
        sys.exit(1)

    # Collect attribution rows that have valid attribution and full decision
    all_rows = []
    for f in shap_files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                if not r.get("error") and r.get("attribution") and r.get("decision"):
                    all_rows.append(r)

    sample = all_rows[:n_samples]
    click.echo(f"Narrating {len(sample)} decisions (from {len(all_rows)} available).")

    narrator = Narrator(client=client, model=model)
    evaluator = FActScoreEvaluator(client=client, judge_model=model)
    out.parent.mkdir(parents=True, exist_ok=True)
    factscores: List[float] = []

    with open(out, "w") as out_f:
        for i, row in enumerate(sample, 1):
            decision = row["decision"]
            attr_dict = row["attribution"]
            attribution = Attribution(
                top_features=[(f["name"], f["shap"]) for f in attr_dict.get("top_features", [])],
                method=attr_dict.get("method", "unknown"),
                expected_value=attr_dict.get("expected_value"),
                raw_shap={f["name"]: f["shap"] for f in attr_dict.get("top_features", [])},
            )

            result = narrator.narrate(decision, attribution)
            fs_result = evaluator.score(result.narrative, result.context) if result.narrative else None
            fs = fs_result.factscore if fs_result and fs_result.factscore is not None else 0.0
            factscores.append(fs)

            out_row = {
                "idx": i,
                "trial_id": row.get("trial_id"),
                "service": row.get("service"),
                "narrative": result.narrative,
                "factscore": fs,
                "factscore_detail": fs_result.to_dict() if fs_result else None,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "error": result.error,
            }
            out_f.write(json.dumps(out_row) + "\n")
            click.echo(f"  [{i}/{len(sample)}] {row.get('service')}: FActScore={fs:.3f}")

    click.echo(f"\n{len(factscores)} narratives written → {out}")

    # H4 test
    if len(factscores) >= 2:
        h4 = run_h4_factscore(factscores)
        click.echo("\n── H4: FActScore ≥ 0.8 ─────────────────────────────────────────")
        click.echo(render_table([h4]))
        h4_path = out.parent / "phase7_h4.json"
        h4_path.write_text(json.dumps(h4.to_dict(), indent=2))
        click.echo(f"H4 result → {h4_path}")


if __name__ == "__main__":
    main()
