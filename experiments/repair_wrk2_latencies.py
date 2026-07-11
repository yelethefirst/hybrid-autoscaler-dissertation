"""Repair wrk2 latency fields in a Phase 5 JSONL result file.

The original _parse_wrk2() missed two cases:
  1. wrk2 uses "m" (minutes) for very-high latencies — parser only handled ms/s/us.
  2. wrk2 compact summary never includes 95th percentile — must come from the
     detailed HdrHistogram section (values in microseconds, percentile as decimal).

This script re-reads every saved wrk2.txt file, re-parses with the fixed parser,
and patches the corresponding JSONL record in-place.

Usage (on the server):
    uv run python -m experiments.repair_wrk2_latencies \\
        --result-dir experiments/results
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click


def parse_wrk2(raw: str) -> dict:
    """Fixed parser — mirrors the corrected _parse_wrk2() in supervisor.py."""
    metrics: dict = {}

    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"(\d+\.\d+)%\s+([\d.]+)(ms|s|us|m)\s*$", line)
        if m:
            pct = float(m.group(1))
            val = float(m.group(2))
            unit = m.group(3)
            if unit == "s":
                val *= 1000
            elif unit == "us":
                val /= 1000
            elif unit == "m":
                val *= 60_000
            if pct == 50.0:
                metrics["p50_latency_ms"] = val
            elif pct == 95.0:
                metrics["p95_latency_ms"] = val
            elif pct == 99.0:
                metrics["p99_latency_ms"] = val
            elif pct == 99.9:
                metrics["p999_latency_ms"] = val
        if "Requests/sec:" in line:
            try:
                metrics["throughput_rps"] = float(line.split()[-1].replace(",", ""))
            except ValueError:
                pass
        if "Non-2xx or 3xx responses:" in line:
            try:
                metrics["errors_total"] = int(line.split()[-1].replace(",", ""))
            except ValueError:
                pass

    in_detailed = False
    for line in raw.splitlines():
        if "Detailed Percentile spectrum:" in line:
            in_detailed = True
            continue
        if not in_detailed:
            continue
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+\d+\s+[\d.inf]+", line)
        if not m:
            continue
        val_ms = float(m.group(1)) / 1000.0
        pct = float(m.group(2))
        if abs(pct - 0.500) < 0.0001 and "p50_latency_ms" not in metrics:
            metrics["p50_latency_ms"] = val_ms
        elif abs(pct - 0.950) < 0.0001 and "p95_latency_ms" not in metrics:
            metrics["p95_latency_ms"] = val_ms
        elif abs(pct - 0.990) < 0.001 and "p99_latency_ms" not in metrics:
            metrics["p99_latency_ms"] = val_ms
        elif abs(pct - 0.999) < 0.001 and "p999_latency_ms" not in metrics:
            metrics["p999_latency_ms"] = val_ms

    return metrics


@click.command()
@click.option(
    "--result-dir",
    default="experiments/results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--dry-run", is_flag=True, help="Print changes without writing.")
def main(result_dir: Path, dry_run: bool) -> None:
    jsonl_files = sorted(result_dir.glob("results_*.jsonl"))
    if not jsonl_files:
        click.echo("No results_*.jsonl found.", err=True)
        sys.exit(1)

    for jsonl_path in jsonl_files:
        click.echo(f"Processing {jsonl_path}")
        records = []
        patched = 0

        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            trial_id = rec.get("trial_id", "")
            wrk2_txt = result_dir / "logs" / trial_id / "wrk2.txt"

            if not wrk2_txt.exists():
                click.echo(f"  SKIP {trial_id}: wrk2.txt not found")
                records.append(rec)
                continue

            raw = wrk2_txt.read_text()
            new_lat = parse_wrk2(raw)

            changed = False
            for key, new_val in new_lat.items():
                old_val = rec.get(key)
                if old_val != new_val:
                    if not dry_run:
                        rec[key] = new_val
                    changed = True

            if changed:
                patched += 1
                p95_old = rec.get("p95_latency_ms") if dry_run else new_lat.get("p95_latency_ms")
                p95_new = new_lat.get("p95_latency_ms")
                click.echo(
                    f"  PATCH {trial_id}: p95 {rec.get('p95_latency_ms'):.1f} → {p95_new:.1f} ms"
                    if p95_new is not None else f"  PATCH {trial_id}: latencies updated"
                )
            records.append(rec)

        if dry_run:
            click.echo(f"  DRY-RUN: {patched} records would be patched")
        else:
            jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            click.echo(f"  Patched {patched}/{len(records)} records → {jsonl_path}")


if __name__ == "__main__":
    main()
