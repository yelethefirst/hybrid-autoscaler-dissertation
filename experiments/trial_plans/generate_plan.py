#!/usr/bin/env python3
"""Generate the canonical A/B trial plan (T1.6).

The pilot's "canonical-10-trials.yaml" claimed 2x4x10 but held 10 trials total
in a fixed hybrid-first order while its header claimed pre-randomisation. This
generator produces the real pre-registered design, reproducibly:

  - 2 controllers x 4 workloads x N pairs (default 10)  = 8N paired trials
  - plus A/A pairs (HPA vs HPA, burst) as the measurement-noise floor
  - within-pair order counterbalanced (half hybrid-first), seeded
  - burst pairs + A/A first (the minimum viable confirmatory core),
    remaining cells shuffled after them
  - phase-aligned measurement offsets per workload (DEV-014)

Usage:
    uv run python experiments/trial_plans/generate_plan.py \
        [--pairs 10] [--aa-pairs 2] [--seed 42] [--peak-users 500] \
        [--probe-rate 100] [--out experiments/trial_plans/canonical-v2.yaml]

Re-run with new --peak-users after the capacity probe (T4.1/T4.2); the file
header records the exact generation command for provenance.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Workload timing: locust runs the FULL profile; measurement starts at the
# phase under study (offset) and runs for wrk2_duration_seconds.
WORKLOADS = {
    "burst": dict(
        locust_duration_seconds=720, locust_spawn_rate=50,
        measure_start_offset_seconds=300, wrk2_duration_seconds=120,
        env=lambda peak: {
            "BASELINE_USERS": "50", "PEAK_USERS": str(peak),
            "WARMUP_SECONDS": "300", "BURST_SECONDS": "120",
            "COOLDOWN_SECONDS": "300",
        },
        locust_file="experiments/workloads/burst.py",
    ),
    "ramp": dict(
        locust_duration_seconds=1500, locust_spawn_rate=1,
        measure_start_offset_seconds=600, wrk2_duration_seconds=300,
        env=lambda peak: {
            "FLOOR_USERS": "50", "PEAK_USERS": str(peak),
            "RAMP_SECONDS": "600", "HOLD_SECONDS": "300",
        },
        locust_file="experiments/workloads/ramp.py",
    ),
    "periodic": dict(
        locust_duration_seconds=1800, locust_spawn_rate=50,
        measure_start_offset_seconds=0, wrk2_duration_seconds=600,
        env=lambda peak: {
            "FLOOR_USERS": "50", "PEAK_USERS": str(peak),
            "PERIOD_SECONDS": "600", "NUM_PERIODS": "3",
        },
        locust_file="experiments/workloads/periodic.py",
    ),
    "trace_replay": dict(
        locust_duration_seconds=1800, locust_spawn_rate=50,
        measure_start_offset_seconds=0, wrk2_duration_seconds=1800,
        env=lambda peak: {
            "TRACE_FILE": "experiments/traces/alibaba_rps_30m.csv",
            "FLOOR_USERS": "50", "PEAK_USERS": str(peak),
        },
        locust_file="experiments/workloads/trace_replay.py",
    ),
}


def trial(trial_id, autoscaler, workload, seed, peak, probe_rate):
    w = WORKLOADS[workload]
    return {
        "trial_id": trial_id,
        "autoscaler": autoscaler,
        "workload": workload,
        "seed": seed,
        "locust_file": w["locust_file"],
        "locust_users": peak,
        "locust_spawn_rate": w["locust_spawn_rate"],
        "locust_duration_seconds": w["locust_duration_seconds"],
        "measure_start_offset_seconds": w["measure_start_offset_seconds"],
        "wrk2_threads": 4,
        "wrk2_connections": 100,
        "wrk2_rate": probe_rate,
        "wrk2_duration_seconds": w["wrk2_duration_seconds"],
        "wrk2_url": "http://localhost:30080/",
        "env": w["env"](peak),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--aa-pairs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peak-users", type=int, default=500)
    ap.add_argument("--probe-rate", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/trial_plans/canonical-v2.yaml")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seed_counter = iter(range(10_000, 99_999))

    def make_pair(workload, pair_i, arms):
        first, second = arms
        if rng.random() < 0.5:          # counterbalance within-pair order
            first, second = second, first
        return [
            trial(f"{workload}-{a}-{pair_i:03d}", a, workload,
                  next(seed_counter), args.peak_users, args.probe_rate)
            for a in (first, second)
        ]

    # Priority block: burst pairs (the minimum viable confirmatory core)…
    burst_pairs = [make_pair("burst", i + 1, ("hybrid", "hpa"))
                   for i in range(args.pairs)]
    rng.shuffle(burst_pairs)
    # …plus the A/A noise floor (HPA vs HPA under identical burst conditions).
    aa_pairs = []
    for i in range(args.aa_pairs):
        pair = [trial(f"aa-burst-hpa{a}-{i + 1:03d}", "hpa", "burst",
                      next(seed_counter), args.peak_users, args.probe_rate)
                for a in ("A", "B")]
        aa_pairs.append(pair)

    # Remaining cells, pair order shuffled across workloads so an early stop
    # still leaves roughly balanced cells.
    rest = []
    for workload in ("ramp", "periodic", "trace_replay"):
        rest += [make_pair(workload, i + 1, ("hybrid", "hpa"))
                 for i in range(args.pairs)]
    rng.shuffle(rest)

    ordered = burst_pairs + aa_pairs + rest
    trials = [t for pair in ordered for t in pair]

    import yaml
    plan = {
        "name": "canonical-ab-v2",
        "description": (
            f"2x4x{args.pairs} A/B + {args.aa_pairs} A/A burst pairs. "
            f"Generated by generate_plan.py --seed {args.seed} "
            f"--peak-users {args.peak_users} --probe-rate {args.probe_rate}. "
            "Within-pair order counterbalanced; burst+A/A first (priority core), "
            "remaining cells shuffled. Measurement offsets per DEV-014."
        ),
        "result_dir": "experiments/results",
        "n_trials_per_cell": args.pairs,
        "hpa_manifest": "experiments/hpa/frontend-hpa.yaml",
        "controller_config": "controller/configs/ab/frontend-ab.yaml",
        "model_registry":
            "experiments/results/phase3_20260703T090249Z_model_registry.yaml",
        "stabilisation_wait_seconds": 180,
        "trials": trials,
    }
    header = (
        "# canonical-v2.yaml — GENERATED, do not hand-edit.\n"
        f"# Regenerate: uv run python experiments/trial_plans/generate_plan.py "
        f"--pairs {args.pairs} --aa-pairs {args.aa_pairs} --seed {args.seed} "
        f"--peak-users {args.peak_users} --probe-rate {args.probe_rate}\n"
        f"# {len(trials)} trials: {args.pairs} pairs/cell x 4 workloads "
        f"+ {args.aa_pairs} A/A burst pairs.\n"
        "# NOTE: model_registry is re-frozen from VM telemetry before the real "
        "run (DEV-018);\n# regenerate after T3.2 if the registry filename "
        "changes, and after T4.2 with the\n# calibrated --peak-users.\n"
    )
    args.out.write_text(header + yaml.safe_dump(plan, sort_keys=False, width=100))
    print(f"wrote {args.out} ({len(trials)} trials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
