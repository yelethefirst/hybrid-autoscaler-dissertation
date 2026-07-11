#!/usr/bin/env bash
#
# run-features.sh — Phase 2 feature engineering + Phase 2 EXIT CRITERION check.
#
# Reads the most recent telemetry Parquet under data/parquet/, runs the
# §3.5 feature pipeline per service, verifies the anti-leakage rule
# empirically, prints a per-service sample count, and writes the resulting
# featurised Parquet under data/features/.
#
# Exits non-zero if leakage is detected — this is the Phase 2 exit criterion.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TELEM_DIR="${TELEM_DIR:-$REPO_ROOT/data/parquet}"
FEAT_DIR="${FEAT_DIR:-$REPO_ROOT/data/features_out}"
mkdir -p "$FEAT_DIR"

cd "$REPO_ROOT"

uv run python -c "
import sys
from pathlib import Path
import pandas as pd
from data.synthetic import to_wide
from data.features import engineer_features, add_upstream_request_rate
from data.features.engineer import feature_schema_for
from data.leakage_check import check_no_leakage
from data.synthetic import DEFAULT_TOPOLOGY

telem_dir = Path('$TELEM_DIR')
feat_dir = Path('$FEAT_DIR')

files = sorted(telem_dir.glob('*.parquet'), key=lambda p: p.stat().st_mtime)
if not files:
    print(f'❌  no parquet files in {telem_dir} — run bin/run-collection.sh first')
    sys.exit(1)

latest = files[-1]
print(f'▶  source: {latest}')
df_long = pd.read_parquet(latest)

services_in_data = sorted(df_long['service'].unique())
topology_by_name = {s.name: s for s in DEFAULT_TOPOLOGY}

failed = False
sample_counts = {}
for svc in services_in_data:
    wide = to_wide(df_long, service=svc)
    if 'cpu' not in wide.columns or wide.empty:
        print(f'   ⚠  skip {svc}: no cpu column or empty')
        continue
    upstream = topology_by_name[svc].upstream if svc in topology_by_name else []
    wide_with_upstream = add_upstream_request_rate(wide, df_long, upstream)

    feats = engineer_features(wide_with_upstream, base_column='cpu')

    # Anti-leakage check (Phase 2 exit criterion)
    rep = check_no_leakage(
        lambda d: engineer_features(d, base_column='cpu'),
        wide,
        base_columns=['cpu'],
        seed=0,
    )
    if not rep.passed:
        failed = True
        print(f'   ❌ {svc}: {rep.summary()}')
        continue

    out_path = feat_dir / f'features_{svc}.parquet'
    feats.to_parquet(out_path, index=False)
    sample_counts[svc] = len(feats)
    print(f'   ✓  {svc}: {len(feats)} samples, {feats.shape[1]} columns → {out_path.name}')

print('')
print('Sample counts per service:', sample_counts)

if failed:
    print('')
    print('❌ Phase 2 exit criterion FAILED — leakage detected (see above)')
    sys.exit(1)

print('')
print('✅ Phase 2 exit criterion MET — no leakage detected, features written')
"
