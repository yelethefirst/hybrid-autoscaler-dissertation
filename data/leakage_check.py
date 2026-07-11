"""Empirical anti-leakage validator (§3.5 absolute rule).

§3.5: "The anti-leakage rule is absolute: no future value, including any
feature derived from a future value, may appear in a training input."

The validator works empirically rather than by convention: it takes a
feature-engineering function, applies it to a series, then perturbs the
future portion of the series and re-applies. If any feature value at time
t ≤ split changes when only values at t > split were perturbed, that
feature has leaked the future. The test is deterministic given a seed and
runs in a few milliseconds.

Use this both as a unit test (built-in tests in `tests/test_leakage_check.py`)
and as a self-check that runs at the end of every Phase 2 collection
campaign — see `bin/run-features.sh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageReport:
    """Outcome of a leakage check."""

    passed: bool
    leaked_columns: List[str]
    n_rows_checked: int
    split_index: int

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return (
                f"✓ no leakage detected over {self.n_rows_checked} rows "
                f"(split at index {self.split_index})"
            )
        return (
            f"✗ LEAKAGE: {len(self.leaked_columns)} feature(s) changed when only "
            f"future values were perturbed: {self.leaked_columns}"
        )


# Type alias for a pipeline that produces a featurised DataFrame from a wide one.
FeaturePipeline = Callable[[pd.DataFrame], pd.DataFrame]


def check_no_leakage(
    pipeline: FeaturePipeline,
    wide: pd.DataFrame,
    base_columns: List[str],
    *,
    split_fraction: float = 0.5,
    perturbation_passes: int = 3,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> LeakageReport:
    """Empirically check that `pipeline` does not leak future values.

    Procedure
    ---------
    1. Featurise the original `wide` frame. Call this `baseline`.
    2. For each pass:
        a. Shuffle/perturb the values in `base_columns` *only* at rows with
           index ≥ split (the "future" portion).
        b. Featurise the perturbed frame. Call this `perturbed`.
        c. For every engineered column not in `base_columns + ['timestamp']`,
           compare baseline[col][:split] vs perturbed[col][:split].
        d. Any column whose past-portion values differ has leaked.

    A passing report is **necessary but not sufficient** to prove the
    pipeline is causal — only a formal proof can do that — but it is a
    strong empirical guard against the common mistakes (negative shifts,
    centred rolling windows, forward-looking diffs).
    """
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be in (0, 1)")
    if perturbation_passes < 1:
        raise ValueError("perturbation_passes must be ≥ 1")

    df = wide.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * split_fraction)
    if split < 2 or split >= len(df) - 1:
        raise ValueError(
            f"split index {split} too close to bounds for n={len(df)} rows"
        )

    baseline = pipeline(df)
    rng = np.random.default_rng(seed)

    # Engineered columns = baseline columns minus the raw inputs and timestamp.
    raw_cols = set(base_columns) | {"timestamp"}
    engineered_cols = [c for c in baseline.columns if c not in raw_cols]
    if not engineered_cols:
        # Nothing was engineered; nothing can leak.
        return LeakageReport(
            passed=True, leaked_columns=[], n_rows_checked=split, split_index=split
        )

    leaked: set[str] = set()
    for _ in range(perturbation_passes):
        perturbed_df = df.copy()
        for col in base_columns:
            if col not in perturbed_df.columns:
                continue
            future_vals = perturbed_df.loc[split:, col].to_numpy(copy=True)
            # Two complementary perturbations: random permutation + heavy multiplicative noise.
            rng.shuffle(future_vals)
            future_vals = future_vals * rng.uniform(0.1, 10.0, size=len(future_vals))
            perturbed_df.loc[split:, col] = future_vals

        perturbed = pipeline(perturbed_df)

        for col in engineered_cols:
            if col not in perturbed.columns:
                # Pipeline produced a different schema — that's its own bug.
                leaked.add(col)
                continue
            a = baseline[col].iloc[:split].to_numpy(dtype=float)
            b = perturbed[col].iloc[:split].to_numpy(dtype=float)
            # NaN equality: treat NaNs as equal so opening-window NaNs don't false-flag.
            mask = ~(np.isnan(a) & np.isnan(b))
            if not np.allclose(a[mask], b[mask], rtol=rtol, atol=atol, equal_nan=False):
                leaked.add(col)

    return LeakageReport(
        passed=not leaked,
        leaked_columns=sorted(leaked),
        n_rows_checked=split,
        split_index=split,
    )


def assert_no_leakage(
    pipeline: FeaturePipeline,
    wide: pd.DataFrame,
    base_columns: List[str],
    *,
    split_fraction: float = 0.5,
    seed: int = 0,
) -> None:
    """Raise AssertionError if any future leakage is detected. CI-friendly."""
    report = check_no_leakage(
        pipeline, wide, base_columns,
        split_fraction=split_fraction, seed=seed,
    )
    if not report.passed:
        raise AssertionError(report.summary())
