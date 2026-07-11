"""Per-service feature engineering.

§3.5 specifies, for each service and each scrape interval, that the
following features are computed from past values only:

    * lag values at 1, 2, 3, 5, 10 scrape intervals (15, 30, 45, 75, 150 s)
    * rolling means and variances at 30, 60, 120 s windows
    * first difference and log first difference

Every transform here uses pandas `shift(k>0)` and `rolling(..., closed='right')`
with explicit `min_periods` so that features at time t depend strictly on
values at times ≤ t. The opening rows (before history is long enough) are
left as NaN — downstream model trainers drop them before fitting.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schema import FeatureSchema

# §3.5 defaults (single source of truth)
DEFAULT_LAG_INTERVALS: List[int] = [1, 2, 3, 5, 10]
DEFAULT_ROLLING_WINDOWS_SEC: List[int] = [30, 60, 120]


def engineer_features(
    wide: pd.DataFrame,
    base_column: str = "cpu",
    *,
    sample_interval_seconds: int = 15,
    lag_intervals: Sequence[int] = DEFAULT_LAG_INTERVALS,
    rolling_windows_seconds: Sequence[int] = DEFAULT_ROLLING_WINDOWS_SEC,
    diff_first: bool = True,
    diff_first_log: bool = True,
) -> pd.DataFrame:
    """Compute the §3.5 feature set on a wide per-service DataFrame.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide per-service frame from `data.synthetic.to_wide` (or the live
        exporter pipeline). Must contain a `timestamp` column and the
        `base_column` to be featurised.
    base_column : str
        Which column to featurise. Default `"cpu"` per §3.5 first metric
        family.
    sample_interval_seconds : int
        Scrape cadence (§3.5: 15 s).
    lag_intervals, rolling_windows_seconds, diff_first, diff_first_log
        Override the §3.5 defaults only with explicit justification.

    Returns
    -------
    pd.DataFrame
        Wide frame with the original columns plus the engineered features.
        Sorted by timestamp ascending. Feature columns at the opening of
        the series are NaN where the lookback would dip below t = 0.
    """
    if "timestamp" not in wide.columns:
        raise ValueError("wide frame must include a 'timestamp' column")
    if base_column not in wide.columns:
        raise ValueError(f"base_column '{base_column}' not in frame")

    df = wide.sort_values("timestamp").reset_index(drop=True).copy()
    base = df[base_column].astype(float)

    # ─── Lags ───────────────────────────────────────────────────────────
    # `series.shift(k)` with k > 0 is past-only by construction.
    for k in lag_intervals:
        if k <= 0:
            raise ValueError(f"lag interval must be ≥ 1, got {k}")
        df[f"{base_column}_lag{k}"] = base.shift(k)

    # ─── Rolling means + variances ──────────────────────────────────────
    # closed='right' includes the value at t and excludes the right-open
    # endpoint — for our purposes (window covers samples in [t-w+1, t])
    # this is the safe past-only choice.
    for w_sec in rolling_windows_seconds:
        n = w_sec // sample_interval_seconds
        if n <= 0:
            raise ValueError(
                f"rolling window {w_sec}s is below sample interval "
                f"{sample_interval_seconds}s"
            )
        roll = base.rolling(window=n, min_periods=n)
        df[f"{base_column}_rmean{w_sec}s"] = roll.mean()
        df[f"{base_column}_rvar{w_sec}s"] = roll.var(ddof=1)

    # ─── Differences ────────────────────────────────────────────────────
    # `diff(1)` = x(t) − x(t−1): past-only.
    if diff_first:
        df[f"{base_column}_diff1"] = base.diff(1)
    if diff_first_log:
        # log(x(t)+ε) − log(x(t−1)+ε) — stable for ≥ 0 series like CPU/RPS.
        eps = 1e-9
        log_base = np.log(base.clip(lower=0.0) + eps)
        df[f"{base_column}_logdiff1"] = log_base.diff(1)

    return df


def feature_schema_for(
    service: str,
    base_column: str = "cpu",
    upstream_services: Optional[List[str]] = None,
    sample_interval_seconds: int = 15,
) -> FeatureSchema:
    """Build a FeatureSchema with the §3.5 defaults for one service."""
    from ..schema import MetricFamily

    return FeatureSchema(
        service=service,
        metric_family=MetricFamily.CPU,           # base is CPU per §3.5 first family
        base_column=base_column,
        lag_intervals=DEFAULT_LAG_INTERVALS,
        rolling_windows_seconds=DEFAULT_ROLLING_WINDOWS_SEC,
        diff_first=True,
        diff_first_log=True,
        upstream_services=upstream_services or [],
        sample_interval_seconds=sample_interval_seconds,
    )
