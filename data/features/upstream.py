"""Upstream request-rate exogenous feature (§3.5).

§3.5: "Upstream service request rate is included as an exogenous feature
for downstream services, partially capturing the cross-service coupling
identified by Hu et al. (2022)."

For a downstream service S with upstream U, this adds the column
`{U}_request_rate_t` aligned by timestamp. The value at time t is the
upstream RPS measured at time t (not shifted into the future).
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ..synthetic import to_wide


def add_upstream_request_rate(
    target_wide: pd.DataFrame,
    long_telemetry: pd.DataFrame,
    upstream_services: List[str],
) -> pd.DataFrame:
    """Join upstream services' request_rate onto a target wide frame.

    Parameters
    ----------
    target_wide : pd.DataFrame
        Wide per-service frame produced by `data.synthetic.to_wide` (or the
        equivalent live exporter pipeline).
    long_telemetry : pd.DataFrame
        The full long-format telemetry that includes the upstream services'
        rows. Used to extract their request rate.
    upstream_services : list[str]
        Names of upstream Deployments to attach.

    Returns
    -------
    pd.DataFrame
        Copy of `target_wide` with one additional column per upstream:
        `{upstream}_request_rate_t`.
    """
    if "timestamp" not in target_wide.columns:
        raise ValueError("target_wide must include 'timestamp'")

    out = target_wide.copy()
    for upstream in upstream_services:
        upstream_wide = to_wide(long_telemetry, service=upstream)
        if "request_rate" not in upstream_wide.columns:
            # The upstream service had no request_rate samples — leave the
            # column as NaN so downstream training treats it as missing
            # rather than silently zero.
            out[f"{upstream}_request_rate_t"] = float("nan")
            continue
        rps = upstream_wide[["timestamp", "request_rate"]].rename(
            columns={"request_rate": f"{upstream}_request_rate_t"}
        )
        out = out.merge(rps, on="timestamp", how="left")
    return out
