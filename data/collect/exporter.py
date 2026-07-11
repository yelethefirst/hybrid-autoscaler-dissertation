"""Prometheus → Parquet exporter (§3.5).

Pulls all five metric families for a set of services across a time range
and writes long-format Parquet in 5-minute batches per §3.5. Designed to
be testable without a live Prometheus: the `client` parameter is duck-typed
so a fake can be substituted in unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Protocol

import pandas as pd

from ..schema import SCHEMA_VERSION
from .queries import build_queries

LOGGER = logging.getLogger(__name__)


class PromRangeClient(Protocol):
    """Minimal Prometheus interface the exporter relies on.

    Implementations: `prometheus_api_client.PrometheusConnect` in production;
    `tests.fakes.FakePromClient` in tests.
    """

    def custom_query_range(
        self, query: str, start_time: datetime, end_time: datetime, step: str
    ) -> list: ...


@dataclass(frozen=True)
class CampaignConfig:
    services: List[str]
    namespace: str = "default"
    sample_interval_seconds: int = 15           # §3.5 cadence
    batch_seconds: int = 5 * 60                 # §3.5 5-minute batches
    output_dir: Path = Path("data/parquet")


class PrometheusExporter:
    """Thin wrapper over a PromRangeClient that emits §3.5 long-format rows."""

    def __init__(self, client: PromRangeClient):
        self.client = client

    def fetch(
        self,
        service: str,
        namespace: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> pd.DataFrame:
        """Fetch all five metric families for one service over [start, end)."""
        rows: List[dict] = []
        step_str = f"{step_seconds}s"
        for family, name, query in build_queries(service, namespace):
            try:
                result = self.client.custom_query_range(
                    query=query, start_time=start, end_time=end, step=step_str
                )
            except Exception:
                LOGGER.exception("query failed for %s/%s — skipping", service, name)
                continue
            for series in result or []:
                labels = dict(series.get("metric", {}))
                # The exporter row "owns" service/namespace explicitly; the
                # rest of the labels are forwarded as the labels map.
                for ts_str, val_str in series.get("values", []):
                    try:
                        ts = datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
                        val = float(val_str)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "timestamp": ts,
                        "service": service,
                        "namespace": namespace,
                        "metric_family": family.value,
                        "metric_name": name,
                        "value": val,
                        "labels": labels,
                    })
        df = pd.DataFrame(rows)
        df.attrs["schema_version"] = SCHEMA_VERSION
        return df


def run_campaign(
    exporter: PrometheusExporter,
    config: CampaignConfig,
    start: datetime,
    end: datetime,
    *,
    on_batch_written: Optional[callable] = None,
) -> List[Path]:
    """Run a §3.5 collection campaign and write Parquet in 5-minute batches.

    Returns the list of written Parquet paths.
    """
    if end <= start:
        raise ValueError("end must be after start")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    batch_start = start
    while batch_start < end:
        batch_end = min(batch_start + timedelta(seconds=config.batch_seconds), end)
        per_service_frames: List[pd.DataFrame] = []
        for service in config.services:
            df = exporter.fetch(
                service=service,
                namespace=config.namespace,
                start=batch_start,
                end=batch_end,
                step_seconds=config.sample_interval_seconds,
            )
            if not df.empty:
                per_service_frames.append(df)
        if per_service_frames:
            batch = pd.concat(per_service_frames, ignore_index=True)
            out = config.output_dir / (
                f"telemetry_{batch_start.strftime('%Y%m%dT%H%M%S')}"
                f"_{batch_end.strftime('%Y%m%dT%H%M%S')}.parquet"
            )
            batch.to_parquet(out, index=False)
            written.append(out)
            LOGGER.info("wrote %d rows → %s", len(batch), out)
            if on_batch_written:
                on_batch_written(out, batch)
        else:
            LOGGER.warning("no data for batch %s..%s — skipping write", batch_start, batch_end)
        batch_start = batch_end

    return written


def load_campaign(parquet_dir: Path) -> pd.DataFrame:
    """Load all Parquet batches in a directory and return a single long DataFrame."""
    parquet_dir = Path(parquet_dir)
    paths = sorted(parquet_dir.glob("telemetry_*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["service", "metric_family", "metric_name", "timestamp"])
    return out.reset_index(drop=True)
