"""Per-decision evidence-bundle writer (§3.8).

§3.8 specifies six evidence-bundle items per decision:
    1. metric window used by the forecaster
    2. forecast point estimate + prediction interval
    3. decision engine output (recommended replicas, state, rate-limit, fallback)
    4. per-decision SHAP attribution vector (top-k)
    5. faithfulness metrics for that attribution
    6. grounded LLM narrative

Phase 1 persists items 1–3 (the controller-side evidence). Phase 6 adds 4–5
(SHAP), Phase 7 adds 6 (narrative).

The store format is JSONL — one decision per line, append-only, atomic at the
line level, trivial to grep/jq/aggregate. JSONL is also the simplest format
to publish as part of the §3.10 reproducibility package.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .state import Decision


class EvidenceBundleWriter:
    """Append-only JSONL writer with a per-instance lock.

    Thread-safe within a process; for cross-process safety, run one controller
    per service (the standard deployment model anyway).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        decision: Decision,
        history: Optional[pd.Series] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one evidence-bundle row for `decision`.

        Parameters
        ----------
        decision : Decision
            The §3.7 decision result.
        history : optional pd.Series
            The metric window the forecaster used. Summarised (first/last
            timestamps, sample count, descriptive stats) — raw series is
            persisted separately to Parquet in Phase 2.
        extra : optional mapping
            Free-form additional fields (e.g. fallback explanation text).
            Reserved for Phase 6/7 to attach SHAP and LLM narrative items.
        """
        row: Dict[str, Any] = decision.model_dump(mode="json")
        if history is not None:
            row["feature_window_summary"] = _summarise(history)
        if extra:
            row.update(dict(extra))

        line = json.dumps(row, sort_keys=True, default=_json_default)
        with self._lock:
            # Append + flush so each line is durable as soon as it is written.
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    # ------------------------------------------------------------------ #
    # Read-back (for tests and Phase 6/7 offline processing)              #
    # ------------------------------------------------------------------ #
    def read_all(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
def _summarise(series: pd.Series) -> Dict[str, Any]:
    """Compact summary of a metric window. Phase 2 persists the raw series."""
    clean = series.dropna()
    if len(clean) == 0:
        return {"n": 0}
    s: Dict[str, Any] = {
        "n": int(len(clean)),
        "first_ts": clean.index[0].isoformat() if hasattr(clean.index[0], "isoformat") else None,
        "last_ts": clean.index[-1].isoformat() if hasattr(clean.index[-1], "isoformat") else None,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "last": float(clean.iloc[-1]),
    }
    return s


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
