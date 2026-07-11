"""Phase 5 trial supervisor — §3.9 A/B experiment harness.

Orchestrates a sequence of TrialSpec entries from a TrialPlan YAML:

For each trial:
    1. Stabilise cluster: ensure autoscaler from previous trial is removed
       and replica count returns to 1 within stabilisation_wait_seconds.
    2. Install autoscaler (HPA via kubectl apply, or Hybrid via subprocess).
    3. Run Locust headless (background subprocess) to generate load.
    4. Run wrk2 (foreground) for coordinated-omission-corrected latency.
    5. Stop Locust.
    6. Remove autoscaler.
    7. Collect Prometheus metrics (replica-seconds, peak replicas).
    8. Parse wrk2 output for p50/p95/p99 latency, throughput, error rate.
    9. Write TrialResult to JSONL.

Usage:
    uv run python -m experiments.supervisor \\
        --plan experiments/trial_plans/canonical-10-trials.yaml \\
        [--dry-run]   # log actions without executing subprocess or kubectl

Design notes:
    - Each subprocess is given a unique log file so nothing is lost even if
      the supervisor crashes mid-trial.
    - The JSONL result log is flushed after every trial so partial runs are
      recoverable.
    - Randomised trial order is fixed by the seed in each TrialSpec; the
      ORDER of TrialSpecs in the YAML must be pre-randomised by the researcher
      (or use --shuffle with a fixed seed).
    - wrk2 must be on PATH; install with: brew install wrk2 (macOS) or
      build from https://github.com/giltene/wrk2.
      Fallback: if wrk2 is absent, `hey` (brew install hey) is used instead.
      hey does not apply coordinated-omission correction — noted as DEV-013.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import requests

from .trial_schema import Autoscaler, TrialPlan, TrialResult, TrialSpec

LOGGER = logging.getLogger(__name__)

# Prometheus base URL (port-forwarded; must be running before the supervisor starts).
PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
# Online Boutique namespace
OB_NS = os.getenv("OB_NAMESPACE", "default")


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prom_query(query: str) -> Optional[float]:
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
        r.raise_for_status()
        result = r.json()["data"]["result"]
        if result:
            return float(result[0]["value"][1])
    except Exception as exc:
        LOGGER.warning("Prometheus query failed (%s): %s", query, exc)
    return None


def _prom_range_query(query: str, start: datetime, end: datetime, step: str = "15s") -> Optional[float]:
    """Return the average of a range-query result (used for replica-seconds)."""
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "step": step,
            },
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()["data"]["result"]
        if not result:
            return None
        values = [float(v[1]) for series in result for v in series["values"]]
        return sum(values) * 15 if values else None  # replica × step = replica-seconds
    except Exception as exc:
        LOGGER.warning("Prometheus range query failed: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Autoscaler install / remove
# ─────────────────────────────────────────────────────────────────────────────

def _install_hpa(manifest: Path, dry_run: bool) -> None:
    cmd = ["kubectl", "-n", OB_NS, "apply", "-f", str(manifest)]
    LOGGER.info("installing HPA: %s", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _remove_hpa(dry_run: bool) -> None:
    cmd = ["kubectl", "-n", OB_NS, "delete", "hpa", "frontend", "--ignore-not-found=true"]
    LOGGER.info("removing HPA: %s", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _start_hybrid(
    spec: TrialSpec,
    plan: "TrialPlan",
    log_path: Path,
    dry_run: bool,
    evidence_path: Optional[Path] = None,
) -> Optional[subprocess.Popen]:
    config = spec.controller_config or plan.controller_config
    if config is None:
        raise ValueError(f"trial {spec.trial_id}: hybrid autoscaler requires controller_config (set in spec or plan)")
    registry = spec.model_registry or plan.model_registry
    cmd = [
        "uv", "run", "python", "-m", "controller.main",
        "--config", str(config),
        "--prometheus-url", PROM_URL,
    ]
    if registry:
        cmd += ["--model-registry", str(registry)]
    if evidence_path is not None:
        # Explicit CLI arg — the pilot set an EVIDENCE_PATH env var the
        # controller never read, so per-trial evidence silently went to the
        # shared config path (DEV-016).
        cmd += ["--evidence-path", str(evidence_path)]
    LOGGER.info("starting hybrid controller: %s", " ".join(cmd))
    if dry_run:
        return None
    env = {**os.environ, "DRY_RUN": ""}
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc


def _stop_hybrid(proc: Optional[subprocess.Popen], dry_run: bool) -> None:
    if proc is None or dry_run:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# ─────────────────────────────────────────────────────────────────────────────
# Workload (Locust)
# ─────────────────────────────────────────────────────────────────────────────

def _start_locust(spec: TrialSpec, log_path: Path, dry_run: bool) -> Optional[subprocess.Popen]:
    env = {**os.environ, **spec.env, "PYTHONUNBUFFERED": "1"}
    # Trial seed reaches the workload: user.py seeds `random` from TRIAL_SEED
    # so the request mix (product choices, quantities) is reproducible per
    # trial rather than seed being recorded-but-unused metadata.
    env["TRIAL_SEED"] = str(spec.seed)
    csv_prefix = spec.locust_file.parent.parent / "results" / f"{spec.trial_id}-locust"
    env["LOCUST_CSV"] = str(csv_prefix)
    # The shape file must be loaded together with user.py: shapes carry no
    # BoutiqueUser of their own (a relative import crashes under Locust's
    # standalone module loading — DEV-009/DEV-017).
    user_file = spec.locust_file.parent / "user.py"
    cmd = [
        "uv", "run", "locust",
        "-f", f"{spec.locust_file},{user_file}",
        "--host", spec.locust_host,
        "--headless",
        "--users", str(spec.locust_users),
        "--spawn-rate", str(spec.locust_spawn_rate),
        "-t", f"{spec.locust_duration_seconds}s",
        "--csv", str(csv_prefix),
    ]
    LOGGER.info("starting Locust: %s", " ".join(cmd))
    if dry_run:
        return None
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return proc


def _assert_locust_alive(proc: Optional[subprocess.Popen], log_path: Path, dry_run: bool) -> None:
    """Fail the trial loudly if Locust died during startup (DEV-017).

    The pilot A/B ran all 10 trials with Locust crashed at import time and the
    supervisor never noticed. Any workload-generator death now aborts the trial.
    """
    if dry_run or proc is None:
        return
    time.sleep(8)  # long enough for import errors / config errors to surface
    if proc.poll() is not None:
        tail = ""
        try:
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-15:])
        except OSError:
            pass
        raise RuntimeError(
            f"Locust exited during startup (rc={proc.returncode}). "
            f"Log tail:\n{tail}"
        )


def _stop_locust(proc: Optional[subprocess.Popen], dry_run: bool) -> None:
    if proc is None or dry_run:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


# ─────────────────────────────────────────────────────────────────────────────
# Measurement (wrk2)
# ─────────────────────────────────────────────────────────────────────────────

def _run_wrk2(spec: TrialSpec, log_path: Path, dry_run: bool) -> tuple[str, str]:
    """Run wrk2 (required) or hey (opt-in fallback) and return (raw_output, tool_used).

    wrk2 is the registered measurement tool (§3.9/§3.10-A). Falling back to hey
    silently would invalidate the CO-correction claim for the whole campaign, so
    the fallback must be requested explicitly with ALLOW_HEY_FALLBACK=1
    (pilot/dev use only — DEV-013).
    """
    if dry_run:
        # No tool detection in dry-run: a hey-only dev machine must not trip
        # the strict wrk2 requirement while rehearsing a plan.
        return "(dry-run: measurement not executed)", "wrk2"
    if shutil.which("wrk2"):
        tool = "wrk2"
        cmd = [
            "wrk2",
            f"-t{spec.wrk2_threads}",
            f"-c{spec.wrk2_connections}",
            f"-d{spec.wrk2_duration_seconds}s",
            f"-R{spec.wrk2_rate}",
            "--latency",
            spec.wrk2_url,
        ]
    elif shutil.which("hey") and os.getenv("ALLOW_HEY_FALLBACK") == "1":
        tool = "hey"
        # hey uses Go flag package: flags and values must be separate args ("-c 100" not "-c100")
        rate_per_worker = max(1, spec.wrk2_rate // spec.wrk2_connections)
        cmd = [
            "hey",
            "-c", str(spec.wrk2_connections),
            "-z", f"{spec.wrk2_duration_seconds}s",
            "-q", str(rate_per_worker),
            spec.wrk2_url,
        ]
        LOGGER.warning("wrk2 not found; ALLOW_HEY_FALLBACK=1 set — using hey "
                       "(no coordinated-omission correction, DEV-013; pilot/dev only)")
    else:
        raise RuntimeError(
            "wrk2 not found on PATH. The final campaign requires wrk2 (§3.10-A); "
            "install it (bootstrap-ubuntu.sh) or set ALLOW_HEY_FALLBACK=1 for a "
            "pilot/dev run with hey (results then carry the DEV-013 caveat)."
        )

    LOGGER.info("running %s: %s", tool, " ".join(cmd))
    if dry_run:
        return f"(dry-run: {tool} not executed)", tool
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=spec.wrk2_duration_seconds + 30)
    output = result.stdout + result.stderr
    log_path.write_text(output)
    if result.returncode != 0:
        # A failed measurement run must fail the trial, not produce a row
        # with error=None and empty latencies (2026-07-05 code review).
        raise RuntimeError(
            f"{tool} exited rc={result.returncode}; output tail: {output[-500:]}"
        )
    return output[:4096], tool


def _parse_wrk2(raw: str, tool: str = "wrk2") -> dict:
    """Extract latency percentiles and throughput from wrk2 or hey output."""
    if tool == "hey":
        return _parse_hey(raw)
    metrics: dict = {}

    # Pass 1: compact summary section ("50.000%    0.96m", "99.000%   45ms", etc.)
    # wrk2 adapts the unit to the magnitude: us / ms / s / m (minutes).
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
            # ms: no conversion
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

    # Pass 2: detailed HdrHistogram section — values are in MICROSECONDS.
    # Used to extract p95 (absent from compact summary) and to back-fill
    # any percentile the compact pass missed (e.g. when unit was "m").
    # Format: "  100270.079     0.950000         1635        20.00"
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
        val_ms = float(m.group(1)) / 1000.0   # us → ms
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


def _parse_hey(raw: str) -> dict:
    """Extract latency percentiles and throughput from hey output."""
    metrics: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        # "  50%% in 0.0034 secs"  (hey uses Go fmt which outputs %% as literal %)
        m = re.match(r"(\d+)%%?\s+in\s+([\d.]+)\s+secs", line)
        if m:
            pct = int(m.group(1))
            val_ms = float(m.group(2)) * 1000
            if pct == 50:
                metrics["p50_latency_ms"] = val_ms
            elif pct == 95:
                metrics["p95_latency_ms"] = val_ms
            elif pct == 99:
                metrics["p99_latency_ms"] = val_ms
        # "Requests/sec:  1234.56"
        if "Requests/sec:" in line:
            try:
                metrics["throughput_rps"] = float(line.split()[-1].replace(",", ""))
            except ValueError:
                pass
        # hey reports errors as "[500] 12 responses"
        m2 = re.match(r"\[([45]\d\d)\]\s+(\d+)\s+responses", line)
        if m2:
            metrics["errors_total"] = metrics.get("errors_total", 0) + int(m2.group(2))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Stabilisation
# ─────────────────────────────────────────────────────────────────────────────

def _reset_baseline(target_replicas: int, timeout_seconds: int, dry_run: bool) -> None:
    """Actively scale frontend to the baseline and BLOCK until it is there.

    The pilot only observed readiness, ignored the result, and never scaled
    down — later trials inherited replicas from earlier ones (DEV-015).
    A trial that cannot reach a clean baseline must not run.
    """
    if dry_run:
        LOGGER.info("DRY-RUN: skipping baseline reset")
        return
    subprocess.run(
        ["kubectl", "-n", OB_NS, "scale", "deployment/frontend",
         f"--replicas={target_replicas}"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    if not _wait_stable(target_replicas, timeout_seconds, dry_run):
        raise RuntimeError(
            f"baseline reset failed: frontend did not reach "
            f"{target_replicas} ready replica(s) in {timeout_seconds}s"
        )


def _wait_stable(target_replicas: int, timeout_seconds: int, dry_run: bool) -> bool:
    """Poll until frontend Deployment has target_replicas ready, or timeout."""
    if dry_run:
        LOGGER.info("DRY-RUN: skipping stabilisation wait")
        return True
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(
                ["kubectl", "-n", OB_NS, "get", "deployment", "frontend",
                 "-o", "jsonpath={.status.readyReplicas}"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            ready = int(out or "0")
            if ready == target_replicas:
                LOGGER.info("cluster stable at %d ready replicas", target_replicas)
                return True
        except Exception as exc:
            LOGGER.debug("readiness check error: %s", exc)
        time.sleep(10)
    LOGGER.warning("cluster did not reach %d replicas within %ds", target_replicas, timeout_seconds)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metric collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_scaling_metrics(start: datetime, end: datetime, deployment: str = "frontend") -> dict:
    """Replica-seconds and peak over the load window (one range query).

    The pilot used a trailing [1h] lookback for peak_replicas, bleeding
    across trials; both metrics now derive from the same windowed series.
    """
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/query_range",
            params={
                "query": f'kube_deployment_spec_replicas{{deployment="{deployment}"}}',
                "start": start.isoformat(),
                "end": end.isoformat(),
                "step": "15s",
            },
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()["data"]["result"]
        values = [float(v[1]) for series in result for v in series["values"]]
        if not values:
            return {"replica_seconds": None, "peak_replicas": None}
        return {
            "replica_seconds": sum(values) * 15,
            "peak_replicas": int(max(values)),
        }
    except Exception as exc:
        LOGGER.warning("scaling metric collection failed: %s", exc)
        return {"replica_seconds": None, "peak_replicas": None}


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-bundle analysis (hybrid only)
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_evidence(evidence_path: Path) -> dict:
    if not evidence_path.is_file():
        return {}
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    # Oscillation: count direction reversals in new_replicas
    replicas = [r["new_replicas"] for r in rows]
    oscillations = sum(
        1 for i in range(1, len(replicas) - 1)
        if (replicas[i] - replicas[i - 1]) * (replicas[i + 1] - replicas[i]) < 0
    )
    # Fallback fraction
    fallback_states = {"FALLBACK_FORECASTER_FAULT", "FALLBACK_UNCERTAINTY"}
    fallback_count = sum(1 for r in rows if r.get("state") in fallback_states)
    # Forecast RMSE vs observed (rough proxy; Phase 8 computes proper RMSE)
    return {
        "oscillation_count": oscillations,
        "fallback_fraction": fallback_count / len(rows) if rows else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core trial runner
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    spec: TrialSpec,
    plan: TrialPlan,
    result_dir: Path,
    dry_run: bool,
) -> TrialResult:
    LOGGER.info("=" * 70)
    LOGGER.info("TRIAL %s  autoscaler=%s  workload=%s", spec.trial_id, spec.autoscaler, spec.workload)
    LOGGER.info("=" * 70)

    log_dir = result_dir / "logs" / spec.trial_id
    log_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    error: Optional[str] = None
    hybrid_proc: Optional[subprocess.Popen] = None
    locust_proc: Optional[subprocess.Popen] = None
    wrk2_raw = ""
    load_tool = "none"

    evidence_path = result_dir / f"{spec.trial_id}-evidence.jsonl"

    load_start: Optional[datetime] = None
    load_end: Optional[datetime] = None

    try:
        # 1. Clean state: remove leftover autoscalers, scale frontend back to
        #    baseline and BLOCK until it is there (aborts the trial otherwise).
        _remove_hpa(dry_run)
        _stop_hybrid(None, dry_run)  # best-effort; previous proc already stopped
        _reset_baseline(1, plan.stabilisation_wait_seconds, dry_run)

        # 2. Install autoscaler
        if spec.autoscaler == Autoscaler.HPA:
            _install_hpa(plan.hpa_manifest, dry_run)
        else:
            hybrid_proc = _start_hybrid(
                spec,
                plan,
                log_dir / "controller.log",
                dry_run,
                evidence_path=evidence_path,
            )
            time.sleep(5 if not dry_run else 0)  # let controller warm up

        # 3. Start Locust and verify it survived startup (DEV-017)
        locust_proc = _start_locust(spec, log_dir / "locust.log", dry_run)
        _assert_locust_alive(locust_proc, log_dir / "locust.log", dry_run)
        load_start = datetime.now(timezone.utc)
        t_load = time.monotonic()

        # 4. Phase alignment (DEV-014): wait until the profile phase under
        #    measurement begins (burst: t=300s → the burst itself; ramp:
        #    t=600s → the plateau), then measure for wrk2_duration_seconds.
        if not dry_run and spec.measure_start_offset_seconds > 0:
            wait = spec.measure_start_offset_seconds - (time.monotonic() - t_load)
            if wait > 0:
                LOGGER.info("waiting %.0fs for measurement phase (offset=%ds)",
                            wait, spec.measure_start_offset_seconds)
                time.sleep(wait)
        wrk2_raw, load_tool = _run_wrk2(spec, log_dir / "wrk2.txt", dry_run)

        # 5. Let the workload profile COMPLETE (DEV-014: the pilot killed
        #    Locust when measurement ended, so no profile ever finished).
        if locust_proc is not None and not dry_run:
            remaining = spec.locust_duration_seconds - (time.monotonic() - t_load) + 60
            try:
                locust_proc.wait(timeout=max(30, remaining))
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Locust exceeded its own duration budget "
                    f"({spec.locust_duration_seconds}s + 60s grace)"
                ) from exc
        load_end = datetime.now(timezone.utc)
        locust_proc = None

        # 6. Stop autoscaler
        if spec.autoscaler == Autoscaler.HPA:
            _remove_hpa(dry_run)
        else:
            _stop_hybrid(hybrid_proc, dry_run)
            hybrid_proc = None

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("trial %s failed", spec.trial_id)
    finally:
        _stop_locust(locust_proc, dry_run)
        _stop_hybrid(hybrid_proc, dry_run)
        _remove_hpa(dry_run)

    end_time = datetime.now(timezone.utc)

    # 7. Collect Prometheus metrics over the LOAD window only (T1.3): the
    #    pilot integrated from supervisor entry — including the stabilisation
    #    wait — giving paired arms unequal windows (DEV-015).
    window_start = load_start or start_time
    window_end = load_end or end_time
    scaling = {} if dry_run else _collect_scaling_metrics(window_start, window_end)

    # 8. Parse load-test output
    latency = _parse_wrk2(wrk2_raw, load_tool)

    # 9. Analyse evidence bundle (hybrid only)
    evidence_metrics = {}
    if spec.autoscaler == Autoscaler.HYBRID:
        evidence_metrics = _analyse_evidence(evidence_path)

    result = TrialResult(
        trial_id=spec.trial_id,
        autoscaler=spec.autoscaler,
        workload=spec.workload,
        seed=spec.seed,
        start_time=start_time,
        end_time=end_time,
        p50_latency_ms=latency.get("p50_latency_ms"),
        p95_latency_ms=latency.get("p95_latency_ms"),
        p99_latency_ms=latency.get("p99_latency_ms"),
        p999_latency_ms=latency.get("p999_latency_ms"),
        throughput_rps=latency.get("throughput_rps"),
        errors_total=latency.get("errors_total"),
        requests_total=(
            round(latency["throughput_rps"] * spec.wrk2_duration_seconds)
            if latency.get("throughput_rps") is not None else None
        ),
        success_rate=(
            max(0.0, 1.0 - latency["errors_total"]
                / (latency["throughput_rps"] * spec.wrk2_duration_seconds))
            if latency.get("throughput_rps") and latency.get("errors_total") is not None
            else None
        ),
        load_tool=load_tool,
        replica_seconds=scaling.get("replica_seconds"),
        peak_replicas=scaling.get("peak_replicas"),
        oscillation_count=evidence_metrics.get("oscillation_count"),
        fallback_fraction=evidence_metrics.get("fallback_fraction"),
        evidence_path=evidence_path if spec.autoscaler == Autoscaler.HYBRID else None,
        wrk2_output_path=log_dir / "wrk2.txt",
        wrk2_raw=wrk2_raw,
        error=error,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command(name="supervisor")
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the TrialPlan YAML.",
)
@click.option(
    "--result-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Override the result directory from the plan.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Log all actions without executing subprocesses or kubectl.",
)
@click.option(
    "--trial-id",
    default=None,
    help="Run only this single trial (by trial_id).",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    plan_path: Path,
    result_dir: Optional[Path],
    dry_run: bool,
    trial_id: Optional[str],
    log_level: str,
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    plan = TrialPlan.from_yaml(plan_path)
    out_dir = result_dir or plan.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = plan.trials
    if trial_id is not None:
        trials = [t for t in trials if t.trial_id == trial_id]
        if not trials:
            click.echo(f"ERROR: trial_id '{trial_id}' not found in plan", err=True)
            sys.exit(1)

    result_log = out_dir / f"results_{plan.name.replace(' ', '_')}.jsonl"
    LOGGER.info("plan: %s  trials: %d  result_log: %s", plan.name, len(trials), result_log)
    if dry_run:
        LOGGER.info("DRY-RUN mode enabled")

    failed: list[str] = []
    with open(result_log, "a") as log_f:
        for i, spec in enumerate(trials, 1):
            LOGGER.info("trial %d/%d", i, len(trials))
            result = run_trial(spec, plan, out_dir, dry_run)
            log_f.write(result.model_dump_json() + "\n")
            log_f.flush()
            if result.error:
                failed.append(spec.trial_id)
            LOGGER.info(
                "trial %s done: p95=%.1fms replica_s=%.0f error=%s",
                result.trial_id,
                result.p95_latency_ms or 0,
                result.replica_seconds or 0,
                result.error or "none",
            )

    if failed:
        # A failed trial must never masquerade as a completed campaign
        # (2026-07-05 review): summarise and exit non-zero.
        click.echo(
            f"❌  {len(failed)}/{len(trials)} trial(s) FAILED: {', '.join(failed)}\n"
            f"    results (with error fields) in {result_log}",
            err=True,
        )
        sys.exit(1)
    click.echo(f"✅  {len(trials)} trial(s) completed cleanly → {result_log}")


if __name__ == "__main__":
    main()
