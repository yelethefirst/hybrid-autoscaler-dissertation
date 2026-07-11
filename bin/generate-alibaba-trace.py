"""Generate a synthetic 30-minute RPS trace with Alibaba cluster trace statistics.

Statistical basis (from published Alibaba trace characterisation papers):
  - Luo et al. (2022): "The Alibaba cluster trace"
  - Weng et al. (2022): "MLaaS in the Wild" (Alibaba MSResource)
  - Guo et al. (2019): "Who limits the resource efficiency of my datacenter"

Key properties reproduced:
  - Self-similar / long-range dependent traffic (Hurst exponent H ≈ 0.75)
    generated via fractional Gaussian noise (fGn) using Hosking's method
  - Log-normal marginal distribution of request rates (heavy-tailed)
  - Burst factor: peak / mean ≈ 3-4x (typical for a 30-min microservice window)
  - One structured burst event (spike at ~t=10 min, decays over ~3 min)
    representing a flash-crowd or upstream cascade

Output: experiments/traces/alibaba_rps_30m.csv
  Two columns, no header: timestamp_seconds, rps
  One row per 15 seconds, 120 rows total (30 min).
  15-second granularity matches Alibaba MSResource container data collection interval.

Reproducible: fixed seed=42.

Usage:
    uv run python bin/generate-alibaba-trace.py [--out PATH] [--seed N] [--plot]
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Fractional Gaussian Noise (Hosking 1984 — exact covariance method)
# Produces self-similar increments with Hurst exponent H.
# ──────────────────────────────────────────────────────────────────────────────

def _fgn_covariance(n: int, H: float) -> np.ndarray:
    """Auto-covariance sequence of fGn with Hurst parameter H."""
    k = np.arange(n, dtype=float)
    cov = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    return cov


def generate_fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Exact Hosking fGn via Cholesky; O(n^2) — fine for n=120."""
    cov = _fgn_covariance(n, H)
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i:] = cov[:n - i]
        C[i:, i] = cov[:n - i]
    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    return L @ rng.standard_normal(n)


# ──────────────────────────────────────────────────────────────────────────────
# Trace generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_trace(
    duration_seconds: int = 1800,
    interval_seconds: int = 15,
    baseline_rps: float = 120.0,
    peak_rps: float = 420.0,
    H: float = 0.75,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Return list of (timestamp_s, rps) pairs covering duration_seconds."""
    rng = np.random.default_rng(seed)
    n = duration_seconds // interval_seconds + 1  # 121 points: 0 … 1800 inclusive

    # 1. Self-similar background via fGn → log-normal RPS
    noise = generate_fgn(n, H, rng)
    # Map fGn to log-normal: exp(mu + sigma * noise)
    # Calibrated so mean ≈ baseline_rps, std ≈ 0.35 * baseline_rps
    sigma_ln = 0.30
    mu_ln = math.log(baseline_rps) - 0.5 * sigma_ln ** 2
    background = np.exp(mu_ln + sigma_ln * noise)

    # 2. Structured burst at ~t=10 min (Alibaba "flash crowd" pattern)
    #    Shape: rapid rise over 1 min, plateau 2 min, exponential decay 3 min
    ts = np.arange(n) * interval_seconds
    burst_centre = 600.0   # 10 min
    burst_amp = peak_rps - baseline_rps

    # Asymmetric Gaussian: steeper rise, slower decay (as in Alibaba traces)
    rise_sigma = 45.0    # ~45 s rise
    fall_sigma = 120.0   # ~2 min fall
    diff = ts - burst_centre
    burst = burst_amp * np.where(
        diff < 0,
        np.exp(-0.5 * (diff / rise_sigma) ** 2),
        np.exp(-0.5 * (diff / fall_sigma) ** 2),
    )

    # 3. Small secondary spike at ~t=22 min (upstream retry storm)
    burst2_centre = 1320.0
    burst2_amp = 0.40 * burst_amp
    diff2 = ts - burst2_centre
    burst2 = burst2_amp * np.where(
        diff2 < 0,
        np.exp(-0.5 * (diff2 / 30.0) ** 2),
        np.exp(-0.5 * (diff2 / 75.0) ** 2),
    )

    rps = background + burst + burst2
    # Clip to a realistic floor (never below 20 RPS)
    rps = np.clip(rps, 20.0, None)

    return [(float(ts[i]), float(rps[i])) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/traces/alibaba_rps_30m.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    args = parser.parse_args()

    trace = generate_trace(seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(trace)

    rps_vals = [r for _, r in trace]
    print(f"Written {len(trace)} rows to {out}")
    print(f"  Duration : {trace[-1][0]:.0f} s ({trace[-1][0]/60:.1f} min)")
    print(f"  Interval : {trace[1][0] - trace[0][0]:.0f} s")
    print(f"  RPS min  : {min(rps_vals):.1f}")
    print(f"  RPS mean : {sum(rps_vals)/len(rps_vals):.1f}")
    print(f"  RPS max  : {max(rps_vals):.1f}")
    print(f"  Burst    : {max(rps_vals)/( sum(rps_vals)/len(rps_vals) ):.2f}x mean")

    if args.plot:
        import matplotlib.pyplot as plt
        ts = [t / 60 for t, _ in trace]
        plt.figure(figsize=(12, 4))
        plt.plot(ts, rps_vals, linewidth=1.2)
        plt.xlabel("Time (min)")
        plt.ylabel("RPS")
        plt.title("Synthetic Alibaba-style trace — alibaba_rps_30m.csv")
        plt.tight_layout()
        plt.savefig(str(out).replace(".csv", ".png"), dpi=150)
        print(f"  Plot saved: {str(out).replace('.csv', '.png')}")
        plt.show()


if __name__ == "__main__":
    main()
