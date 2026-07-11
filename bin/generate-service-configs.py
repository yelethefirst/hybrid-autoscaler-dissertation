#!/usr/bin/env python3
"""Generate per-service controller configs from manifest CPU limits + model registry.

Usage:
    uv run python bin/generate-service-configs.py \
        --registry experiments/results/phase3_20260703T090249Z_model_registry.yaml \
        --out-dir controller/configs/ab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# CPU REQUESTS (cores) from the Online Boutique v0.10.5 manifest.
# rho derives from REQUESTS, not limits: Kubernetes HPA computes utilisation
# against requests, so a limit-based rho gave the hybrid a 2x-more-tolerant
# per-pod threshold than the HPA arm — an unfair efficiency head start
# (2026-07-05 code review; DEV-020). Pilot configs used limits (DEV-012).
CPU_REQUESTS = {
    "adservice":            0.200,
    "cartservice":          0.200,
    "checkoutservice":      0.100,
    "currencyservice":      0.100,
    "emailservice":         0.100,
    "frontend":             0.100,
    "paymentservice":       0.100,
    "productcatalogservice":0.100,
    "recommendationservice":0.100,
    "shippingservice":      0.100,
}

# Conservative replica bounds per service.
R_MAX = {
    "frontend":             8,
    "currencyservice":      6,
    "adservice":            4,
    "cartservice":          4,
    "recommendationservice":4,
    "productcatalogservice":4,
    "checkoutservice":      4,
    "emailservice":         2,
    "paymentservice":       2,
    "shippingservice":      2,
}

TARGET_UTILISATION = 0.50   # matches Kubernetes HPA default


def rho(service: str) -> float:
    req = CPU_REQUESTS.get(service)
    if req is None:
        raise ValueError(f"unknown service: {service}")
    return round(req * TARGET_UTILISATION, 4)


CONFIG_TEMPLATE = """\
# {service}-ab.yaml
#
# A/B experiment controller config for {service}.
# rho = cpu_request({limit_raw}) x target_utilisation({target_pct}%) = {rho:.4f} cores (HPA parity — DEV-020).
# Forecaster selected by Phase 3 real-data burst campaign (FULL_GRID=1, N_SPLITS=5).
#
service: {service}
namespace: default

r_min: 1
r_max: {r_max}
delta_s: 2

rho: {rho:.4f}
k: 1.5
sigma_max: 0.20
sigma_warn_ratio: 0.7

horizon_seconds: 30
tick_seconds: 15
history_seconds: 600

target_utilisation: {target_dec}
hpa_stabilisation_window_seconds: 300

metric_source: cpu

evidence_path: experiments/results/ab-{service}.jsonl
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True, type=Path)
    p.add_argument("--out-dir", default="controller/configs/ab", type=Path)
    args = p.parse_args()

    registry = yaml.safe_load(args.registry.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    services_in_registry = {e["service"] for e in registry.get("entries", [])}

    print(f"{'Service':30s} {'Req':7s} {'rho':8s} {'r_max':6s} {'forecaster':20s} {'RMSE':8s}")
    print("-" * 90)

    generated = []
    for svc in sorted(CPU_REQUESTS.keys()):
        if svc not in services_in_registry:
            print(f"{svc:30s}  (not in registry — skip)")
            continue

        entry = next(e for e in registry["entries"] if e["service"] == svc)
        lim = CPU_REQUESTS[svc]
        r = rho(svc)
        r_max = R_MAX.get(svc, 4)
        forecaster = entry["forecaster"]
        rmse = entry["validation"]["mean_rmse"]

        limit_raw = f"{int(lim * 1000)}m"
        content = CONFIG_TEMPLATE.format(
            service=svc,
            limit_raw=limit_raw,
            target_pct=int(TARGET_UTILISATION * 100),
            rho=r,
            r_max=r_max,
            target_dec=TARGET_UTILISATION,
        )
        out_path = args.out_dir / f"{svc}-ab.yaml"
        out_path.write_text(content)
        generated.append(out_path)

        print(f"{svc:30s} {limit_raw:7s} {r:.4f}   {r_max:<6d} {forecaster:20s} {rmse:.5f}")

    print(f"\nWrote {len(generated)} configs to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
