# DeathStarBench Social Network — Deep-Topology Portability Demonstration (T5.3)

**Purpose** (supervisor-requested, 2026-07-05): show the hybrid controller
operating unmodified against a 28-service application with deep fan-out call
graphs — the external-validity concern §1.8/§2.9 acknowledge about Online
Boutique's shallow topology. **Qualitative demonstration only**: no
statistical claims, not pre-registered, reported as appendix material
("the pipeline transfers without code changes; statistical evaluation remains
future work").

**Why it runs on the VM and never ran on the Mac**: DSB images are largely
x86_64-only — the arm64 dev machine physically could not host it. First
execution doubles as the smoke test.

**Hard gates** (roadmap T5.3): main campaign archived first · ≥ 30 credit-hours
remaining · one-day hard stop.

## Why the controller needs zero code changes

The controller is application-agnostic by construction: it reads
`container_cpu_usage_seconds_total` by pod-name prefix from Prometheus
(cAdvisor scrapes every pod in the cluster, DSB included — no ServiceMonitor
needed) and actuates any Deployment via the standard `/scale` subresource.
Pointing it at DSB is a config file, not a port:
[`controller/configs/dsb/nginx-thrift-dsb.yaml`](../../controller/configs/dsb/nginx-thrift-dsb.yaml)
(frontend analogue) and
[`compose-post-dsb.yaml`](../../controller/configs/dsb/compose-post-dsb.yaml)
(mid-graph service with the widest fan-out).

## Protocol (~half a day)

```bash
# 0. Prereqs: main campaign archived; OB scaled to 0 (install script prompts)
bash infra/deathstarbench/install.sh

# 1. Load — the chart ships wrk2 lua scripts; compose-post exercises the
#    deepest write path (nginx → compose-post → {user, media, text, unique-id,
#    url-shorten, user-mention} → post-storage → {home,user}-timeline fan-out)
kubectl -n socialnet port-forward svc/nginx-thrift 8081:8080 &
# dsb-wrk2 = DSB's bundled wrk2 fork, built by install.sh — required because
# the lua scripts need luasocket (verified at the pinned commit; plain wrk2
# would fail with "module 'socket' not found").
dsb-wrk2 -t4 -c50 -d600s -R80 --latency \
  -s ~/DeathStarBench/socialNetwork/wrk2/scripts/social-network/compose-post.lua \
  http://localhost:8081/wrk2-api/post/compose

# 2. Controller (separate terminal / tmux pane) — Seasonal Naive or
#    Holt-Winters run online with NO training campaign needed; that keeps the
#    demo inside its one-day budget. (Optional stretch: 1–2 h telemetry
#    collect + XGBoost for one service.)
MAX_TICKS=60 bash bin/run-controller.sh controller/configs/dsb/nginx-thrift-dsb.yaml

# 3. Evidence for the appendix
#    experiments/results/dsb-nginx-thrift.jsonl — scaling decisions with SHAP
#    (for online statistical models: component attributions), states, forecasts
#    under deep-fan-out load. One figure + half a page of prose.

# 4. Tear down (same day — credit clock)
bash infra/deathstarbench/install.sh delete
```

## What goes in the dissertation

- **Appendix**: deployment fingerprint (chart @ commit `6ecb0970`), one
  evidence-bundle excerpt, one replica/CPU timeline figure during the wrk2 run.
- **§6.4 upgrade**: from "a DeathStarBench run is planned" to "an initial
  deep-topology portability demonstration was performed (Appendix X);
  statistical evaluation remains future work."
- **What NOT to claim**: any latency/efficiency comparison — no baseline, no
  repetitions, no calibration was performed by design.
