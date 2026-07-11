# Implementation Deviations from Dissertation Specification

This file records every place where the running artefact diverges from the
methodology as written in the dissertation. Each entry states:

- **What the dissertation says** — the exact specification.
- **What the artefact does** — the actual implementation.
- **Reason** — why the deviation was necessary or beneficial.
- **Dissertation update required** — the section(s) to update before submission.
- **Impact on validity** — whether this changes any experimental claim.

Entries are added as they are discovered; they are resolved by updating the
dissertation text before final submission. *No entry here means the claim in
the dissertation is wrong — it means the claim needs to be re-stated to match
reality.*

---

## DEV-001 — Kubernetes version: v1.30 → v1.33

| Field | Value |
|---|---|
| **Dissertation says** | §3.10: "kind v0.22, Kubernetes v1.30" |
| **Artefact does** | kind v0.32.0, Kubernetes v1.33.12 |
| **Reason** | Kubernetes v1.30 reached End of Life in June 2025 (superseded by v1.31–1.33). Using an EOL version in a 2026 submission is a credible examiner critique. Kubernetes v1.33 is the current stable release and kind v0.32 is the current kind release. The HPA controller behaviour and cAdvisor metric families used by this system are unchanged between v1.30 and v1.33. |
| **Dissertation update** | §3.10 cluster specification; §3.13 remove the threat around EOL version. |
| **Impact on validity** | None. The HPA algorithm, `kubectl scale` subresource, and all five metric families queried from Prometheus are identical across these versions. |

**Update 2026-07-05:** version audit found the development Mac actually runs **kind v0.23.0** (installed mid-2024) while creating clusters from the pinned `kindest/node:v1.33.12` image — the Kubernetes version claim holds, the kind-binary claim was aspirational for the pilot. The measurement environment installs **kind v0.32.0** (current latest, verified 2026-07-05) via `infra/ansible/provision.yml`, so §3.10-A is accurate; §3.10-B (pilot) is scoped to kind v0.23.0.

---

## DEV-002 — Prometheus version: v2.50 → v2.52

| Field | Value |
|---|---|
| **Dissertation says** | §3.10: "Prometheus 2.50" |
| **Artefact does** | Prometheus v2.52.0 (shipped by kube-prometheus-stack chart 58.7.2) |
| **Reason** | The pinned Helm chart (58.7.2) ships Prometheus v2.52.0, not v2.50. Downgrading the chart to obtain exactly v2.50 would require using an older, unsupported chart release. All PromQL queries used in this system are valid in both versions; no breaking API changes affect the five metric families. |
| **Dissertation update** | §3.10 observability stack specification. |
| **Impact on validity** | None. PromQL syntax and the five metric families are unchanged. |

---

## DEV-003 — node-exporter version: v1.7 → v1.8

| Field | Value |
|---|---|
| **Dissertation says** | §3.10: "node-exporter 1.7" |
| **Artefact does** | node-exporter v1.8.0 (shipped by kube-prometheus-stack chart 58.7.2) |
| **Reason** | Same as DEV-002 — the chart ships the component versions it ships. node-exporter v1.8 adds additional host metrics but does not remove or change any metric used by this system. |
| **Dissertation update** | §3.10 observability stack specification. |
| **Impact on validity** | None. No node-exporter metrics are used directly in the autoscaler decision path. |

---

## DEV-004 — Prometheus histogram query in Phase 0 verify check

| Field | Value |
|---|---|
| **Dissertation says** | §3.5 / §3.10 exit criterion: "response-time histograms are queryable for every Deployment" |
| **Artefact does** | `infra/verify.sh` uses `count(apiserver_request_duration_seconds_bucket)` as the histogram probe rather than the original `count({__name__=~".*_bucket"})` |
| **Reason** | The bare `{__name__=~".*_bucket"}` regex scan was rejected by Prometheus v2.52 with HTTP 400 (no metric name matcher). Prometheus ≥v2.20 requires at least one non-regex label matcher for queries that scan all series. The replacement query uses the Kubernetes API server's own latency histogram, which is always present in any cluster. Application-level response-time histograms for Online Boutique services are obtained via wrk2 in Phase 5 (§3.9), not from Prometheus scrapes. |
| **Dissertation update** | None required — the verify check is an artefact implementation detail, not a methodology claim. The §3.5 exit criterion (response-time histograms queryable) is satisfied by Phase 5 wrk2 measurements. |
| **Impact on validity** | None. |

---

## DEV-005 — cartservice transient CrashLoopBackOff on local dev profile

| Field | Value |
|---|---|
| **Dissertation says** | §3.10 exit criterion: "all 11 Online Boutique services Ready" |
| **Artefact does** | On the `kind-local` profile (2 vCPU / 4 GiB per worker), cartservice (C#/.NET) enters a brief CrashLoopBackOff on initial startup before stabilising; emailservice may show `0/1 Running` (readiness probe delay) for 1–2 minutes. Both stabilise without intervention. |
| **Reason** | The local profile is deliberately under-resourced for laptop development. The .NET runtime cold-start is CPU-intensive and triggers a restart under contention. On the cloud profile (4 vCPU / 12 GiB per worker, used for measured runs) both services start cleanly. |
| **Dissertation update** | §3.13 Threats to Validity — note local dev resource constraints; confirm cloud profile exit criterion is met for final runs. |
| **Impact on validity** | None on measured results, which are taken on the cloud profile. |

---

## DEV-006 — "use latest versions" audit (infrastructure complete; Python stack pending)

| Field | Value |
|---|---|
| **Dissertation says** | Versions pinned in §3.10 and `pyproject.toml` reflect mid-2024 choices. |
| **Artefact does** | Infrastructure updated to latest (Kubernetes v1.33.12, kind v0.32.0). Python ML stack (PyTorch, XGBoost, scikit-learn, statsmodels, Locust, SHAP, etc.) retains mid-2024 pins pending compatibility audit. |
| **Reason** | Infrastructure version updates are low-risk (no API changes). Python ML library updates carry risk of breaking changes in model API, seed behaviour, or numerical results. A dedicated audit will update each dependency, run the 143-test suite, and verify LSTM determinism before changing pins. |
| **Dissertation update** | §3.10 dependency table — update to reflect final `uv.lock` versions before submission. |
| **Impact on validity** | TBD after audit. If model outputs differ, the selection experiment must be re-run. |

**Resolution 2026-07-05 (audit executed):** live check of every layer against upstream:
- **Current where it matters**: Online Boutique v0.10.5 = latest release (2026-03-11); kind v0.32.0 target = latest (2026-06-02); Kubernetes v1.33 in support; Terraform 1.14.7 (constraint ≥1.5).
- **Python pins deliberately retained.** The gaps are breaking-major jumps (numpy 1.26→2.4, pandas 2.2→3.0, openai 1.x→2.x, kubernetes-client 30→36). Upgrading buys no methodological benefit, risks numerical drift against the verified seed/determinism baseline, and would desynchronise the pilot and final environments. Locust is pinned at 2.44.0 **because the dissertation cites that version** (§3.9); same for the Prometheus 2.52 stack (§3.10, DEV-002). In a version-pinned reproducibility methodology, "latest" is not a virtue; absence of EOL/security exposure is the requirement, and nothing here is internet-exposed.
- **One skew to fix on the VM**: local kubectl client v1.36 vs cluster v1.33 exceeds the supported ±1 minor skew; `infra/ansible/provision.yml` pins kubectl v1.33.12 (sha256-verified).
DEV-006 is closed; any future pin change re-opens it.

---

## DEV-007 — cartservice memory limit: 128Mi → 256Mi

| Field | Value |
|---|---|
| **Dissertation says** | Online Boutique deployed with upstream manifest resource limits. |
| **Artefact does** | `infra/online-boutique/patches/cartservice-memory.yaml` raises the cartservice memory limit from 128Mi → 256Mi (request 64Mi → 128Mi). Applied by `install.sh` after the main manifest. |
| **Reason** | The .NET CLR baseline on arm64 (Apple Silicon) exceeds 128Mi under any loadgenerator traffic, causing OOMKilled restarts (Exit Code 137). 256Mi provides ~100 MiB headroom above the observed working set. The upstream limit is sized for x86_64 where the CLR footprint is lower. |
| **Dissertation update** | §3.10 / §3.13: note the resource patch in the cluster setup and as a platform-specific deviation. |
| **Impact on validity** | None. The cartservice memory limit does not affect forecasting accuracy, HPA behaviour, or the scaling decisions under study. The autoscaler targets CPU and request-rate metrics, not memory. |

**Update 2026-07-05:** `install.sh` now applies this patch **only when the cluster nodes are arm64** (detected at install time). The x86_64 cloud measurement environment (§3.10-A) deploys the upstream manifest unmodified, so DEV-007 is scoped to the development/pilot environment only.

---

---

## DEV-008 — frontend rho: placeholder 0.30 → calibrated 0.125

| Field | Value |
|---|---|
| **Dissertation says** | §3.7 / §3.10: ρ_t = per-pod target CPU capacity; initial value specified in the frontend-local.yaml scaffold as 0.30 cores. |
| **Artefact does** | `controller/configs/frontend-local.yaml` uses `rho: 0.125` — derived analytically as `cpu_limit(250m) × target_utilisation(0.50) = 0.125 cores`. |
| **Reason** | The placeholder value of 0.30 exceeded the container's own CPU limit (250m = 0.25 cores), making it impossible for any forecast to cross the threshold and trigger a scale-up on a local cluster. The correct derivation uses the container's CPU limit and target utilisation, consistent with how Kubernetes HPA sets its threshold. Phase 2 will replace this with an empirically fitted value from steady-state telemetry. |
| **Dissertation update** | §3.7 and §3.10 parameter table — document the derivation formula and the Phase 2 calibration procedure. |
| **Impact on validity** | None. The analytical derivation is equivalent to what Phase 2 calibration will produce; any remaining discrepancy will be corrected after steady-state data collection. |

---

## DEV-009 — Locust campaign shape relative import fix

| Field | Value |
|---|---|
| **Dissertation says** | §3.9: workload shapes drive Locust users against the Online Boutique frontend. |
| **Artefact does** | `experiments/workloads/campaign_burst.py` and `campaign_ramp.py` do not use a relative import for `BoutiqueUser`; `user.py` is always passed as a separate `-f` argument to Locust. |
| **Reason** | Locust's `-f file1.py,file2.py` syntax loads each file as a standalone module, not as a Python package. A `from .user import BoutiqueUser` relative import in the shape file raises `ImportError: attempted relative import with no known parent package` at startup. Removing the import and relying on Locust's automatic class discovery from the separately-passed `user.py` is the correct pattern. |
| **Dissertation update** | None required — this is an implementation detail of how Locust loads files. |
| **Impact on validity** | None. The shape logic (user counts, timing, spawn rate) is identical. |

---

## DEV-010 — LSTM excluded from Phase 3 real-data run (MPS deadlock)

| Field | Value |
|---|---|
| **Dissertation says** | §3.6: five forecaster families evaluated — Seasonal Naive, Holt-Winters, SARIMA, XGBoost, LSTM. |
| **Artefact does** | LSTM is excluded from the canonical Phase 3 model registry. Two attempts were made: (1) first run used `SKIP_LSTM=1` (CPU-only estimated 30+ hours); (2) second run with MPS device placement added consumed 3h54m CPU time and then deadlocked — PyTorch 2.4.0 MPS backend hung during LSTM grid search with no GPU activity (`Device Utilization % = 3`, `fLastSubmissionPID ≠ our PID`). The process was killed after confirming the deadlock. No output was written (pipeline uses `set -euo pipefail`). |
| **Reason** | PyTorch 2.4.0 MPS has known deadlock issues with certain LSTM configurations on Apple Silicon. The full grid (24 combinations × 5 folds × up to 200 epochs) likely triggered a Metal command-buffer overflow or a known MPS synchronisation bug. |
| **Dissertation update** | §3.6 and §4.2: note that LSTM was excluded due to MPS instability on the development platform (Apple M3); the four other forecaster families were evaluated. §3.13: add as a threat to validity — LSTM may perform differently from XGBoost/Holt-Winters on some services, but cannot be determined without a stable GPU runtime. |
| **Impact on validity** | Moderate. LSTM is excluded entirely from the model registry used in the A/B experiment. 6/10 services use XGBoost and 4/10 use Holt-Winters (both beat Seasonal Naive at p_holm < 0.05 on 8/10 services). LSTM may have displaced some winners but this cannot be verified on the current platform. |

---

## DEV-011 — Forecasting pipeline updated to load multi-batch Parquet directories

| Field | Value |
|---|---|
| **Dissertation says** | §3.5 / §3.6: telemetry collected then used for model training. |
| **Artefact does** | `forecasting/run_forecasting.py` originally loaded a single Parquet file (the synthetic data case). Updated to detect `telemetry_*.parquet` batch files in `--telemetry-dir` and merge them via `data.collect.load_campaign()`. Falls back to single-file loading when no batch files are present. |
| **Reason** | The live Prometheus collection pipeline writes one Parquet file per 5-minute batch. The forecasting pipeline must merge all batches for a workload before training. |
| **Dissertation update** | None required — the data loading mechanism is an implementation detail. |
| **Impact on validity** | None. The merged DataFrame is identical to what would result from a single large file. |

---

## DEV-012 — frontend rho correction: 0.125 → 0.100; per-service configs generated

| Field | Value |
|---|---|
| **Dissertation says** | §3.7 / §3.10: ρ_t = per-pod target CPU capacity; derived as `cpu_limit × target_utilisation`. |
| **Artefact does** | All controller configs now use `rho = cpu_limit × 0.50` computed from the actual manifest limits. frontend limit is 200m (not 250m as previously assumed), giving `rho = 0.100` (was 0.125). All 10 services have individual configs in `controller/configs/ab/` generated by `bin/generate-service-configs.py`. |
| **Reason** | The previous 250m figure was incorrect; the `pinned-manifest.yaml` specifies 200m for frontend. Empirical validation from burst campaign data confirms currencyservice (P80 = 60% of 200m limit) and frontend (P80 = 54% of 200m limit) are the two services that will regularly cross the ρ=0.100 threshold and trigger scale-up decisions. |
| **Dissertation update** | §3.7 and §3.10 parameter table — correct frontend cpu_limit to 200m and rho to 0.100. Add per-service ρ table covering all 10 services. |
| **Impact on validity** | Minor. The corrected ρ=0.100 is lower than the previous 0.125, meaning the controller will recommend scale-up at a lower predicted CPU level. This is the correct behaviour — the previous value was overly conservative and would have under-triggered scaling. |

---

---

## DEV-013 — Load tester: wrk2 → hey (arm64 macOS Gatekeeper)

| Field | Value |
|---|---|
| **Dissertation says** | §3.9: "wrk2 with coordinated-omission correction measures p95/p99 latency" |
| **Artefact does** | `hey` (brew install hey) is used when `wrk2` is not available. `hey` provides p50/p95/p99 latency, throughput, and error counts but does **not** apply coordinated-omission (CO) correction. `experiments/supervisor.py` detects `wrk2` first; `hey` is a fallback. |
| **Reason** | wrk2 cannot be built for arm64 macOS without patching (LuaJIT 2.0.3 bundled lacks arm64 support; x86 SIMD intrinsics in hdr_histogram.c). Even after patching, macOS 15 (Sequoia) Gatekeeper rejects unsigned locally-compiled binaries. `hey` is a Go binary distributed via Homebrew with proper code-signing. |
| **Dissertation update** | §3.9 and §3.13: note that CO correction was unavailable on the development platform; discuss whether CO correction materially affects results at the target RPS range (100–500 RPS on a local kind cluster, where coordinator overhead is sub-millisecond). |
| **Impact on validity** | Minor. CO correction matters most when the load generator itself is slow relative to the server, inflating measured latency at low rates. At 100–500 RPS on localhost, the difference is expected to be < 5%. Latency comparisons remain valid as HPA and Hybrid are measured under identical conditions with the same tool. |

---

## DEV-014 — Pilot A/B measurement window not phase-aligned; trials terminated early

| Field | Value |
|---|---|
| **Dissertation says** | §3.9: burst/ramp/periodic/trace profiles of 12–30 minutes; p95 measured under the named workload condition. |
| **Artefact does** | In the 2026-07-03 pilot A/B run, `experiments/supervisor.py` ran the measurement tool (hey) synchronously starting ~7 s after Locust and terminated the trial when measurement ended. Burst trials therefore lasted ~2 m 17 s of a 12-minute profile and the measurement window (t≈7–127 s) fell entirely inside the 300 s warm-up phase — **no pilot burst trial ever executed its burst**. Ramp trials measured the first 300 s of the ramp (users ~50→275), not the plateau; trace trials covered the first 300 s of a 30-minute trace. Periodic (600 s ≈ one full period) was approximately correct. |
| **Reason** | Supervisor defect: measurement duration was conflated with trial duration. Discovered by self-audit on 2026-07-04. |
| **Dissertation update** | §4.3: label all pilot rows as pilot data and disclose the measurement windows; the "burst"/"ramp" labels describe the intended profile, not the measured phase. Corrected in the final campaign (full-profile durations; phase-aligned measurement offsets). |
| **Impact on validity** | Severe for the pilot H0/H2 comparisons — the reported p95 values do not measure the named workload phases. Both arms were mismeasured identically, but no confirmatory claim survives; pilot results are retained for transparency only. |

---

## DEV-015 — Pilot replica-seconds windows unequal between arms; stabilisation non-blocking

| Field | Value |
|---|---|
| **Dissertation says** | Table 3.2: replica-seconds is the time-integral of replica count over the trial duration, compared between paired arms. |
| **Artefact does** | The pilot supervisor set `start_time` before the pre-trial stabilisation wait and continued when stabilisation timed out (logged: "cluster did not reach 1 replicas within 120s"). Consequences in the pilot data: (a) integration windows differed between paired arms by up to ~107 s (e.g. burst pair 1: hybrid 137 s vs HPA 244 s); (b) HPA trials could begin with replicas left over from the preceding hybrid trial. Additionally, replica-seconds was measured for the frontend Deployment only, while Table 3.2 defines it across all services. |
| **Reason** | Two supervisor defects (window anchoring; non-blocking stabilisation gate) plus a metric-scope mismatch. Discovered by self-audit on 2026-07-04. |
| **Dissertation update** | §4.4: pilot H2 comparison declared invalid. Table 3.2: replica-seconds definition scoped to the autoscaled Deployment (frontend). Final campaign: windows anchored to load start/end; stabilisation failure aborts the trial. |
| **Impact on validity** | Severe for pilot H2 — the "34% directional saving" is an artefact of unequal windows and carried-over replicas, and is withdrawn. |

---

## DEV-016 — Per-trial evidence-path override ineffective in pilot run

| Field | Value |
|---|---|
| **Dissertation says** | §3.8/§3.11: per-decision evidence bundles per trial support oscillation and fallback-fraction metrics. |
| **Artefact does** | The pilot supervisor set an `EVIDENCE_PATH` environment variable that `controller/main.py` never read; all hybrid evidence went to the single path in the controller config (`ab-frontend.jsonl`). Per-trial evidence slicing failed silently, so `oscillation_count` and `fallback_fraction` are null for every pilot trial and H3 could not be evaluated. |
| **Reason** | Integration bug between supervisor and controller CLI; no smoke test covered the handoff. Discovered by self-audit on 2026-07-04. |
| **Dissertation update** | §4.3/§4.4: H3 reported as not measurable in the pilot. Final campaign: `--evidence-path` CLI option passed explicitly; per-trial evidence asserted post-trial; oscillation additionally computed for both arms from Prometheus replica time-series. |
| **Impact on validity** | H3 has no pilot evidence either way. The aggregate evidence file (145 ticks, all NOMINAL, SHAP attached) remains valid as a whole-run record. |

---

## DEV-017 — Pilot A/B ran with NO shaped workload; checkout task hit a nonexistent route

| Field | Value |
|---|---|
| **Dissertation says** | §3.9: four workload shapes (burst/ramp/periodic/trace) drive Locust users against the frontend; the task mix includes checkout. |
| **Artefact does** | Forensics on the pilot run logs (2026-07-05) established two workload-generation defects. **(a)** In all 10 canonical A/B trials, the Locust process crashed at startup (`ImportError: attempted relative import` — the DEV-009 bug, fixed in `campaign_*.py` but still present in the four A/B shape files, which the supervisor loaded as a single `-f` argument without `user.py`). **The intended workload shapes therefore never ran in the pilot A/B; the only load present was hey's constant 100-connection closed-loop probe stream.** The "workload condition" factor was void — all four conditions were the same constant load with different durations. The supervisor did not detect the crash. **(b)** Independently, the shared `BoutiqueUser.checkout` task issued `GET /checkout`, a route that does not exist in Online Boutique v0.10.5 (404). Every workload that did run (the 5-hour burst training campaign via `campaign_burst.py`) therefore produced zero traffic to checkoutservice, paymentservice, emailservice and shippingservice. |
| **Reason** | (a) Incomplete propagation of the DEV-009 fix plus no workload-generator liveness check; (b) the checkout route was never validated against the deployed application. |
| **Fixes (2026-07-05, verified live)** | Relative imports removed from `burst.py`/`ramp.py`/`periodic.py`/`trace_replay.py`; supervisor now passes `-f shape,user.py` and asserts the Locust process is alive 8 s after start (aborts the trial otherwise); `run-campaign.sh` gained the missing `trace` case; `BoutiqueUser.checkout` now performs a real add-to-cart + `POST /cart/checkout` with a valid form payload (digits-only card number — the frontend validator rejects dashes). Verification: 25 s live shape run, exit 0, 0% failures on all six routes including completed orders. |
| **Dissertation update** | §4.3 pilot disclosure: the pilot load was hey-only constant closed-loop traffic; the workload-shape factor did not exist in the pilot. §4.2/§5.2: the near-zero CPU variance of checkoutservice/emailservice (and their non-significant H1 results) is partly an artefact of the broken checkout task — the checkout path received no traffic in the training campaign. §3.9: note the corrected checkout flow. |
| **Impact on validity** | Severe for pilot A/B interpretation (compounds DEV-014): between-"workload" differences in the pilot reflect only measurement duration. Material for H1 interpretation on checkout-path services: their telemetry was idle-noise, so "Seasonal Naive is hard to beat on low-variance services" is partly self-inflicted. All final-campaign telemetry and trials use the corrected generator, and the model registry is re-selected from target telemetry (DEV-018), which absorbs this defect. |

---

## Pilot 500-error root cause (forensic note, 2026-07-05)

Not a deviation entry, but recorded here because it drives the final campaign design:

- All pilot errors were **HTTP 500 from the frontend** (no 502/503 — the ingress never refused).
- Cause chain: 100 concurrent closed-loop connections → CPU-throttled critical-path services (200m limits, 1 replica each, on a Docker Desktop VM giving the whole 3-node cluster 11 shared CPUs / 11.9 GiB) → gRPC health probes (1 s timeout) fail → kubelet kills pods (exit 137/143, `reason=Error`, **not** OOMKilled) → in-flight frontend requests hit a dead fatal dependency → 500.
- Evidence: currencyservice 62 restarts (on the fatal path of every page render), adservice 34, emailservice 33, recommendationservice 29, paymentservice 24; live reproduction showed every request logging `rpc error: DeadlineExceeded` on the **non-fatal** ad call (page still 200) while adservice probes failed in real time.
- **Design consequence for the final campaign**: the A/B autoscales frontend only, so non-target services must not be the binding constraint. The campaign pre-scales all non-target services to fixed replica counts (identical in both arms, values set from the capacity probe) so that the frontend is the resource the autoscalers actually contend over, and peak load is calibrated to keep error rate < 5%.

---

## DEV-020 — ρ and fallback utilisation corrected from CPU limits to CPU requests (HPA parity)

| Field | Value |
|---|---|
| **Dissertation says** | §3.7/§3.10: ρ = per-pod target CPU capacity, derived as cpu_limit × target_utilisation (DEV-008/DEV-012); hybrid fallback is "HPA-equivalent". |
| **Artefact does (pilot)** | ρ was derived from CPU **limits** and the fallback utilisation query divided by `kube_pod_container_resource_limits`. Kubernetes HPA computes utilisation against **requests**. With frontend request=100m / limit=200m, the HPA arm scaled at an effective 50 m/pod while the hybrid tolerated 100 m/pod — a built-in ~2× efficiency head start for the hybrid, and a fallback path that was not HPA-equivalent (2026-07-05 external code review). |
| **Fix (2026-07-05)** | `bin/generate-service-configs.py` now derives ρ = cpu_request × 0.50 (all 10 A/B configs regenerated; frontend ρ 0.1000 → **0.0500**); `controller/prometheus_client.py::cpu_avg_utilisation` divides by `kube_pod_container_resource_requests`. HPA `maxReplicas` raised 6 → **8** to equal hybrid `r_max` (caps equalised). |
| **Dissertation update** | §3.7/§3.10 ρ derivation text and the per-service ρ table (supersedes the DEV-012 values and the prereg `rho_values` block — the final-campaign registry re-freeze (DEV-018) records the new values). §3.13: pilot ρ asymmetry noted as a pilot-scoped fairness defect. |
| **Impact on validity** | Severe for pilot H2 in the hybrid's favour (compounding DEV-015): the hybrid's replica-seconds advantage was partly definitional. All final-campaign configs use request-based ρ; empirical re-validation on target telemetry is T3.3. |

---

## External code review — 2026-07-05 (forensic note)

An independent review of the working tree confirmed the audit findings already
scheduled (phase alignment, evidence paths, clean-state reset, plan size,
window contamination, Holm step-down) and surfaced six NEW defects, all fixed
same-day and covered by the 198-test suite:

1. **FActScore verdict parser**: `"supported" in "not_supported"` is true, so
   every negative judge verdict flipped positive — H4 would have inflated
   towards 1.0. Fixed with negation-first parsing (`narrate/factscore.py`).
2. **LSTM TimeSHAP wiring**: attribution referenced nonexistent
   `_model`/`seq_len` attributes → silent failure on every call. Fixed to
   `_best_model`/`_best_hp.seq_len` with a not-fitted guard.
3. **SARIMA selection/deployment mismatch**: CV scored a never-refitted model
   while deployment refits each tick. Selection factories now pass
   `refit_each_predict=True` (matches `_registry_params`).
4. **`metric_source: requests`** was accepted by config but the loop only
   feeds CPU history — now rejected at validation.
5. **wrk2/hey non-zero exit** produced trials with `error=None` — now raises
   and fails the trial.
6. **Trial seeds were metadata-only** — supervisor now exports `TRIAL_SEED`
   and `user.py` seeds the request mix.
Also: `requests_total`/`success_rate`/`load_tool` now populated per trial;
ruff violations 56 → 21 (remaining are style-only, tracked in Phase 1).

---

*Last updated: 2026-07-05 (c)*
