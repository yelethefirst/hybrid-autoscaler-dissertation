# Seed-management convention

This document specifies the seeding discipline required by §3.6 of the dissertation. Every stochastic step in the pipeline draws from a **single seed source** and the seed used is **logged per trial** alongside the trial identifier.

## Why this matters

The §3.6 LSTM training procedure makes a bit-exact reproducibility claim. Achieving that requires control over three independent randomness sources, plus disabling CUDA non-determinism. This document is the single source of truth for how that control is implemented.

## Three seeds per training run

For every LSTM (or other stochastic) training trial, the supervisor sets three seeds derived from a single root seed `S` for that trial:

| Seed | Source | Purpose |
|------|--------|---------|
| `S_numpy = S` | `numpy.random.seed(S_numpy)` | Data shuffling, train/val splits, all NumPy randomness. |
| `S_torch = S + 1` | `torch.manual_seed(S_torch)` + `torch.cuda.manual_seed_all(S_torch)` | Parameter initialisation, dropout sampling. |
| `S_loader = S + 2` | passed to `DataLoader(worker_init_fn=...)` | Per-worker dataloader seeding. |

The triple `(S_numpy, S_torch, S_loader)` is **logged to the experiment log alongside the trial identifier** and included in the reproducibility package (§3.10).

## Disable CUDA non-determinism

When CUDA is active:

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# PyTorch >= 1.8: enforce deterministic algorithms (raises on non-deterministic ops)
torch.use_deterministic_algorithms(True, warn_only=False)
```

The §3.6 methodology explicitly accepts the modest training-time cost of this in exchange for reproducibility.

## Determinism verification

After training, re-run the **same seed configuration** and check that the resulting model parameters match **bit-exactly**:

```python
import torch
def assert_bit_exact(state_dict_a, state_dict_b):
    for k in state_dict_a:
        assert torch.equal(state_dict_a[k], state_dict_b[k]), f"divergence at {k}"
```

This check is run as part of CI and as part of the Phase 3 exit criterion.

## Where seeds are set

| Component | When | How |
|-----------|------|-----|
| LSTM training | start of each trial | `forecasting/lstm_model.py::set_seeds(S)` |
| XGBoost training | start of each trial | `random_state=S` kwarg |
| Train/val split | once per dataset | `numpy.random.default_rng(S)` |
| Experiment trial ordering | per workload cell | supervisor uses `random.Random(S).shuffle(trials)` |
| Bootstrap resampling (§3.12) | per analysis run | `numpy.random.default_rng(S)` |

## Root-seed registry

Final-run root seeds are **drawn once, frozen, and committed to version control** before the measured runs begin (see `preregistration/seeds.json` — created in Phase 8 alongside the OSF pre-registration). This prevents post-hoc seed selection.

For development trials, use seeds from the `dev_seeds` namespace (any integers); these results are not reported in the dissertation.
