"""Seed-management helpers (§3.6 / docs/SEEDS.md).

Single root seed S yields three derived seeds, applied consistently across
every stochastic component:

    S_numpy  = S        → numpy.random.seed
    S_torch  = S + 1    → torch.manual_seed (covers CUDA + MPS in PyTorch ≥ 2.x)
    S_loader = S + 2    → DataLoader(worker_init_fn=...)

CUDA non-determinism is disabled when torch is present. The seed triple is
returned by `set_seeds` so it can be logged per trial alongside the trial id.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedTriple:
    """The three derived seeds for one trial (per docs/SEEDS.md)."""

    numpy: int
    torch: int
    loader: int

    @classmethod
    def from_root(cls, root: int) -> "SeedTriple":
        return cls(numpy=root, torch=root + 1, loader=root + 2)


def set_seeds(root: int, *, strict_determinism: bool = True) -> SeedTriple:
    """Set NumPy / Python / torch seeds from a single root.

    Parameters
    ----------
    root : int
        Root seed S. Derived seeds are (S, S+1, S+2).
    strict_determinism : bool
        If True (default), enable torch's deterministic-algorithm flag and
        disable cuDNN benchmark + non-determinism. §3.6 explicitly accepts
        the modest training-time cost in exchange for reproducibility.

    Returns
    -------
    SeedTriple
        The triple actually applied. Log this alongside the trial id.
    """
    if not isinstance(root, int) or root < 0:
        raise ValueError("root seed must be a non-negative integer")

    triple = SeedTriple.from_root(root)

    np.random.seed(triple.numpy)
    random.seed(triple.numpy)

    try:
        import torch
        torch.manual_seed(triple.torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(triple.torch)
        if strict_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # `warn_only=True` so we don't crash on PyTorch ops that have no
            # deterministic implementation (e.g. some scatter ops); we accept
            # the warning and rely on the bit-exact re-run check.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                # Older torch — best-effort.
                pass
    except ImportError:
        # torch is not installed; fine for non-LSTM workflows.
        pass

    return triple


def dataloader_worker_init(triple: SeedTriple):
    """Return a worker_init_fn suitable for torch DataLoader.

    Each worker process gets a deterministic seed derived from the loader seed.
    """
    def _init(worker_id: int) -> None:
        seed = triple.loader + worker_id
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    return _init


def verify_bit_exact(state_dict_a, state_dict_b) -> bool:
    """Assert that two torch state_dicts are bit-exactly equal.

    Returns True if all tensor values match; False otherwise.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("verify_bit_exact requires torch") from e

    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        return False
    for k in state_dict_a:
        if not torch.equal(state_dict_a[k], state_dict_b[k]):
            return False
    return True
