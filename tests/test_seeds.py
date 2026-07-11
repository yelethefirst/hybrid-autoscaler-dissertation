"""Tests for the seed-management helper (docs/SEEDS.md)."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.seeds import SeedTriple, set_seeds


def test_seed_triple_derivation():
    t = SeedTriple.from_root(42)
    assert t.numpy == 42
    assert t.torch == 43
    assert t.loader == 44


def test_set_seeds_makes_numpy_deterministic():
    set_seeds(7)
    a = np.random.randn(10)
    set_seeds(7)
    b = np.random.randn(10)
    np.testing.assert_array_equal(a, b)


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        set_seeds(-1)


def test_set_seeds_torch_optional():
    """When torch is absent, set_seeds should still succeed (best-effort)."""
    triple = set_seeds(0)
    assert triple.numpy == 0
    assert triple.torch == 1
    assert triple.loader == 2
