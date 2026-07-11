"""Tests for time-ordered splits and walk-forward CV (§3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from data.splits import time_ordered_split, walk_forward_splits_list


# ────────────────────────────────────────────────────────────────────────
# 60 / 20 / 20 split
# ────────────────────────────────────────────────────────────────────────
def test_time_ordered_split_default_ratios():
    s = time_ordered_split(100)
    assert len(s.train) == 60
    assert len(s.val) == 20
    assert len(s.test) == 20


def test_split_indices_are_chronological():
    s = time_ordered_split(100)
    assert s.train.max() < s.val.min()
    assert s.val.max() < s.test.min()


def test_split_indices_cover_everything_no_overlap():
    s = time_ordered_split(123)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert set(all_idx) == set(range(123))
    assert len(all_idx) == len(set(all_idx))  # no overlap


def test_too_few_rows_rejected():
    with pytest.raises(ValueError):
        time_ordered_split(2)


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        time_ordered_split(100, train_fraction=0.7, val_fraction=0.5)  # > 1 total


# ────────────────────────────────────────────────────────────────────────
# Walk-forward CV
# ────────────────────────────────────────────────────────────────────────
def test_default_walk_forward_yields_five_splits():
    splits = walk_forward_splits_list(1000)
    assert len(splits) == 5


def test_each_split_is_train_then_val_chronological():
    splits = walk_forward_splits_list(1000)
    for train, val in splits:
        assert train.max() < val.min()


def test_expanding_train_grows_across_splits():
    splits = walk_forward_splits_list(1000, expanding=True)
    for (tr1, _), (tr2, _) in zip(splits, splits[1:]):
        assert len(tr2) > len(tr1)
        # Each subsequent train starts at 0 and extends further.
        assert tr1[0] == 0 and tr2[0] == 0
        assert tr2[-1] > tr1[-1]


def test_rolling_train_keeps_fixed_width():
    splits = walk_forward_splits_list(1000, expanding=False)
    widths = {len(tr) for tr, _ in splits}
    # Rolling window: all training arrays the same width.
    assert len(widths) == 1


def test_val_windows_are_contiguous_and_non_overlapping():
    splits = walk_forward_splits_list(1000)
    vals = [val for _, val in splits]
    for v1, v2 in zip(vals, vals[1:]):
        assert v1.max() < v2.min()
        # Contiguous: next val starts immediately after previous ends.
        assert v2.min() == v1.max() + 1


def test_train_never_touches_its_validation():
    splits = walk_forward_splits_list(1000)
    for tr, val in splits:
        assert set(tr).isdisjoint(set(val))


def test_short_series_still_yields_at_least_one_split():
    splits = walk_forward_splits_list(50, val_fraction_per_split=0.1)
    assert len(splits) >= 1


def test_invalid_val_fraction_rejected():
    with pytest.raises(ValueError):
        walk_forward_splits_list(1000, val_fraction_per_split=0.5)  # too big


def test_too_few_rows_rejected_walk_forward():
    with pytest.raises(ValueError):
        walk_forward_splits_list(3)
