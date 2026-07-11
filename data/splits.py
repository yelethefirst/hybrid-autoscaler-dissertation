"""Time-ordered splits and walk-forward cross-validation (§3.5).

§3.5: "Train, validation and test splits are time-ordered and
non-overlapping. The first sixty per cent of each scraped window is used
for training, the next twenty per cent for validation (hyperparameter
selection) and the final twenty per cent for held-out test reporting.
Walk-forward cross-validation with five splits is applied during model
selection to provide statistical confidence in performance estimates."

Two helpers:

    time_ordered_split(n, ...) → (train_idx, val_idx, test_idx)

    walk_forward_splits(n, n_splits=5, ...) → iterator of (train_idx, val_idx)

Both return numpy index arrays into a chronologically-sorted DataFrame.
The caller is responsible for ensuring the input is sorted by timestamp
before indexing — the splits *only* see counts, not values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeOrderedSplit:
    """The 60 / 20 / 20 §3.5 split, in index form."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        # Sanity: train < val < test in chronological order.
        if len(self.train) and len(self.val):
            assert self.train.max() < self.val.min(), "train and val overlap"
        if len(self.val) and len(self.test):
            assert self.val.max() < self.test.min(), "val and test overlap"


def time_ordered_split(
    n: int,
    *,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> TimeOrderedSplit:
    """Return the 60/20/20 §3.5 split as index arrays.

    Parameters
    ----------
    n : int
        Number of rows in the (chronologically sorted) input.
    train_fraction, val_fraction : float
        Sizes of the train and val portions; the remainder is test.
        Defaults match §3.5 exactly. Both must be in (0, 1) and their
        sum must be in (0, 1).
    """
    if n < 3:
        raise ValueError("need at least 3 rows to make a 3-way split")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if not 0 < (train_fraction + val_fraction) < 1:
        raise ValueError("train_fraction + val_fraction must be < 1 (leave room for test)")

    train_end = max(1, int(n * train_fraction))
    val_end = max(train_end + 1, int(n * (train_fraction + val_fraction)))
    val_end = min(val_end, n - 1)        # ensure test has ≥ 1 row

    train = np.arange(0, train_end, dtype=np.int64)
    val = np.arange(train_end, val_end, dtype=np.int64)
    test = np.arange(val_end, n, dtype=np.int64)

    return TimeOrderedSplit(train=train, val=val, test=test)


def walk_forward_splits(
    n: int,
    *,
    n_splits: int = 5,
    initial_train_fraction: float = 0.50,
    val_fraction_per_split: float = 0.10,
    expanding: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield Hyndman-style walk-forward (train_idx, val_idx) pairs.

    With defaults (n_splits=5, initial_train_fraction=0.50,
    val_fraction_per_split=0.10) the splits divide the chronologically
    sorted series as follows:

        split 1  train: [0, 0.5N)  val: [0.5N, 0.6N)
        split 2  train: [0, 0.6N)  val: [0.6N, 0.7N)
        split 3  train: [0, 0.7N)  val: [0.7N, 0.8N)
        split 4  train: [0, 0.8N)  val: [0.8N, 0.9N)
        split 5  train: [0, 0.9N)  val: [0.9N, 1.0N)

    Parameters
    ----------
    n_splits : int
        Number of (train, val) pairs to yield. §3.5 specifies 5.
    initial_train_fraction : float
        Fraction of n that forms the training window in split 1.
    val_fraction_per_split : float
        Fraction of n that each validation window covers.
    expanding : bool
        If True (default, Hyndman), training is *expanding* — each split
        grows the training window. If False, training is *rolling* (fixed
        width) — for very long series where ancient history is irrelevant.
        §3.5 default is expanding.
    """
    if n < 4:
        raise ValueError("need at least 4 rows for walk-forward CV")
    if n_splits < 1:
        raise ValueError("n_splits must be ≥ 1")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be in (0, 1)")
    if not 0 < val_fraction_per_split <= (1 - initial_train_fraction) / n_splits:
        # Make sure the last split's val window still fits.
        # Strict ≤ here is fine: equality means the last val ends exactly at n.
        raise ValueError(
            f"val_fraction_per_split={val_fraction_per_split} doesn't fit "
            f"n_splits={n_splits} after initial_train_fraction="
            f"{initial_train_fraction} (would overrun n)"
        )

    initial_train_end = max(1, int(n * initial_train_fraction))
    val_width = max(1, int(n * val_fraction_per_split))

    for split in range(n_splits):
        train_end = initial_train_end + split * val_width
        val_start = train_end
        val_end = min(val_start + val_width, n)
        if val_start >= n:
            break
        train_start = 0 if expanding else max(0, train_end - initial_train_end)
        train_idx = np.arange(train_start, train_end, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)
        yield train_idx, val_idx


def walk_forward_splits_list(n: int, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
    """List version of `walk_forward_splits` for convenience in tests / notebooks."""
    return list(walk_forward_splits(n, **kwargs))
