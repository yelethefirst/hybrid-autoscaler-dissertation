"""Forecasting error metrics (§3.6 + §3.11).

§3.6 selects models on **RMSE**; reports MAE alongside; and logs a
pinball-loss robustness check. All three are implemented here as pure
functions over numpy arrays so they can be used inside both the model-
selection harness and the Chapter-4 analysis notebooks without coupling.
"""

from __future__ import annotations

from typing import Union

import numpy as np

Array = Union[np.ndarray, list, tuple]


def _to_array(x: Array) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rmse(y_true: Array, y_pred: Array) -> float:
    """Root Mean Squared Error. §3.6 primary selection metric."""
    yt, yp = _to_array(y_true), _to_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("empty inputs")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Array, y_pred: Array) -> float:
    """Mean Absolute Error. §3.6 secondary metric, reported alongside RMSE."""
    yt, yp = _to_array(y_true), _to_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("empty inputs")
    return float(np.mean(np.abs(yt - yp)))


def pinball_loss(y_true: Array, y_pred: Array, quantile: float = 0.7) -> float:
    """Pinball (quantile) loss. §3.6 logged robustness check.

    The §3.6 footnote notes that an asymmetric loss biased towards
    under-prediction (e.g. pinball at q > 0.5) is the cost-aligned choice
    for autoscaling. RMSE is retained as primary because the decision
    engine's k·σ term already provides an asymmetric safety bias; pinball
    is logged so the choice is auditable.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    yt, yp = _to_array(y_true), _to_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("empty inputs")
    err = yt - yp
    # Under-prediction (yt > yp) → err > 0 → penalty quantile · err
    # Over-prediction  (yt < yp) → err < 0 → penalty (quantile-1) · err = (1-quantile)·|err|
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))


def mape(y_true: Array, y_pred: Array, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error. Reported for context; not used in selection."""
    yt, yp = _to_array(y_true), _to_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("empty inputs")
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
