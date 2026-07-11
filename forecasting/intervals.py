"""Prediction-interval helpers (§3.6).

§3.6: "Forecast outputs include the point estimate and a
one-standard-deviation prediction interval, computed from the model's
residual distribution on the validation set (parametric for SARIMA and
Holt-Winters; quantile-residual for XGBoost and LSTM)."

This module provides:

    parametric_sigma(residuals)   → σ from std of residuals (Gaussian)
    quantile_residual_sigma(res)  → σ from interquantile width
                                    (more robust to heavy tails)

Both return a single non-negative float, matching the contract of
`forecasting.base.Forecast.sigma`.
"""

from __future__ import annotations

from typing import Union

import numpy as np

Array = Union[np.ndarray, list, tuple]


def parametric_sigma(residuals: Array, ddof: int = 1) -> float:
    """σ̂ = sample std of validation residuals (Gaussian assumption).

    Used for SARIMA and Holt-Winters per §3.6, where the residuals are
    well-approximated as Gaussian after the parametric model has removed
    the structure.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    if len(r) < 2:
        raise ValueError("need at least 2 residuals for sample std")
    s = float(np.std(r, ddof=ddof))
    return max(s, 0.0)


def quantile_residual_sigma(residuals: Array, quantile_low: float = 0.1587,
                            quantile_high: float = 0.8413) -> float:
    """σ̂ from a one-σ interquantile width of the empirical residual distribution.

    Default quantiles are the standard-normal ±1σ points (16th and 84th
    percentiles), giving the most direct empirical analogue of "one σ"
    without assuming a parametric form. Used for XGBoost and LSTM per
    §3.6, whose residuals often have heavier tails than the model class
    assumes.

    For symmetric Gaussian noise this returns the same value as
    `parametric_sigma`; for heavy-tailed residuals it is more robust to
    outliers than the moment-based std.
    """
    if not 0 < quantile_low < quantile_high < 1:
        raise ValueError("need 0 < quantile_low < quantile_high < 1")
    r = np.asarray(residuals, dtype=float).ravel()
    if len(r) < 4:
        raise ValueError("need at least 4 residuals for quantile estimate")
    q_low, q_high = np.quantile(r, [quantile_low, quantile_high])
    # The (q_high - q_low) width spans ≈ 2σ for Gaussian; divide by 2.
    return max(float((q_high - q_low) / 2.0), 0.0)
