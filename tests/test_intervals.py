"""Tests for §3.6 prediction-interval helpers."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.intervals import parametric_sigma, quantile_residual_sigma


def test_parametric_sigma_recovers_known_std():
    rng = np.random.default_rng(0)
    sigma_true = 1.7
    residuals = rng.normal(0, sigma_true, size=10_000)
    assert parametric_sigma(residuals) == pytest.approx(sigma_true, rel=0.05)


def test_quantile_residual_sigma_matches_parametric_on_gaussian():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1.0, size=10_000)
    assert quantile_residual_sigma(residuals) == pytest.approx(1.0, rel=0.1)


def test_quantile_residual_robust_to_outliers():
    """Heavy-tailed residuals: quantile sigma should be smaller than parametric."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1.0, size=1000)
    # Inject 1% extreme outliers
    base[:10] = 50.0
    p = parametric_sigma(base)
    q = quantile_residual_sigma(base)
    assert q < p, "quantile-based estimator should resist outliers"


def test_parametric_sigma_rejects_short_input():
    with pytest.raises(ValueError, match="at least 2"):
        parametric_sigma([0.5])


def test_quantile_sigma_rejects_invalid_quantiles():
    with pytest.raises(ValueError):
        quantile_residual_sigma([1, 2, 3, 4], quantile_low=0.5, quantile_high=0.4)


def test_zero_residuals_yields_zero_sigma():
    residuals = np.zeros(100)
    assert parametric_sigma(residuals) == 0.0
    assert quantile_residual_sigma(residuals) == 0.0
