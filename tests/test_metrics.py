"""Tests for forecasting metrics (§3.6 + §3.11)."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.metrics import mae, mape, pinball_loss, rmse


def test_rmse_zero_for_perfect_prediction():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # diffs = [-1, 0, 1] → squared = [1, 0, 1] → mean = 2/3 → sqrt(2/3)
    assert rmse([1, 2, 3], [2, 2, 2]) == pytest.approx(np.sqrt(2/3))


def test_mae_known_value():
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2/3)


def test_pinball_under_prediction_penalised_more_at_high_q():
    """At q=0.9, under-prediction (truth > pred) is heavily weighted."""
    truth = [10.0]
    under = pinball_loss(truth, [5.0], quantile=0.9)
    over = pinball_loss(truth, [15.0], quantile=0.9)
    assert under > over


def test_pinball_symmetric_at_q_05():
    truth = [10.0]
    assert pinball_loss(truth, [5.0], quantile=0.5) == pytest.approx(
        pinball_loss(truth, [15.0], quantile=0.5)
    )


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape"):
        rmse([1, 2], [1, 2, 3])


def test_empty_inputs_rejected():
    with pytest.raises(ValueError, match="empty"):
        mae([], [])


def test_invalid_quantile_rejected():
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1], [1], quantile=0)


def test_mape_handles_zero_truths():
    # eps prevents division by zero; result should be finite.
    val = mape([0, 1, 2], [0.1, 1.0, 2.0])
    assert np.isfinite(val)
