"""Tests for the LSTM forecaster (§3.6).

Includes the bit-exact determinism check required by §3.6.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from data.synthetic import generate, to_wide
from forecasting import LSTMForecaster
from forecasting.base import ForecasterFaultError
from forecasting.lstm_model import LSTMHyperparams, _LSTMModel
from forecasting.seeds import verify_bit_exact


def _periodic_series(n_samples: int = 300):
    df_long = generate(workload="periodic", duration_seconds=n_samples * 15, seed=0)
    wide = to_wide(df_long, service="frontend")
    return wide.set_index("timestamp")["cpu"]


def _tiny_lstm(seed: int = 0) -> LSTMForecaster:
    """Smallest valid LSTM for fast tests."""
    return LSTMForecaster(
        horizon_seconds=30,
        sample_interval_seconds=15,
        hidden_sizes=[16],
        seq_lens=[20],
        dropouts=[0.0],
        num_layers_grid=[1],
        max_epochs=5,
        patience=2,
        seed=seed,
        device="cpu",
    )


def test_lstm_fit_predict_smoke():
    s = _periodic_series(n_samples=300)
    fc = _tiny_lstm()
    fc.fit(s)
    f = fc.predict(s, horizon_seconds=30)
    assert np.isfinite(f.point)
    assert f.sigma >= 0


def test_lstm_predict_before_fit_raises():
    fc = _tiny_lstm()
    s = _periodic_series(n_samples=300)
    with pytest.raises(ForecasterFaultError, match="before fit"):
        fc.predict(s, horizon_seconds=30)


def test_lstm_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        LSTMForecaster(device="tpu")


def test_lstm_bit_exact_determinism():
    """§3.6 reproducibility claim: same seed → bit-exact state_dict."""
    s = _periodic_series(n_samples=300)
    fc_a = _tiny_lstm(seed=42)
    fc_a.fit(s)
    fc_b = _tiny_lstm(seed=42)
    fc_b.fit(s)
    assert verify_bit_exact(
        fc_a._best_model.state_dict(),
        fc_b._best_model.state_dict(),
    )


def test_lstm_different_seeds_diverge():
    s = _periodic_series(n_samples=300)
    fc_a = _tiny_lstm(seed=1)
    fc_a.fit(s)
    fc_b = _tiny_lstm(seed=2)
    fc_b.fit(s)
    assert not verify_bit_exact(
        fc_a._best_model.state_dict(),
        fc_b._best_model.state_dict(),
    )


def test_lstm_n_parameters_positive_after_fit():
    s = _periodic_series(n_samples=300)
    fc = _tiny_lstm()
    fc.fit(s)
    assert fc.n_parameters() > 0


def test_lstm_horizon_mismatch_rejected():
    s = _periodic_series(n_samples=300)
    fc = _tiny_lstm()
    fc.fit(s)
    with pytest.raises(ValueError, match="horizon"):
        fc.predict(s, horizon_seconds=60)


def test_lstm_artifact_round_trip(tmp_path):
    s = _periodic_series(n_samples=40)
    fc = LSTMForecaster(
        horizon_seconds=30,
        sample_interval_seconds=15,
        hidden_sizes=[4],
        seq_lens=[3],
        dropouts=[0.0],
        num_layers_grid=[1],
        seed=7,
    )
    fc._best_hp = LSTMHyperparams(num_layers=1, hidden_size=4, seq_len=3, dropout=0.0)
    fc._best_model = _LSTMModel(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)
    fc._best_val_mae = 0.01
    fc._val_residuals = np.asarray([0.10, -0.05, 0.02, -0.01], dtype=float)
    fc._train_mean = float(s.mean())
    fc._train_std = float(s.std(ddof=1))
    before = fc.predict(s, horizon_seconds=30)

    artifact_dir = fc.save_artifact(tmp_path / "frontend-lstm")
    assert (artifact_dir / "model.pt").exists()
    assert (artifact_dir / "metadata.json").exists()

    loaded = LSTMForecaster.load_artifact(artifact_dir)
    after = loaded.predict(s, horizon_seconds=30)
    assert after.point == pytest.approx(before.point)
    assert after.sigma == pytest.approx(before.sigma)
