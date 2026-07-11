"""LSTM forecaster (§3.6).

§3.6 hyperparameter grid (verbatim):
    single-layer and two-layer variants
    hidden sizes {32, 64, 128}
    sequence length {30, 60} scrape intervals
    dropout {0.0, 0.2}
    optimiser Adam, learning rate 1e-3
    loss MSE
    early stopping on validation MAE with patience of ten epochs
    maximum 200 epochs

Stochasticity is controlled by the three-seed convention from
`docs/SEEDS.md`, applied via `forecasting.seeds.set_seeds`. CUDA
non-determinism is disabled. After fitting, the resulting state_dict is
bit-exactly reproducible from the same root seed — see
`tests/test_lstm_model.py::test_lstm_bit_exact_determinism`.

Sequence representation
-----------------------
Input is the raw multivariate sequence per §3.6. For Phase 3 the
multivariate input is just CPU (matching the §3.5 first family); Phase 4
extends to memory + request rate. The model consumes the last `seq_len`
samples of the standardised series and predicts the value `horizon_samples`
ahead.

Prediction interval
-------------------
Quantile-residual σ̂ from validation residuals (§3.6 for ML models).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Forecast, Forecaster, ForecasterFaultError
from .intervals import quantile_residual_sigma
from .seeds import SeedTriple, set_seeds


# §3.6 hyperparameter grid (single source of truth)
DEFAULT_HIDDEN_SIZES = [32, 64, 128]
DEFAULT_SEQ_LENS = [30, 60]
DEFAULT_DROPOUTS = [0.0, 0.2]
DEFAULT_LAYERS = [1, 2]
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 10
DEFAULT_MAX_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class LSTMHyperparams:
    num_layers: int
    hidden_size: int
    seq_len: int
    dropout: float


class LSTMForecaster(Forecaster):
    """PyTorch LSTM with §3.6 hyperparameter grid and three-seed determinism."""

    name = "lstm"

    def __init__(
        self,
        horizon_seconds: int = 30,
        sample_interval_seconds: int = 15,
        *,
        hidden_sizes: Optional[List[int]] = None,
        seq_lens: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        num_layers_grid: Optional[List[int]] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        patience: int = DEFAULT_PATIENCE,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 0,
        device: str = "auto",
    ):
        self.horizon_seconds = horizon_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.horizon_samples = max(1, round(horizon_seconds / sample_interval_seconds))
        self.hidden_sizes = list(hidden_sizes or DEFAULT_HIDDEN_SIZES)
        self.seq_lens = list(seq_lens or DEFAULT_SEQ_LENS)
        self.dropouts = list(dropouts or DEFAULT_DROPOUTS)
        self.num_layers_grid = list(num_layers_grid or DEFAULT_LAYERS)
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = int(seed)
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        self.device_preference = device

        self._best_model = None
        self._best_hp: Optional[LSTMHyperparams] = None
        self._best_val_mae: float = float("inf")
        self._val_residuals: np.ndarray = np.array([])
        self._train_mean: float = 0.0
        self._train_std: float = 1.0
        self._fit_seconds: float = 0.0
        self._applied_seeds: Optional[SeedTriple] = None
        self._device = None  # set during fit from device_preference

    # ------------------------------------------------------------------ #
    # Forecaster interface                                                #
    # ------------------------------------------------------------------ #
    def fit(self, history: pd.Series) -> "LSTMForecaster":
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError(
                "LSTMForecaster requires torch. Install via uv: `uv sync`."
            ) from e

        # Apply three-seed discipline; record what we applied.
        self._applied_seeds = set_seeds(self.seed, strict_determinism=True)

        self._device = _resolve_device(torch, self.device_preference)
        print(f"  LSTM device: {self._device}")

        clean = history.dropna().astype(float).values
        max_seq = max(self.seq_lens)
        if len(clean) < max_seq + self.horizon_samples + 30:
            raise ForecasterFaultError(
                f"LSTM needs ≥ {max_seq + self.horizon_samples + 30} clean samples "
                f"(have {len(clean)})"
            )

        # Standardise on training portion only — 80/20 time-ordered split.
        split = int(len(clean) * 0.8)
        train_raw, val_raw = clean[:split], clean[split:]
        self._train_mean = float(np.mean(train_raw))
        self._train_std = float(np.std(train_raw, ddof=1)) or 1.0
        train_std = (train_raw - self._train_mean) / self._train_std
        val_std = (val_raw - self._train_mean) / self._train_std


        best_mae = float("inf")
        best_model = None
        best_hp: Optional[LSTMHyperparams] = None
        best_val_preds: Optional[np.ndarray] = None
        best_val_true: Optional[np.ndarray] = None

        t0 = time.perf_counter()
        for n_layers in self.num_layers_grid:
            for hidden in self.hidden_sizes:
                for seq_len in self.seq_lens:
                    for dropout in self.dropouts:
                        if n_layers == 1 and dropout > 0:
                            # PyTorch dropout has no effect with 1 layer; skip duplicate.
                            continue

                        # Re-seed for *each* candidate so the grid itself is
                        # deterministic regardless of search order.
                        set_seeds(self.seed, strict_determinism=True)

                        hp = LSTMHyperparams(n_layers, hidden, seq_len, dropout)
                        try:
                            model, val_mae, val_preds, val_true = self._train_one(
                                train_std, val_std, hp, torch=torch, nn=nn,
                                device=self._device,
                            )
                        except Exception:
                            continue
                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_model = model
                            best_hp = hp
                            best_val_preds = val_preds
                            best_val_true = val_true

        if best_model is None:
            raise ForecasterFaultError("LSTM: no candidate trained successfully")

        self._best_model = best_model
        self._best_hp = best_hp
        self._best_val_mae = float(best_mae)
        # Un-standardise the val residuals so σ is in the original units.
        self._val_residuals = (
            np.asarray(best_val_true) - np.asarray(best_val_preds)
        ) * self._train_std
        self._fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, history: pd.Series, horizon_seconds: int) -> Forecast:
        if self._best_model is None:
            raise ForecasterFaultError("LSTM: predict called before fit")
        if horizon_seconds != self.horizon_seconds:
            raise ValueError(
                f"LSTM was trained for horizon {self.horizon_seconds}s; "
                f"asked for {horizon_seconds}s"
            )

        try:
            import torch
        except ImportError as e:
            raise RuntimeError("LSTMForecaster requires torch") from e

        clean = history.dropna().astype(float).values
        seq_len = self._best_hp.seq_len
        if len(clean) < seq_len:
            raise ForecasterFaultError(
                f"need ≥ {seq_len} samples to feed the LSTM (have {len(clean)})"
            )

        tail = clean[-seq_len:]
        tail_std = (tail - self._train_mean) / self._train_std
        device = self._device if self._device is not None else torch.device("cpu")
        x = torch.tensor(tail_std, dtype=torch.float32).reshape(1, seq_len, 1).to(device)
        self._best_model.to(device)
        self._best_model.eval()
        with torch.no_grad():
            y_std = float(self._best_model(x).cpu().numpy().ravel()[0])
        point = y_std * self._train_std + self._train_mean

        if not math.isfinite(point):
            raise ForecasterFaultError(f"LSTM produced non-finite point: {point}")

        if len(self._val_residuals) < 4:
            raise ForecasterFaultError("not enough validation residuals for σ")
        sigma = quantile_residual_sigma(self._val_residuals)
        return Forecast(point=point, sigma=sigma)

    # ------------------------------------------------------------------ #
    # Parsimony tiebreaker                                                #
    # ------------------------------------------------------------------ #
    def n_parameters(self) -> int:
        if self._best_model is None:
            return 0
        return sum(p.numel() for p in self._best_model.parameters())

    # ------------------------------------------------------------------ #
    # Artefact persistence                                                #
    # ------------------------------------------------------------------ #
    def save_artifact(self, path: str | Path) -> Path:
        """Persist a trained LSTM forecaster for live-controller loading."""
        if self._best_model is None or self._best_hp is None:
            raise ForecasterFaultError("LSTM: cannot save before fit")
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("LSTMForecaster requires torch") from e

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        # Always save on CPU so the artefact is device-portable.
        cpu_state = {k: v.cpu() for k, v in self._best_model.state_dict().items()}
        torch.save({"state_dict": cpu_state}, out / "model.pt")

        metadata = {
            "forecaster": self.name,
            "horizon_seconds": self.horizon_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "horizon_samples": self.horizon_samples,
            "best_hp": asdict(self._best_hp),
            "best_val_mae": self._best_val_mae,
            "val_residuals": self._val_residuals.tolist(),
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device_preference": self.device_preference,
            "fit_seconds": self._fit_seconds,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return out

    @classmethod
    def load_artifact(cls, path: str | Path) -> "LSTMForecaster":
        """Load a trained LSTM forecaster saved by `save_artifact`."""
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("LSTMForecaster requires torch") from e

        src = Path(path)
        metadata = json.loads((src / "metadata.json").read_text())
        hp = LSTMHyperparams(**metadata["best_hp"])

        obj = cls(
            horizon_seconds=int(metadata["horizon_seconds"]),
            sample_interval_seconds=int(metadata["sample_interval_seconds"]),
            hidden_sizes=[int(hp.hidden_size)],
            seq_lens=[int(hp.seq_len)],
            dropouts=[float(hp.dropout)],
            num_layers_grid=[int(hp.num_layers)],
            learning_rate=float(metadata["learning_rate"]),
            patience=int(metadata["patience"]),
            max_epochs=int(metadata["max_epochs"]),
            batch_size=int(metadata["batch_size"]),
            seed=int(metadata["seed"]),
            device=str(metadata.get("device_preference", "auto")),
        )
        model = _LSTMModel(
            input_size=1,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            dropout=hp.dropout,
        )
        try:
            checkpoint = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(src / "model.pt", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        obj._best_model = model
        obj._best_hp = hp
        obj._best_val_mae = float(metadata["best_val_mae"])
        obj._val_residuals = np.asarray(metadata["val_residuals"], dtype=float)
        obj._train_mean = float(metadata["train_mean"])
        obj._train_std = float(metadata["train_std"])
        obj._fit_seconds = float(metadata.get("fit_seconds", 0.0))
        return obj

    # ------------------------------------------------------------------ #
    # Training one candidate                                              #
    # ------------------------------------------------------------------ #
    def _train_one(
        self,
        train_std: np.ndarray,
        val_std: np.ndarray,
        hp: LSTMHyperparams,
        *,
        torch,
        nn,
        device,
    ) -> Tuple[object, float, np.ndarray, np.ndarray]:
        """Train a single (num_layers, hidden, seq_len, dropout) candidate.

        Returns (model, val_mae, val_preds, val_true) on standardised scale.
        """
        # Build sequence-target pairs from standardised series.
        Xtr, ytr = _make_sequences(train_std, hp.seq_len, self.horizon_samples)
        Xva, yva = _make_sequences(val_std, hp.seq_len, self.horizon_samples)
        if len(Xtr) < 8 or len(Xva) < 4:
            raise RuntimeError(
                f"insufficient sequence pairs (train={len(Xtr)}, val={len(Xva)}) "
                f"for seq_len={hp.seq_len}"
            )

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1).to(device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
        Xva_t = torch.tensor(Xva, dtype=torch.float32).unsqueeze(-1).to(device)
        yva_t = torch.tensor(yva, dtype=torch.float32).to(device)

        model = _LSTMModel(
            input_size=1,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            dropout=hp.dropout,
        ).to(device)
        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        best_val_mae = float("inf")
        patience_left = self.patience
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for epoch in range(self.max_epochs):
            model.train()
            # Mini-batches (no shuffling within an epoch to keep determinism
            # straightforward — for sub-minute autoscaling time series this
            # is the safer reproducibility choice).
            for i in range(0, len(Xtr_t), self.batch_size):
                xb = Xtr_t[i : i + self.batch_size]
                yb = ytr_t[i : i + self.batch_size]
                optim.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t).squeeze(-1)
                val_mae = float((val_pred - yva_t).abs().mean().item())

            if val_mae < best_val_mae - 1e-9:
                best_val_mae = val_mae
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t).squeeze(-1).cpu().numpy()
        return model, best_val_mae, val_pred, yva_t.cpu().numpy()


# ---------------------------------------------------------------------- #
# Model class — lazy-imported torch                                       #
# ---------------------------------------------------------------------- #
def _LSTMModel(*args, **kwargs):
    """Factory so importing this module does not require torch at import time."""
    import torch.nn as nn
    import torch

    class _Model(nn.Module):
        def __init__(self, input_size: int, hidden_size: int,
                     num_layers: int, dropout: float):
            super().__init__()
            # nn.LSTM dropout only applies between stacked layers (num_layers > 1).
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)

    return _Model(*args, **kwargs)


def _make_sequences(series: np.ndarray, seq_len: int, horizon: int):
    """Slide a window of length seq_len; target = value horizon steps ahead."""
    X, y = [], []
    for i in range(len(series) - seq_len - horizon + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len + horizon - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _resolve_device(torch, preference: str):
    """Resolve the requested PyTorch device with clear availability errors."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ForecasterFaultError("LSTM device 'cuda' requested but CUDA is unavailable")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise ForecasterFaultError("LSTM device 'mps' requested but Apple MPS is unavailable")
        return torch.device("mps")

    # Auto keeps the previous preference order. For full real-data runs on
    # Apple Silicon, use LSTM_DEVICE=cpu to avoid known PyTorch/MPS LSTM hangs.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
