"""Modèles PyTorch (LSTM, CNN 1D) et boucle d'entraînement."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMRegressor(nn.Module):
    """LSTM empilé ; la RUL est prédite à partir du dernier état caché."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):  # x : (batch, fenêtre, capteurs)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """Convolutions 1D le long de l'axe du temps, puis moyenne globale."""

    def __init__(self, n_features: int, channels: int = 32, kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))  # (batch, canaux, fenêtre)
        return self.head(h.mean(dim=-1)).squeeze(-1)


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    models = {"lstm": LSTMRegressor, "cnn": CNN1DRegressor}
    return models[name.lower()](n_features, **kwargs)


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train, y_train,
    X_val=None, y_val=None,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Entraîne avec arrêt anticipé sur la RMSE de validation.

    Retourne l'historique et la durée d'entraînement ; le modèle est remis
    à ses meilleurs poids.
    """
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_dl = _loader(X_train, y_train, batch_size, True)
    history = {"train_rmse": [], "val_rmse": []}
    best, best_state, wait = np.inf, None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        history["train_rmse"].append(float(np.sqrt(total / n)))

        if X_val is not None:
            val_rmse = float(np.sqrt(np.mean((predict(model, X_val, device=device) - y_val) ** 2)))
            history["val_rmse"].append(val_rmse)
            if val_rmse < best - 1e-3:
                best, wait = val_rmse, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"époque {epoch:3d}  RMSE entraînement {history['train_rmse'][-1]:6.2f}  validation {val_rmse:6.2f}")
            if wait >= patience:
                if verbose:
                    print(f"Arrêt anticipé à l'époque {epoch} (meilleure validation : {best:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["seconds"] = time.time() - start
    return history


@torch.no_grad()
def predict(model: nn.Module, X, batch_size: int = 1024, device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = [model(X[i : i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_mc_dropout(model: nn.Module, X, n_samples: int = 50, device: torch.device | None = None):
    """Intervalle d'incertitude par MC Dropout : le dropout reste actif à la prédiction.

    Retourne (moyenne, quantile 5 %, quantile 95 %).
    """
    device = device or next(model.parameters()).device
    model.train()  # active le dropout
    X = torch.as_tensor(X, dtype=torch.float32).to(device)
    draws = np.stack([model(X).cpu().numpy() for _ in range(n_samples)])
    model.eval()
    return draws.mean(0), np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
