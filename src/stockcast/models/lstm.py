"""LSTM global con pérdida cuantil (versión corregida del modelo del notebook original).

Cambios respecto al notebook:
- Un solo modelo para todas las acciones (más datos, menos sobreajuste).
- Predice retornos logarítmicos a h días directamente, con 3 cuantiles.
- El escalado se ajusta sólo con datos de entrenamiento (sin fuga de información).

`predict(X)` necesita que X incluya al menos `window` filas previas de cada
ticker; las filas sin historia suficiente devuelven NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_out: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class LSTMModel:
    name = "lstm"

    def __init__(self, quantiles: list[float], window: int = 60, hidden_size: int = 64,
                 epochs: int = 30, lr: float = 1e-3, batch_size: int = 256):
        self.quantiles, self.window, self.hidden = quantiles, window, hidden_size
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = (X[self.columns] - self.mu) / self.sd
        return Z.clip(-5, 5).fillna(0.0).astype(np.float32)

    def _sequences(self, Z: pd.DataFrame):
        """Devuelve (array de secuencias, índice de la fila final de cada secuencia)."""
        seqs, idx = [], []
        for _, g in Z.groupby(level="ticker", sort=False):
            arr = g.to_numpy()
            for i in range(self.window - 1, len(arr)):
                seqs.append(arr[i - self.window + 1:i + 1])
                idx.append(g.index[i])
        return np.stack(seqs) if seqs else np.empty((0, self.window, Z.shape[1]), np.float32), idx

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "LSTMModel":
        self.columns = [c for c in X.columns if X[c].dtype != object]
        self.mu, self.sd = X[self.columns].mean(), X[self.columns].std().replace(0, 1)
        Z = self._scale(X).sort_index(level=["ticker", "date"])
        seqs, idx = self._sequences(Z)
        target = y.reindex(idx).to_numpy(np.float32)
        ok = ~np.isnan(target)
        self.y_sd = float(np.nanstd(target)) or 1.0
        Xt = torch.tensor(seqs[ok])
        yt = torch.tensor(target[ok] / self.y_sd).unsqueeze(1)
        self.net = _Net(len(self.columns), self.hidden, len(self.quantiles)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        q = torch.tensor(self.quantiles, device=self.device).unsqueeze(0)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                             batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                err = yb - self.net(xb)
                loss = torch.maximum(q * err, (q - 1) * err).mean()  # pérdida pinball
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._scale(X).sort_index(level=["ticker", "date"])
        seqs, idx = self._sequences(Z)
        cols = [f"q{q}" for q in self.quantiles]
        out = pd.DataFrame(np.nan, index=X.index, columns=cols)
        if len(seqs):
            self.net.eval()
            preds = []
            for i in range(0, len(seqs), 4096):
                preds.append(self.net(torch.tensor(seqs[i:i + 4096]).to(self.device)).cpu().numpy())
            p = np.sort(np.concatenate(preds) * self.y_sd, axis=1)
            out.loc[pd.MultiIndex.from_tuples(idx, names=X.index.names)] = p
        return out
