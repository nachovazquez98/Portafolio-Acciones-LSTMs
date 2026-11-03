"""Réplica del método del notebook original, para compararlo con la misma validación.

Igual que `notebooks/multi-rnn-lstm-*-stocks.ipynb`:
- un LSTM por acción, entrenado con precios en *nivel* + indicadores técnicos en nivel
  (SMA, EMA, Bollinger, RSI, MACD), escalados con MinMax;
- predice el precio del día siguiente y, para h > 1, se alimenta de su propia
  predicción congelando el resto de variables (pronóstico recursivo).
Diferencia: hiperparámetros fijos en vez de 70 pruebas de Optuna (para que la
comparación no tarde horas) y escalador ajustado sólo con el train (el original lo
ajustaba con todo el histórico, lo que filtra información del futuro y lo favorecería).
Las bandas p10/p90 se derivan de la volatilidad, ya que el original no las producía.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm


def _level_features(close: pd.Series) -> pd.DataFrame:
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(20).mean()
    loss = (-delta.clip(upper=0)).rolling(20).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return pd.DataFrame({"close": close, "sma": sma, "ema": close.ewm(span=20, adjust=False).mean(),
                         "bb_up": sma + 2 * std, "bb_lo": sma - 2 * std,
                         "rsi": 100 - 100 / (1 + gain / loss.replace(0, 1e-9)),
                         "macd": macd, "macd_sig": macd.ewm(span=9, adjust=False).mean()}).bfill()


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])


class LegacyLSTMModel:
    name = "lstm_legacy"
    needs_close = True

    def __init__(self, quantiles: list[float], window: int = 60, hidden_size: int = 64,
                 epochs: int = 30, lr: float = 1e-3, batch_size: int = 64):
        self.quantiles, self.window, self.hidden = quantiles, window, hidden_size
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets, self.scalers = {}, {}

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "LegacyLSTMModel":
        self.horizon = horizon
        for t, g in X.groupby(level="ticker"):
            feats = _level_features(g["close"].droplevel("ticker")).to_numpy(np.float32)
            if len(feats) <= self.window + 10:
                continue
            lo, hi = feats.min(0), feats.max(0)
            self.scalers[t] = (lo, np.where(hi > lo, hi - lo, 1))
            z = (feats - lo) / self.scalers[t][1]
            Xs = np.stack([z[i - self.window:i] for i in range(self.window, len(z))])
            ys = z[self.window:, 0:1]
            net = _Net(z.shape[1], self.hidden).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(Xs), torch.tensor(ys)),
                batch_size=self.batch_size, shuffle=True)
            net.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(xb.to(self.device)), yb.to(self.device))
                    loss.backward()
                    opt.step()
            self.nets[t] = net.eval()
        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [f"q{q}" for q in self.quantiles]
        out = pd.DataFrame(np.nan, index=X.index, columns=cols)
        for t, g in X.groupby(level="ticker"):
            if t not in self.nets:
                continue
            lo, rng = self.scalers[t]
            feats = _level_features(g["close"].droplevel("ticker")).to_numpy(np.float32)
            z = (feats - lo) / rng
            ends = np.arange(self.window, len(z) + 1)
            if not len(ends):
                continue
            seq = torch.tensor(np.stack([z[e - self.window:e] for e in ends])).to(self.device)
            for _ in range(self.horizon):  # recursivo, como el notebook
                nxt = self.nets[t](seq)
                step = seq[:, -1:, :].clone()
                step[:, 0, 0] = nxt[:, 0]
                seq = torch.cat([seq[:, 1:, :], step], dim=1)
            pred_close = seq[:, -1, 0].cpu().numpy() * rng[0] + lo[0]
            last_close = feats[ends - 1, 0]
            med = np.log(np.clip(pred_close, 1e-6, None) / last_close)
            vol = g["vol_63"].to_numpy()[ends - 1] if "vol_63" in g else np.full(len(ends), 0.3)
            sigma = np.nan_to_num(vol, nan=0.3) / np.sqrt(252) * np.sqrt(self.horizon)
            idx = g.index[ends - 1]
            out.loc[idx, cols] = np.column_stack([med + norm.ppf(q) * sigma for q in self.quantiles])
        return out
