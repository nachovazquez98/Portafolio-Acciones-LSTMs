"""Modelos simples para comparar contra LightGBM/LSTM.

- MomentumModel: una sola señal clásica (momentum 6-1 meses, Jegadeesh & Titman 1993)
  con un coeficiente ajustado por MCO. Sin features macro.
- RidgeModel: regresión lineal regularizada sobre todas las features; los cuantiles
  salen de los residuos de entrenamiento.
- EnsembleModel: promedio de LightGBM y Ridge (combinar modelos distintos suele
  reducir varianza; Timmermann 2006, "Forecast Combinations").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge


class MomentumModel:
    name = "momentum"

    def __init__(self, quantiles: list[float]):
        self.quantiles = quantiles

    @staticmethod
    def _signal(X: pd.DataFrame) -> pd.Series:
        return (X["ret_126"] - X["ret_21"]).fillna(0)

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "MomentumModel":
        s = self._signal(X)
        self.beta = float(np.cov(s, y)[0, 1] / s.var()) if s.var() > 0 else 0.0
        self.alpha = float(y.mean() - self.beta * s.mean())
        self.horizon = horizon
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        med = self.alpha + self.beta * self._signal(X).to_numpy()
        sigma = X["vol_63"].fillna(X["vol_63"].median()).to_numpy() / np.sqrt(252) * np.sqrt(self.horizon)
        return pd.DataFrame({f"q{q}": med + norm.ppf(q) * sigma for q in self.quantiles}, index=X.index)


class RidgeModel:
    name = "ridge"

    def __init__(self, quantiles: list[float], alpha: float = 300.0):
        self.quantiles, self.alpha = quantiles, alpha

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.columns] - self.mu) / self.sd).clip(-5, 5).fillna(0).to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "RidgeModel":
        self.columns = list(X.columns)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        Z = self._z(X)
        self.model = Ridge(alpha=self.alpha).fit(Z, y)
        resid = y.to_numpy() - self.model.predict(Z)
        self.offsets = {q: float(np.quantile(resid, q) - np.quantile(resid, 0.5)) for q in self.quantiles}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        med = self.model.predict(self._z(X))
        return pd.DataFrame({f"q{q}": med + self.offsets[q] for q in self.quantiles}, index=X.index)


class EnsembleModel:
    name = "ensemble"

    def __init__(self, quantiles: list[float], members: list):
        self.quantiles, self.members = quantiles, members

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "EnsembleModel":
        for m in self.members:
            m.fit(X, y, horizon)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return sum(m.predict(X) for m in self.members) / len(self.members)

    def feature_importance(self) -> pd.Series:
        for m in self.members:
            if hasattr(m, "feature_importance"):
                return m.feature_importance()
        return pd.Series(dtype=float)
