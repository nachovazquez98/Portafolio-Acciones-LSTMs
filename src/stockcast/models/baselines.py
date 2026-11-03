"""Baseline de caminata aleatoria: retorno esperado 0 y bandas según volatilidad histórica.

Cualquier modelo que no le gane a esto en datos fuera de muestra no aporta valor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class NaiveModel:
    name = "naive"

    def __init__(self, quantiles: list[float]):
        self.quantiles = quantiles
        self.vol_col = "vol_63"

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "NaiveModel":
        self.horizon = horizon
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        sigma = X[self.vol_col].fillna(X[self.vol_col].median()) / np.sqrt(252) * np.sqrt(self.horizon)
        return pd.DataFrame({f"q{q}": norm.ppf(q) * sigma.to_numpy() for q in self.quantiles}, index=X.index)
