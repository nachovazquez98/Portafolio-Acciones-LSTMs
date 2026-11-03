"""LightGBM con regresión cuantil: un modelo global para todas las acciones.

Por qué: en la literatura (Gu, Kelly & Xiu 2020; benchmarks M5) los árboles de
gradient boosting igualan o superan a redes recurrentes en datos tabulares
financieros, entrenan en segundos en CPU y no requieren escalado ni GPU.
Entrenar un solo modelo con todas las acciones (panel) da muchos más datos que
un LSTM por acción y reduce sobreajuste.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


class QuantileGBM:
    name = "lightgbm"

    def __init__(self, quantiles: list[float], **params):
        self.quantiles = quantiles
        self.params = {"verbosity": -1, "n_jobs": -1, **params}
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int = 1) -> "QuantileGBM":
        self.columns = list(X.columns)
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = np.column_stack([self.models[q].predict(X[self.columns]) for q in self.quantiles])
        preds.sort(axis=1)  # evita cruce de cuantiles
        return pd.DataFrame(preds, index=X.index, columns=[f"q{q}" for q in self.quantiles])

    def feature_importance(self) -> pd.Series:
        m = self.models[0.5] if 0.5 in self.models else next(iter(self.models.values()))
        imp = m.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.columns).sort_values(ascending=False)
