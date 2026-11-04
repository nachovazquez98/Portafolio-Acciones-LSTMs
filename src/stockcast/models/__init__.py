"""Modelos de pronóstico con interfaz común: fit(X, y, horizon) / predict(X) -> DataFrame de cuantiles."""
from __future__ import annotations

from stockcast.models.baselines import NaiveModel
from stockcast.models.gbm import QuantileGBM
from stockcast.models.linear import EnsembleModel, MomentumModel, RidgeModel

ENGINES = ["naive", "momentum", "ridge", "lightgbm", "ensemble"]


def make_model(engine: str, quantiles: list[float], params: dict | None = None):
    params = params or {}
    if engine == "lightgbm":
        return QuantileGBM(quantiles, **params)
    if engine == "naive":
        return NaiveModel(quantiles)
    if engine == "momentum":
        return MomentumModel(quantiles)
    if engine == "ridge":
        return RidgeModel(quantiles, **params)
    if engine == "ensemble":
        return EnsembleModel(quantiles, [QuantileGBM(quantiles, **params.get("lightgbm", {})),
                                         RidgeModel(quantiles, **params.get("ridge", {}))])
    raise ValueError(f"Motor desconocido: {engine}. Opciones: {ENGINES}")
