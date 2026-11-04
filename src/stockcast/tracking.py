"""Seguimiento en vivo (forward test): ¿cómo les fue de verdad a los pronósticos ya emitidos?

Cada corrida guarda sus pronósticos en artifacts/<mercado>/forecast_history.parquet. Aquí se cruzan con
los precios reales una vez que llega su fecha objetivo. Es la evidencia más limpia posible: no hubo
ninguna selección de modelo ni de periodo después de ver el resultado (McLean & Pontiff 2016 muestran
que el desempeño "en papel" suele caer al operar en vivo; esto lo mide).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockcast import pipeline
from stockcast.data import market


def realized(market_key: str) -> pd.DataFrame:
    """Pronósticos vencidos con su retorno real."""
    path = pipeline.artifacts_dir(market_key) / "forecast_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    fh = pd.read_parquet(path)
    if fh.empty:
        return fh
    prices = market.load_prices(sorted(fh["ticker"].unique()), refresh=False)
    close = market.close_matrix(prices)
    last_day = close.index.max()
    fh = fh[pd.to_datetime(fh["target_date"]) <= last_day].copy()
    if fh.empty:
        return fh
    fh["as_of"] = pd.to_datetime(fh["as_of"])
    fh["target_date"] = pd.to_datetime(fh["target_date"])

    def px(row, col):
        s = close.get(row["ticker"])
        return s.asof(row[col]) if s is not None else np.nan

    fh["close_target"] = fh.apply(px, axis=1, col="target_date")
    fh["ret_real"] = fh["close_target"] / fh["close"] - 1
    return fh.dropna(subset=["ret_real"])


def summary(market_key: str) -> pd.DataFrame:
    """Por plazo: nº de pronósticos vencidos, IC real y ganancia long-short real (20% sup. − 20% inf.)."""
    r = realized(market_key)
    if r.empty:
        return pd.DataFrame()
    rows = []
    for h, g in r.groupby("horizon"):
        ics, ls = [], []
        for _, day in g.groupby("as_of"):
            if len(day) < 5 or day["ret_q0.5"].nunique() < 2:
                continue
            ics.append(day["ret_q0.5"].corr(day["ret_real"], method="spearman"))
            q = day["ret_q0.5"].rank(pct=True)
            ls.append(0.5 * (day.loc[q > 0.8, "ret_real"].mean() - day.loc[q <= 0.2, "ret_real"].mean()))
        inside = ((g["ret_real"] >= g["ret_q0.1"]) & (g["ret_real"] <= g["ret_q0.9"])).mean()
        rows.append({"horizon": int(h), "fechas": g["as_of"].nunique(), "pronosticos": len(g),
                     "ic_real": float(np.nanmean(ics)) if ics else np.nan,
                     "long_short_real": float(np.nanmean(ls)) if ls else np.nan,
                     "cobertura_real": float(inside)})
    return pd.DataFrame(rows)
