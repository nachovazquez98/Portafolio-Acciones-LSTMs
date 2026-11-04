"""Series de FRED vía el endpoint CSV público (sin API key).

Se usan como indicadores de estrés/incertidumbre, en la línea del "mapa térmico
con alertas tempranas" (Martínez Ledesma, Jiménez-Preciado & Venegas-Martínez, 2022)
y del análisis de incertidumbre de política económica (Venegas-Martínez &
Jiménez-Preciado, 2025).
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from stockcast import config

log = logging.getLogger(__name__)
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"


def load_fred(series: dict[str, str] | None = None, refresh: bool = True, start: str = "1990-01-01") -> pd.DataFrame:
    """Matriz diaria fecha x serie (forward-filled). `series` = {id_fred: descripción}."""
    series = series if series is not None else config.load().get("fred", {})
    cache = config.path("data") / "fred.parquet"
    cached = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    if refresh and series:
        cols = {}
        for sid in series:
            try:
                r = requests.get(URL.format(sid=sid, start=start), timeout=30)
                r.raise_for_status()
                s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
                cols[sid] = pd.to_numeric(s, errors="coerce")
            except (requests.RequestException, ValueError) as e:
                log.warning("FRED %s falló: %s", sid, e)
                if sid in cached:
                    cols[sid] = cached[sid]
        if cols:  # sólo las series configuradas (las retiradas, como el high yield, se descartan)
            cached = pd.DataFrame(cols).sort_index()
            cached.to_parquet(cache)
    if cached.empty:
        return cached
    idx = pd.date_range(cached.index.min(), pd.Timestamp.today().normalize(), freq="D")
    return cached.reindex(idx).ffill().rename_axis("date")
