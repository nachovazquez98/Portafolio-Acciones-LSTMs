"""CETES desde la API SIE de Banxico (requiere token gratuito).

Token: https://www.banxico.org.mx/SieAPIRest/service/v1/token  → guardarlo en la variable de entorno
BANXICO_TOKEN (p. ej. en ~/.bashrc o en el archivo .env del servicio de systemd).

Series (tasa de rendimiento de la subasta primaria, % anual, semanal):
  SF43936 CETES 28 días · SF43939 91 días · SF43942 182 días · SF43945 364 días

Como un CETE es un bono cupón cero, el rendimiento de *mantener* un plazo m se aproxima día a día con
  r_t = y_{t-1}·Δdías/360  −  D_m · (y_t − y_{t-1})        D_m = (m/360) / (1 + y·m/360)
(devengo + efecto del cambio de tasa sobre el precio; duración modificada de un cupón cero). Con eso se
arma un índice de "precio" por plazo que el resto del sistema trata como cualquier activo.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

from stockcast import config

log = logging.getLogger(__name__)
API = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/{ids}/datos/{start}/{end}"
CETES = {"CETES28": ("SF43936", 28), "CETES91": ("SF43939", 91), "CETES182": ("SF43942", 182),
         "CETES364": ("SF43945", 364)}


def token() -> str | None:
    return os.environ.get("BANXICO_TOKEN")


def _fetch(ids: list[str], start: str) -> pd.DataFrame:
    tok = token()
    if not tok:
        raise RuntimeError("Falta BANXICO_TOKEN (token gratuito: banxico.org.mx/SieAPIRest/service/v1/token)")
    url = API.format(ids=",".join(ids), start=start, end=pd.Timestamp.today().date().isoformat())
    r = requests.get(url, headers={"Bmx-Token": tok, "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    cols = {}
    for s in r.json()["bmx"]["series"]:
        d = pd.DataFrame(s.get("datos", []))
        if d.empty:
            continue
        idx = pd.to_datetime(d["fecha"], format="%d/%m/%Y")
        vals = pd.to_numeric(d["dato"].str.replace(",", ""), errors="coerce")
        cols[s["idSerie"]] = pd.Series(vals.to_numpy(), index=idx)
        log.info("Banxico %s: %s (%d datos)", s["idSerie"], s.get("titulo", "")[:60], len(d))
    return pd.DataFrame(cols).sort_index()


def holding_index(yields_pct: pd.Series, maturity_days: int) -> pd.Series:
    """Índice de rendimiento total de mantener un CETE de `maturity_days` (a partir de su tasa anual en %)."""
    y = yields_pct.dropna() / 100
    days = y.index.to_series().diff().dt.days.fillna(0).to_numpy()
    dur = (maturity_days / 360) / (1 + y.shift(1) * maturity_days / 360)
    ret = (y.shift(1) * days / 360 - dur * y.diff()).fillna(0).to_numpy()
    return pd.Series(100 * np.cumprod(1 + ret), index=y.index)


def cetes_prices(refresh: bool = True) -> pd.DataFrame:
    """Precios sintéticos (formato largo de market.load_prices) para cada plazo de CETES, en días hábiles."""
    cache = config.path("data") / "banxico_cetes.parquet"
    if refresh or not cache.exists():
        raw = _fetch([sid for sid, _ in CETES.values()], config.load().get("history_start", "2000-01-01"))
        raw.to_parquet(cache)
    raw = pd.read_parquet(cache)
    bdays = pd.bdate_range(raw.index.min(), pd.Timestamp.today().normalize())
    frames = []
    for name, (sid, m) in CETES.items():
        if sid not in raw:
            continue
        y = raw[sid].reindex(bdays).ffill()          # subasta semanal → tasa vigente cada día hábil
        idx = holding_index(y, m)
        frames.append(pd.DataFrame({"date": idx.index, "ticker": name, "open": idx.values, "high": idx.values,
                                    "low": idx.values, "close": idx.values, "volume": np.nan}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_yields() -> pd.Series:
    """Tasa vigente de cada plazo (para mostrar la curva actual en el dashboard)."""
    cache = config.path("data") / "banxico_cetes.parquet"
    if not cache.exists():
        return pd.Series(dtype=float)
    raw = pd.read_parquet(cache).ffill().iloc[-1]
    return pd.Series({name: raw.get(sid) for name, (sid, _) in CETES.items()})
