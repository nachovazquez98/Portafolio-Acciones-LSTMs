"""Tasas de política monetaria de bancos centrales.

Fuente principal: BIS (Bank for International Settlements), dataset WS_CBPOL
— tasas oficiales de ~40 bancos centrales, gratis, sin API key, actualizado
semanalmente. https://data.bis.org/topics/CBPOL
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from stockcast import config

log = logging.getLogger(__name__)

BIS_URL = "https://stats.bis.org/api/v1/data/WS_CBPOL/{freq}..?format=csv&detail=dataonly&startPeriod={start}"

# Nombre legible de cada área (código ISO / BIS).
BANKS = {
    "US": "Reserva Federal (EE. UU.)", "XM": "Banco Central Europeo", "GB": "Banco de Inglaterra",
    "JP": "Banco de Japón", "CN": "Banco Popular de China", "CA": "Banco de Canadá",
    "CH": "Banco Nacional Suizo", "AU": "Banco de la Reserva de Australia",
    "NZ": "Banco de la Reserva de Nueva Zelanda", "MX": "Banco de México",
    "BR": "Banco Central do Brasil", "CL": "Banco Central de Chile", "CO": "Banco de la República (Colombia)",
    "PE": "Banco Central de Reserva del Perú", "AR": "Banco Central de la República Argentina",
    "IN": "Banco de la Reserva de India", "KR": "Banco de Corea", "ID": "Banco de Indonesia",
    "TH": "Banco de Tailandia", "MY": "Banco Negara Malasia", "PH": "Bangko Sentral ng Pilipinas",
    "HK": "Autoridad Monetaria de Hong Kong", "SA": "Banco Central de Arabia Saudita",
    "TR": "Banco Central de Turquía", "ZA": "Banco de la Reserva de Sudáfrica", "RU": "Banco de Rusia",
    "SE": "Riksbank (Suecia)", "NO": "Norges Bank", "DK": "Danmarks Nationalbank",
    "PL": "Banco Nacional de Polonia", "CZ": "Banco Nacional Checo", "HU": "Banco Nacional de Hungría",
    "RO": "Banco Nacional de Rumanía", "IS": "Banco Central de Islandia", "IL": "Banco de Israel",
    "KW": "Banco Central de Kuwait", "MK": "Banco Nacional de Macedonia del Norte", "RS": "Banco Nacional de Serbia",
    "HR": "Banco Nacional de Croacia", "BG": "Banco Nacional de Bulgaria",
}


def _fetch(freq: str, start: str) -> pd.DataFrame:
    r = requests.get(BIS_URL.format(freq=freq, start=start), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df.rename(columns={"REF_AREA": "area", "TIME_PERIOD": "date", "OBS_VALUE": "rate"})[["area", "date", "rate"]]


def load_policy_rates(refresh: bool = True, start: str = "1999-01-01") -> pd.DataFrame:
    """Matriz diaria fecha x área con la tasa de política vigente (forward-filled).

    Combina la serie diaria del BIS con la mensual para cubrir bancos que sólo
    publican mensualmente (p. ej. China o India).
    """
    cache = config.path("data") / "central_banks.parquet"
    if refresh or not cache.exists():
        try:
            daily = _fetch("D", start)
            monthly = _fetch("M", start)
            daily["date"] = pd.to_datetime(daily["date"])
            monthly["date"] = pd.to_datetime(monthly["date"]) + pd.offsets.MonthEnd(0)
            monthly = monthly[~monthly["area"].isin(daily["area"].unique())]
            long = pd.concat([daily, monthly], ignore_index=True).dropna()
            long.to_parquet(cache, index=False)
        except requests.RequestException as e:
            if not cache.exists():
                raise
            log.warning("No se pudo actualizar BIS (%s); usando caché", e)
    long = pd.read_parquet(cache)

    areas = config.load()["central_banks"]["areas"]
    if areas != "all":
        long = long[long["area"].isin(areas)]
    wide = long.pivot_table(index="date", columns="area", values="rate").sort_index()
    idx = pd.date_range(wide.index.min(), pd.Timestamp.today().normalize(), freq="D")
    return wide.reindex(idx).ffill().rename_axis("date")


def summary(rates: pd.DataFrame) -> pd.DataFrame:
    """Tabla para el dashboard: tasa actual, cambio en 3/12 meses y fecha del último movimiento."""
    rows = []
    for area in rates.columns:
        s = rates[area].dropna()
        if s.empty:
            continue
        changed = s[s.diff().fillna(0) != 0]
        rows.append({
            "Área": area,
            "Banco central": BANKS.get(area, area),
            "Tasa actual %": s.iloc[-1],
            "Cambio 3m (pb)": (s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=91))) * 100,
            "Cambio 12m (pb)": (s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=365))) * 100,
            "Último movimiento": changed.index[-1].date() if not changed.empty else None,
        })
    return pd.DataFrame(rows).sort_values("Área").reset_index(drop=True)
