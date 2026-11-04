"""Universo de acciones multi-bolsa a partir de los componentes de índices principales.

Yahoo Finance no publica una lista de "todas" sus acciones (son >100k símbolos,
la mayoría ilíquidos e imposibles de vender en corto). Lo práctico y gratuito es
usar los componentes de los índices líderes de cada bolsa: son las acciones
líquidas que cualquier broker permite comprar y, casi siempre, vender en corto.

Las listas se leen de Wikipedia y se guardan en data/universe/<índice>.json
(se refrescan cada 7 días). Agrega o quita índices en config/settings.yaml.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time

import pandas as pd
import requests

from stockcast import config

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (stockcast; investigación local)"}
MAX_AGE_DAYS = 7


def _hk(code: str) -> str:
    n = re.sub(r"\D", "", code)
    return f"{int(n):04d}.HK" if n else ""


def _after_colon(code: str) -> str:
    return code.split(":")[-1].strip()


def _digits(width: int):
    def fn(code: str) -> str:
        n = re.sub(r"\D", "", str(code).split(".")[0])
        return n.zfill(width) if n else ""
    return fn


# índice -> (url, columna, sufijo de Yahoo, transformación[, todas_las_tablas])
SOURCES = {
    # América
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "", lambda s: s.replace(".", "-")),
    "tsx60": ("https://en.wikipedia.org/wiki/S%26P/TSX_60", "Symbol", ".TO", lambda s: s.replace(".", "-")),
    "ibovespa": ("https://pt.wikipedia.org/wiki/Lista_de_companhias_citadas_no_Ibovespa", "Código", ".SA", None),
    # Europa
    "ftse100": ("https://en.wikipedia.org/wiki/FTSE_100_Index", "Ticker", ".L", lambda s: s.replace(".", "")),
    "dax": ("https://en.wikipedia.org/wiki/DAX", "Ticker", "", None),
    "cac40": ("https://en.wikipedia.org/wiki/CAC_40", "Ticker", "", None),
    "eurostoxx50": ("https://en.wikipedia.org/wiki/EURO_STOXX_50", "Ticker", "", None),
    "smi": ("https://en.wikipedia.org/wiki/Swiss_Market_Index", "Ticker", ".SW", None),
    "ibex35": ("https://en.wikipedia.org/wiki/IBEX_35", "Ticker", "", None),
    "aex": ("https://en.wikipedia.org/wiki/AEX_index", "Ticker", "", None),
    "ftsemib": ("https://en.wikipedia.org/wiki/FTSE_MIB", "Ticker", "", None),
    "omx30": ("https://en.wikipedia.org/wiki/OMX_Stockholm_30", "Ticker", "", None),
    # Asia-Pacífico
    "nikkei225": ("https://ja.wikipedia.org/wiki/日経平均株価", "証券コード", ".T", _digits(4), True),
    "hangseng": ("https://en.wikipedia.org/wiki/Hang_Seng_Index", "Ticker", "", _hk),
    "kospi200": ("https://en.wikipedia.org/wiki/KOSPI_200", "Symbol", ".KS", _digits(6)),
    "nifty50": ("https://en.wikipedia.org/wiki/NIFTY_50", "Symbol", ".NS", None),
    "sti": ("https://en.wikipedia.org/wiki/Straits_Times_Index", "Stock symbol", ".SI", _after_colon),
    "asx50": ("https://en.wikipedia.org/wiki/S%26P/ASX_50", "Symbol", ".AX", None),
}

# Sin tabla confiable en Wikipedia: lista fija de las más grandes del Nikkei 225.
STATIC = {
    "nikkei_top": ["7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "9432.T", "8035.T", "6098.T", "4063.T",
                   "9983.T", "6501.T", "7974.T", "8058.T", "8001.T", "4502.T", "6367.T", "7267.T", "6902.T",
                   "8316.T", "6954.T", "4568.T", "6273.T", "7741.T", "8766.T", "6981.T"],
}


def _scrape(name: str) -> list[str]:
    url, col, suffix, fn, *rest = SOURCES[name]
    all_tables = bool(rest and rest[0])  # p. ej. Nikkei: una tabla por sector
    html = requests.get(url, headers=UA, timeout=30).text
    values: list[str] = []
    for t in pd.read_html(io.StringIO(html)):
        if col in t.columns and (all_tables or len(t) >= 15):
            values += t[col].dropna().astype(str).str.strip().str.replace("\xa0", " ").tolist()
            if not all_tables:
                break
    out = []
    for v in values:
        v = fn(v) if fn else v
        if v and not v.endswith(suffix):
            v = v + suffix
        if v and " " not in v:
            out.append(v)
    if not out:
        raise ValueError(f"No se encontró la tabla de {name}")
    return sorted(set(out))


NASDAQ_API = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
GICS_ES = {"Information Technology": "Tecnología", "Communication Services": "Comunicaciones",
           "Consumer Discretionary": "Consumo discrecional", "Consumer Staples": "Consumo básico",
           "Financials": "Financiero", "Health Care": "Salud", "Industrials": "Industrial", "Energy": "Energía",
           "Materials": "Materiales", "Utilities": "Servicios públicos", "Real Estate": "Inmobiliario"}


def _nasdaq100() -> list[str]:
    """Componentes del Nasdaq-100 desde la API pública de Nasdaq (Wikipedia ya no publica la tabla)."""
    h = {**UA, "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36", "Accept": "application/json"}
    rows = requests.get(NASDAQ_API, headers=h, timeout=30).json()["data"]["data"]["rows"]
    return sorted({r["symbol"].replace(".", "-") for r in rows})


def _cached(name: str, fetch) -> list | dict:
    d = config.path("data") / "universe"
    d.mkdir(exist_ok=True)
    cache = d / f"{name}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return json.loads(cache.read_text())
    try:
        val = fetch()
        cache.write_text(json.dumps(val))
        return val
    except Exception as e:
        if cache.exists():
            log.warning("No se pudo actualizar %s (%s); uso caché", name, e)
            return json.loads(cache.read_text())
        log.warning("No se pudo obtener %s: %s", name, e)
        return [] if not name.endswith("sectors") else {}


def sp500_sectors() -> dict[str, str]:
    """Ticker -> sector GICS (en español) para el S&P 500."""
    def fetch():
        html = requests.get(SOURCES["sp500"][0], headers=UA, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        return {str(s).replace(".", "-"): GICS_ES.get(g, g) for s, g in zip(t["Symbol"], t["GICS Sector"])}
    return _cached("sp500_sectors", fetch)


def index_members(name: str) -> list[str]:
    if name in STATIC:
        return STATIC[name]
    if name == "nasdaq100":
        return _cached("nasdaq100", _nasdaq100)
    if name not in SOURCES:
        raise ValueError(f"Índice desconocido: {name}. Opciones: {sorted(SOURCES) + sorted(STATIC)}")
    d = config.path("data") / "universe"
    d.mkdir(exist_ok=True)
    cache = d / f"{name}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < MAX_AGE_DAYS * 86400:
        return json.loads(cache.read_text())
    try:
        members = _scrape(name)
        cache.write_text(json.dumps(members))
        return members
    except Exception as e:
        if cache.exists():
            log.warning("No se pudo actualizar %s (%s); uso caché", name, e)
            return json.loads(cache.read_text())
        log.warning("No se pudo obtener %s: %s", name, e)
        return []


def resolve(indices: list[str], extra: list[str] | None = None) -> list[str]:
    tickers = set(extra or [])
    for name in indices:
        tickers.update(index_members(name))
    return sorted(tickers)


def liquid(prices: pd.DataFrame, min_dollar_volume: float, window: int = 63) -> list[str]:
    """Tickers con volumen operado mediano (precio x volumen, moneda local) >= umbral."""
    recent = prices[prices["date"] >= prices["date"].max() - pd.Timedelta(days=window * 1.5)]
    dv = (recent["close"] * recent["volume"]).groupby(recent["ticker"]).median()
    return sorted(dv[dv >= min_dollar_volume].index)


# sufijo de Yahoo -> (bolsa, país)
EXCHANGES = {
    "": ("NYSE/NASDAQ", "EE. UU."), "TO": ("Toronto", "Canadá"), "SA": ("B3 São Paulo", "Brasil"),
    "MX": ("BMV", "México"), "L": ("Londres", "Reino Unido"), "DE": ("Xetra Frankfurt", "Alemania"),
    "F": ("Frankfurt", "Alemania"), "PA": ("Euronext París", "Francia"), "AS": ("Euronext Ámsterdam", "Países Bajos"),
    "BR": ("Euronext Bruselas", "Bélgica"), "MI": ("Borsa Italiana", "Italia"), "MC": ("BME Madrid", "España"),
    "SW": ("SIX Zúrich", "Suiza"), "ST": ("Nasdaq Estocolmo", "Suecia"), "HE": ("Nasdaq Helsinki", "Finlandia"),
    "CO": ("Nasdaq Copenhague", "Dinamarca"), "IR": ("Euronext Dublín", "Irlanda"), "LS": ("Euronext Lisboa", "Portugal"),
    "VI": ("Viena", "Austria"), "T": ("Tokio", "Japón"), "HK": ("Hong Kong", "Hong Kong"),
    "KS": ("Korea Exchange", "Corea del Sur"), "NS": ("NSE India", "India"), "SI": ("SGX Singapur", "Singapur"),
    "AX": ("ASX Sídney", "Australia"),
}


def exchange_of(ticker: str) -> tuple[str, str]:
    if ticker.endswith("-USD"):
        return ("Cripto", "Cripto")
    suffix = ticker.rsplit(".", 1)[1] if "." in ticker else ""
    return EXCHANGES.get(suffix, (suffix, suffix))
