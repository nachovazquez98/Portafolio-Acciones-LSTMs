"""Carga de configuración (config/settings.yaml) con rutas resueltas respecto a la raíz del repo."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("STOCKCAST_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("STOCKCAST_CONFIG", ROOT / "config" / "settings.yaml"))


@dataclass(frozen=True)
class Market:
    key: str
    name: str
    currency: str
    benchmark: str
    tickers: list[str]
    macro: list[str]
    news_lang: dict[str, str]
    universe: list[str] = field(default_factory=list)  # índices a expandir (ver data/universe.py)
    min_dollar_volume: float = 0.0                       # filtro de liquidez
    news: bool = True                                    # descargar noticias en cada corrida
    top_k: int | None = None                             # acciones por lado en el backtest (None = global)
    sectors: dict[str, str] = field(default_factory=dict)  # ticker -> sector (para agrupar en el dashboard)
    # Universo punto en el tiempo: en cada fecha sólo entran las `pit_top_n` acciones con mayor volumen
    # operado (mediana de los `pit_window` días ANTERIORES). Evita elegir la lista a mano o por resultados.
    pit_top_n: int | None = None
    pit_window: int = 126
    pit_min_dollar_volume: float = 0.0   # mínimo de liquidez punto en el tiempo (moneda local)
    cost_bps: float | None = None        # costo por lado propio del mercado (None = backtest.cost_bps)
    fred_extra: list[str] = field(default_factory=list)  # series FRED adicionales sólo para este mercado
    long_only: bool = False              # recomendar sólo compras (p. ej. CETES: no se venden en corto)
    requires: str | None = None          # fuente que necesita credencial (p. ej. "banxico")


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(name: str) -> Path:
    p = ROOT / load()["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p


def market(key: str) -> Market:
    m = dict(load()["markets"][key])
    if m.get("universe"):
        from stockcast.data.universe import resolve, sp500_sectors
        m["tickers"] = resolve(m["universe"], m.get("tickers", []))
        if not m.get("sectors"):
            sec = sp500_sectors()
            m["sectors"] = {t: sec.get(t, "Otro") for t in m["tickers"]}
    return Market(key=key, **m)


def markets() -> list[str]:
    return list(load()["markets"])
