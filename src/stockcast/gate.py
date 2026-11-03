"""Filtro de "¿vale la pena predecir?" a partir de `stockcast compare`.

Un horizonte de un mercado queda *validado* sólo si, fuera de muestra:
  1. ordena acciones mejor que el azar      (p-valor del IC < max_p_ic e IC >= min_ic)
  2. la estrategia long-short gana           (Sharpe >= min_sharpe_ls)
  3. y esa ganancia no es suerte             (p-valor del Sharpe < max_p_sharpe; más estricto
                                             para mercados candidatos)
Para cada horizonte se elige el modelo barato con mayor Sharpe long-short entre los
que pasan. Si ningún horizonte pasa, el mercado candidato no se ejecuta y el
dashboard no emite señales para él.
"""
from __future__ import annotations

import json

import pandas as pd

from stockcast import config, pipeline


def decide(market_key: str) -> dict:
    g = config.load()["gate"]
    path = pipeline.artifacts_dir(market_key) / "comparison.json"
    if not path.exists():
        return {"evaluated": False, "active": market_key in g["core_markets"], "horizons": {}}
    df = pd.DataFrame(json.loads(path.read_text()))
    df = df[df["engine"].isin(g["engines"])]
    max_p_sharpe = g["max_p_sharpe"] if market_key in g["core_markets"] else g.get("candidate_max_p_sharpe",
                                                                                     g["max_p_sharpe"])
    ok = df[(df["p_value_ic"] < g["max_p_ic"]) & (df["ic"] >= g.get("min_ic", 0))
            & (df["sharpe_long_short"] >= g["min_sharpe_ls"])
            & (df["p_value_sharpe"] < max_p_sharpe)]
    horizons = {}
    for h, grp in df.groupby("horizon"):
        passing = ok[ok["horizon"] == h]
        best = (passing if not passing.empty else grp).sort_values("sharpe_long_short", ascending=False).iloc[0]
        horizons[int(h)] = {"engine": best["engine"], "validated": not passing.empty,
                            "ic": float(best["ic"]), "p_value_ic": float(best["p_value_ic"]),
                            "sharpe_long_short": float(best["sharpe_long_short"]),
                            "p_value_sharpe": float(best["p_value_sharpe"])}
    any_ok = any(v["validated"] for v in horizons.values())
    return {"evaluated": True, "active": market_key in g["core_markets"] or any_ok, "horizons": horizons}


def active_markets() -> list[str]:
    return [mk for mk in config.markets() if decide(mk)["active"]]
