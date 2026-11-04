"""Filtro de "¿vale la pena predecir?" a partir de `stockcast compare`.

Niveles de evidencia por plazo (criterios fijados antes de ver resultados):
  fuerte   = cumple 1–3 (abajo) y Deflated Sharpe >= min_dsr
  moderada = cumple 1–3 pero no llega al Deflated Sharpe exigido
  ninguna  = no cumple 1–3 → sin señal
Condiciones, fuera de muestra:
  1. ordena acciones mejor que el azar      (p-valor del IC < max_p_ic e IC >= min_ic)
  2. la estrategia long-short gana           (Sharpe >= min_sharpe_ls)
  3. y esa ganancia no es suerte             (p-valor del Sharpe < max_p_sharpe; más estricto
                                             para mercados candidatos)
Con `stockcast compare` reciente, la evaluación es la de la *selección anidada* (el modelo de cada bloque se
elige sólo con bloques anteriores) y además se exige un Deflated Sharpe >= min_dsr, que corrige por el
número total de configuraciones probadas (Bailey & López de Prado, 2014). El modelo de producción es el que
la regla anidada elige usando todos los bloques. Si ningún horizonte pasa, el mercado candidato no se ejecuta y el
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
    full = pd.DataFrame(json.loads(path.read_text()))
    nested_rows = full[full["engine"] == "nested"].set_index("horizon") if "engine" in full else pd.DataFrame()
    for h, grp in df.groupby("horizon"):
        if h in nested_rows.index:
            # Evaluación honesta: el desempeño es el del PROCEDIMIENTO de selección anidada, y además se exige
            # Deflated Sharpe >= min_dsr (corrige por todas las configuraciones probadas en el mercado).
            nr = nested_rows.loc[h]
            dsr = nr.get("dsr")
            beats_random = bool(nr["p_value_ic"] < g["max_p_ic"] and nr["ic"] >= g.get("min_ic", 0)
                                and nr["sharpe_long_short"] >= g["min_sharpe_ls"]
                                and nr["p_value_sharpe"] < max_p_sharpe)
            strong = beats_random and dsr is not None and dsr == dsr and dsr >= g.get("min_dsr", 0)
            evidence = "fuerte" if strong else "moderada" if beats_random else "ninguna"
            horizons[int(h)] = {"engine": nr["final_engine"], "validated": strong, "evidence": evidence,
                                "method": "nested",
                                "ic": float(nr["ic"]), "p_value_ic": float(nr["p_value_ic"]),
                                "sharpe_long_short": float(nr["sharpe_long_short"]),
                                "p_value_sharpe": float(nr["p_value_sharpe"]),
                                "dsr": None if dsr is None or dsr != dsr else float(dsr),
                                "n_trials": int(nr.get("n_trials", 0) or 0)}
            continue
        passing = ok[ok["horizon"] == h]  # formato anterior (sin selección anidada)
        best = (passing if not passing.empty else grp).sort_values("sharpe_long_short", ascending=False).iloc[0]
        horizons[int(h)] = {"engine": best["engine"], "validated": not passing.empty, "method": "best",
                            "evidence": "moderada" if not passing.empty else "ninguna",
                            "ic": float(best["ic"]), "p_value_ic": float(best["p_value_ic"]),
                            "sharpe_long_short": float(best["sharpe_long_short"]),
                            "p_value_sharpe": float(best["p_value_sharpe"])}
    # un mercado candidato se activa con al menos un plazo con evidencia (moderada o fuerte)
    any_ok = any(v.get("evidence", "ninguna") != "ninguna" for v in horizons.values())
    return {"evaluated": True, "active": market_key in g["core_markets"] or any_ok, "horizons": horizons}


def active_markets() -> list[str]:
    """Mercados a entrenar a diario: activos según el gate y no descartados en el registro de experimentos."""
    from stockcast import experiments
    return [mk for mk in config.markets()
            if decide(mk)["active"] and experiments.get(mk)["status"] != "descartado"]
