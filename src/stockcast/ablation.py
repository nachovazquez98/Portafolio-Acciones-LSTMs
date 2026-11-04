"""¿Cuánto aporta cada grupo de variables a la predicción? (estudio de ablación)

  stockcast ablation --market us

Para cada horizonte se reentrena el modelo elegido por el gate quitando un grupo de
variables a la vez y se mide, fuera de muestra, cuánto cambian el IC y el Sharpe
long-short. Si al quitar un grupo la predicción NO empeora, ese grupo no aporta y
sólo agrega ruido y cómputo. Resultado en artifacts/<mercado>/ablation.json y en la
pestaña "Validación" del dashboard.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from stockcast import config, evaluation, gate, pipeline
from stockcast.features import feature_columns

log = logging.getLogger(__name__)

GROUPS = {
    "tecnicos": "Técnicos de la acción (momentum, volatilidad, RSI, MACD, volumen…)",
    "macro": "Mercado y macro (índices, VIX, tasa 10a, dólar, petróleo…)",
    "bancos_centrales": "Tasas de bancos centrales (BIS)",
    "fred": "Estrés e incertidumbre (FRED: EPU, high yield, curva, tasa real)",
    "noticias": "Sentimiento de noticias",
}


def dropped_groups(market_key: str) -> list[str]:
    """Grupos que la última ablación marcó como irrelevantes para este mercado."""
    if not config.load().get("features", {}).get("auto_drop", True):
        return []
    p = pipeline.artifacts_dir(market_key) / "ablation.json"
    return json.loads(p.read_text()).get("drop_groups", []) if p.exists() else []


def group_of(col: str) -> str:
    if col.startswith("cb_"):
        return "bancos_centrales"
    if col.startswith("m_"):
        return "macro"
    if col.startswith("f_"):
        return "fred"
    if col.startswith("news_"):
        return "noticias"
    return "tecnicos"


def run(market_key: str, horizons: list[int] | None = None) -> pd.DataFrame:
    cfg = config.load()
    q = cfg["quantiles"]
    inputs = pipeline.load_inputs(market_key, refresh=False, with_news=True, drop_irrelevant=False)
    panel, mkt = inputs["panel"], inputs["market"]
    feats = feature_columns(panel)
    present = sorted({group_of(c) for c in feats})
    decision = gate.decide(market_key)
    horizons = horizons or cfg["horizons"]
    top_k = mkt.top_k or cfg["backtest"]["top_k"]
    rows = []
    for h in horizons:
        eng = decision["horizons"].get(h, {}).get("engine", cfg["model"]["engine"])
        variants = {"todas": feats}
        variants.update({f"sin_{g}": [c for c in feats if group_of(c) != g] for g in present if g != "tecnicos"})
        variants["solo_tecnicos"] = [c for c in feats if group_of(c) == "tecnicos"]
        for name, cols in variants.items():
            oos = evaluation.oos_predictions(panel, cols, h, eng, q, pipeline._model_params(eng),
                                             cfg["model"]["cv_splits"])
            m = evaluation.metrics(oos, q)
            bt = evaluation.summarize_backtest(
                evaluation.backtest(oos, h, top_k, cfg["backtest"]["cost_bps"], None,
                                    cfg["backtest"]["borrow_bps_annual"]), h)
            rows.append({"market": market_key, "horizon": h, "engine": eng, "variant": name,
                         "n_features": len(cols), "ic": m["ic"],
                         "sharpe_long_short": bt.get("long_short", {}).get("sharpe", np.nan)})
            log.info("[%s] h=%d %-22s n=%3d ic=%.3f sharpeLS=%.2f", market_key, h, name, len(cols),
                     m["ic"], rows[-1]["sharpe_long_short"])
    df = pd.DataFrame(rows)
    base = df[df["variant"] == "todas"].set_index("horizon")[["ic", "sharpe_long_short"]]
    df["delta_ic"] = df["ic"] - df["horizon"].map(base["ic"])
    df["delta_sharpe"] = df["sharpe_long_short"] - df["horizon"].map(base["sharpe_long_short"])
    contrib = contribution(df)
    # un grupo se descarta si quitarlo no empeora ni el IC ni el Sharpe (promedio entre horizontes)
    drop = sorted(g for g, r in contrib.iterrows() if r["aporte_ic"] <= 0 and r["aporte_sharpe"] <= 0)
    out = {"rows": df.to_dict("records"), "groups_present": present,
           "contribution": contrib.reset_index().to_dict("records"), "drop_groups": drop}
    (pipeline.artifacts_dir(market_key) / "ablation.json").write_text(json.dumps(out, indent=2, default=str))
    return df


def contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aporte de cada grupo = cuánto empeora la predicción al quitarlo (promedio entre horizontes).
    Positivo = el grupo ayuda; ≤ 0 = no aporta (quitarlo da igual o mejora)."""
    sub = df[df["variant"].str.startswith("sin_")].copy()
    sub["grupo"] = sub["variant"].str.removeprefix("sin_")
    return (sub.groupby("grupo")[["delta_ic", "delta_sharpe"]].mean() * -1).rename(
        columns={"delta_ic": "aporte_ic", "delta_sharpe": "aporte_sharpe"})
