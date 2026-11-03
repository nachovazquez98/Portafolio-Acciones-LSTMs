"""Comparación de modelos fuera de muestra + prueba contra el azar.

  stockcast compare --market us

Para cada motor y horizonte: métricas walk-forward, backtest (long y long-short)
y prueba de permutación (¿es mejor que un modelo aleatorio?). Resultado en
artifacts/<mercado>/comparison.json y en la pestaña "Validación" del dashboard.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import time

import pandas as pd

from stockcast import config, evaluation, pipeline
from stockcast.features import feature_columns

log = logging.getLogger(__name__)


def run(market_key: str, refresh: bool = False, engines: list[str] | None = None) -> pd.DataFrame:
    cfg = config.load()
    cc = cfg["compare"]
    engines = engines or cc["engines"]
    if importlib.util.find_spec("torch") is None:
        log.warning("torch no instalado: se omiten los LSTM (instala con `pip install -e .[deep]`)")
        engines = [e for e in engines if not e.startswith("lstm")]
    q = cfg["quantiles"]
    bt_cfg = cfg["backtest"]
    inputs = pipeline.load_inputs(market_key, refresh=refresh, with_news=False)
    panel, mkt = inputs["panel"], inputs["market"]
    feats = feature_columns(panel)
    rows = []
    for h in cc["horizons"]:
        for eng in engines:
            t0 = time.time()
            # el LSTM original trabaja con precios en nivel, así que recibe también `close`
            f_eng = feats + ["close"] if eng == "lstm_legacy" else feats
            oos = evaluation.oos_predictions(panel, f_eng, h, eng, q, pipeline._model_params(eng),
                                             cfg["model"]["cv_splits"])
            secs = time.time() - t0
            m = evaluation.metrics(oos, q)
            bt = evaluation.summarize_backtest(
                evaluation.backtest(oos, h, (mkt.top_k or bt_cfg["top_k"]), bt_cfg["cost_bps"], None, bt_cfg["borrow_bps_annual"]), h)
            rnd = (evaluation.random_test(oos, h, (mkt.top_k or bt_cfg["top_k"]), bt_cfg["cost_bps"], cc["permutations"])
                   if eng != "naive" else {})  # naive pronostica 0 para todos: no ordena
            rows.append({
                "market": market_key, "horizon": h, "engine": eng, "seconds": round(secs, 1),
                "mae": m["mae"], "skill_vs_naive": m["skill_vs_naive"], "hit_rate": m["hit_rate"], "hit_always_up": m["hit_always_up"],
                "p_value_hit": rnd.get("p_value_hit"), "ic": m["ic"], "ic_random_p95": rnd.get("ic_random_p95"),
                "p_value_ic": rnd.get("p_value_ic"), "coverage": m["coverage"],
                "brier": m["brier"], "brier_base": m["brier_base"],
                "n_confident_up": m["n_confident_up"], "up_rate_when_confident_up": m["up_rate_when_confident_up"],
                "n_confident_down": m["n_confident_down"],
                "up_rate_when_confident_down": m["up_rate_when_confident_down"],
                "sharpe_long": bt.get("estrategia", {}).get("sharpe"),
                "sharpe_long_short": bt.get("long_short", {}).get("sharpe"),
                "sharpe_ls_random_p95": rnd.get("sharpe_random_p95"),
                "p_value_sharpe": rnd.get("p_value_sharpe"),
                "sharpe_equal_weight": bt.get("equiponderado", {}).get("sharpe"),
            })
            r = rows[-1]
            log.info("[%s] h=%d %-9s %5.0fs  hit=%.3f ic=%.3f (p=%s) sharpeLS=%s", market_key, h, eng, secs,
                     r["hit_rate"], r["ic"], r["p_value_ic"], r["sharpe_long_short"])
    df = pd.DataFrame(rows)
    out = pipeline.artifacts_dir(market_key) / "comparison.json"
    if out.exists():  # conserva motores/horizontes de corridas anteriores no repetidos ahora
        prev = pd.DataFrame(json.loads(out.read_text()))
        keep = ~prev.set_index(["horizon", "engine"]).index.isin(df.set_index(["horizon", "engine"]).index)
        df = pd.concat([prev[keep], df], ignore_index=True)
    order = {e: i for i, e in enumerate(cc["engines"])}
    df = df.sort_values(["horizon", "engine"], key=lambda s: s.map(order) if s.name == "engine" else s)
    out.write_text(json.dumps(df.to_dict("records"), indent=2, default=str), encoding="utf-8")
    return df.reset_index(drop=True)


def report(path=None) -> str:
    """Tabla Markdown con la comparación de todos los mercados (docs/RESULTS.md)."""
    def f(v, fmt):
        return "–" if v is None or (isinstance(v, float) and v != v) else format(v, fmt)

    lines = ["# Resultados de validación fuera de muestra", "",
             "Generado con `stockcast compare` + `stockcast report`. Walk-forward de 4 bloques sobre ~40% final "
             "de la historia, con purga del horizonte. p-valores contra 300 permutaciones aleatorias de los "
             "pronósticos entre acciones (p < 0.05 → mejor que el azar).", ""]
    for mk in config.markets():
        f_json = pipeline.artifacts_dir(mk) / "comparison.json"
        if not f_json.exists():
            continue
        df = pd.DataFrame(json.loads(f_json.read_text()))
        lines += [f"## {config.load()['markets'][mk]['name']}", "",
                  "| Horizonte | Modelo | IC | p IC | Sharpe long-short | p Sharpe | Acierto dirección "
                  "| 'Siempre sube' | Cobertura p10–p90 | Brier (base) | Seg. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            brier = f"{f(r.get('brier'), '.4f')} ({f(r.get('brier_base'), '.4f')})" if "brier" in r else "–"
            lines.append(f"| {r['horizon']} d | {r['engine']} | {f(r['ic'], '.3f')} | {f(r['p_value_ic'], '.3f')} | "
                         f"{f(r['sharpe_long_short'], '.2f')} | {f(r['p_value_sharpe'], '.3f')} | "
                         f"{f(r['hit_rate'], '.1%')} | {f(r['hit_always_up'], '.1%')} | {f(r['coverage'], '.1%')} | "
                         f"{brier} | {f(r['seconds'], '.0f')} |")
        lines.append("")
    text = "\n".join(lines)
    out = path or config.ROOT / "docs" / "RESULTS.md"
    out.write_text(text, encoding="utf-8")
    return text
