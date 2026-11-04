"""Comparación de modelos fuera de muestra + prueba contra el azar.

  stockcast compare --market us

Para cada motor y horizonte: métricas walk-forward, backtest (long y long-short)
y prueba de permutación (¿es mejor que un modelo aleatorio?). Resultado en
artifacts/<mercado>/comparison.json y en la pestaña "Validación" del dashboard.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from stockcast import config, evaluation, pipeline
from stockcast.features import feature_columns

log = logging.getLogger(__name__)


def run(market_key: str, refresh: bool = False, engines: list[str] | None = None) -> pd.DataFrame:
    cfg = config.load()
    cc = cfg["compare"]
    engines = engines or cc["engines"]
    q = cfg["quantiles"]
    bt_cfg = cfg["backtest"]
    inputs = pipeline.load_inputs(market_key, refresh=refresh, with_news=False)
    panel, mkt = inputs["panel"], inputs["market"]
    feats = feature_columns(panel)
    rows, ls_series, ppy = [], {}, {}
    top_k = mkt.top_k or bt_cfg["top_k"]
    cost = mkt.cost_bps if mkt.cost_bps is not None else bt_cfg["cost_bps"]
    # retorno a 5 días de cada acción: base de los portafolios traslapados (Jegadeesh & Titman 1993)
    y_step = panel["y_5"] if bt_cfg.get("overlapping", False) and "y_5" in panel else None

    def evaluate(oos, h, eng, secs, extra=None):
        m = evaluation.metrics(oos, q)
        bt_df = evaluation.backtest(oos, h, top_k, cost, None, bt_cfg["borrow_bps_annual"], y_step=y_step)
        bt = evaluation.summarize_backtest(bt_df, h)
        rnd = (evaluation.random_test(oos, h, top_k, cost, cc["permutations"], y_step=y_step,
                                      borrow_bps_annual=bt_cfg["borrow_bps_annual"])
               if eng != "naive" else {})
        if "long_short" in bt_df:
            ls_series[(h, eng)] = bt_df["long_short"]
            ppy[(h, eng)] = bt_df.attrs.get("periods_per_year", 252 / h)
        row = {
            "market": market_key, "horizon": h, "engine": eng, "seconds": round(secs, 1),
            "mae": m["mae"], "skill_vs_naive": m["skill_vs_naive"], "hit_rate": m["hit_rate"],
            "hit_always_up": m["hit_always_up"],
            "p_value_hit": rnd.get("p_value_hit"), "ic": m["ic"], "ic_random_p95": rnd.get("ic_random_p95"),
            "p_value_ic": rnd.get("p_value_ic"), "coverage": m["coverage"],
            "brier": m["brier"], "brier_base": m["brier_base"],
            "sharpe_long": bt.get("estrategia", {}).get("sharpe"),
            "sharpe_long_short": bt.get("long_short", {}).get("sharpe"),
            "sharpe_ls_random_p95": rnd.get("sharpe_random_p95"),
            "p_value_sharpe": rnd.get("p_value_sharpe"),
            "sharpe_equal_weight": bt.get("equiponderado", {}).get("sharpe"),
            **(extra or {}),
        }
        rows.append(row)
        log.info("[%s] h=%d %-9s %5.0fs  hit=%.3f ic=%.3f (p=%s) sharpeLS=%s", market_key, h, eng, secs,
                 row["hit_rate"], row["ic"], row["p_value_ic"], row["sharpe_long_short"])

    eligible = config.load()["gate"]["engines"]
    for h in cc["horizons"]:
        oos_by_engine = {}
        computed = {}
        for eng in engines:
            t0 = time.time()
            if eng == "ensemble" and {"lightgbm", "ridge"} <= computed.keys():
                # el ensamble es el promedio exacto de LightGBM y Ridge (ambos deterministas): se reutilizan
                # sus predicciones en vez de volver a entrenarlos (~1/3 menos tiempo de cómputo)
                a, b = computed["lightgbm"], computed["ridge"]
                qc = [f"q{x}" for x in q]
                oos = a.copy()
                oos[qc] = (a[qc] + b.loc[a.index, qc]) / 2
            else:
                oos = evaluation.oos_predictions(panel, feats, h, eng, q, pipeline._model_params(eng),
                                                 cfg["model"]["cv_splits"])
            computed[eng] = oos
            evaluate(oos, h, eng, time.time() - t0)
            if eng in eligible:
                oos_by_engine[eng] = oos
        if len(oos_by_engine) >= 1:
            # Selección anidada: elige el modelo de cada bloque sólo con bloques anteriores
            nested, choices, final = evaluation.nested_selection(oos_by_engine)
            if not nested.empty:
                evaluate(nested, h, "nested", 0.0,
                         {"nested_choices": choices, "final_engine": final})

    # Deflated Sharpe: corrige por las configuraciones ENTRE LAS QUE SE ELIGE (motores elegibles × plazos).
    # Los Sharpe de las pruebas se expresan en la frecuencia de la serie evaluada (Bailey & López de Prado).
    trials_ann = [r["sharpe_long_short"] for r in rows
                  if r["engine"] in eligible and r["sharpe_long_short"] is not None and np.isfinite(r["sharpe_long_short"])]
    for r in rows:
        key = (r["horizon"], r["engine"])
        s = ls_series.get(key)
        if s is None or r["engine"] == "naive":
            r["dsr"] = None
        else:
            f = np.sqrt(ppy[key])
            r["dsr"] = evaluation.deflated_sharpe(s, np.asarray(trials_ann) / f)
        r["n_trials"] = len(trials_ann)

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
             "pronósticos entre acciones (p < 0.05 → mejor que el azar). `nested` = selección anidada del modelo "
             "(lo que realmente se usa); Deflated Sharpe = probabilidad de ganancia real tras descontar todas las "
             "configuraciones probadas en el mercado.", ""]
    for mk in config.markets():
        f_json = pipeline.artifacts_dir(mk) / "comparison.json"
        if not f_json.exists():
            continue
        df = pd.DataFrame(json.loads(f_json.read_text()))
        lines += [f"## {config.load()['markets'][mk]['name']}", "",
                  "| Horizonte | Modelo | IC | p IC | Sharpe long-short | p Sharpe | Deflated Sharpe | Acierto dirección "
                  "| 'Siempre sube' | Cobertura p10–p90 | Brier (base) | Seg. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            brier = f"{f(r.get('brier'), '.4f')} ({f(r.get('brier_base'), '.4f')})" if "brier" in r else "–"
            lines.append(f"| {r['horizon']} d | {r['engine']} | {f(r['ic'], '.3f')} | {f(r['p_value_ic'], '.3f')} | "
                         f"{f(r['sharpe_long_short'], '.2f')} | {f(r['p_value_sharpe'], '.3f')} | "
                         f"{f(r.get('dsr'), '.0%')} | "
                         f"{f(r['hit_rate'], '.1%')} | {f(r['hit_always_up'], '.1%')} | {f(r['coverage'], '.1%')} | "
                         f"{brier} | {f(r['seconds'], '.0f')} |")
        lines.append("")
    text = "\n".join(lines)
    out = path or config.ROOT / "docs" / "RESULTS.md"
    out.write_text(text, encoding="utf-8")
    return text


def screen(market_key: str, n_perm: int = 100) -> tuple[bool, list[dict]]:
    """Etapa 1 (prueba rápida, ~segundos): sólo Ridge y `n_perm` permutaciones del IC.

    Condición necesaria para que haya ganancia long-short: que el modelo ordene mejor que el azar en
    algún plazo. Si ni eso pasa, no tiene sentido entrenar LightGBM ni hacer ablación (early stopping, como
    en successive halving / Hyperband). Descartar nunca crea falsos positivos, sólo ahorra cómputo.
    """
    cfg = config.load()
    g = cfg["gate"]
    q = cfg["quantiles"]
    inputs = pipeline.load_inputs(market_key, refresh=False, with_news=False)
    panel, mkt = inputs["panel"], inputs["market"]
    feats = feature_columns(panel)
    top_k = mkt.top_k or cfg["backtest"]["top_k"]
    rows = []
    for h in cfg["compare"]["horizons"]:
        oos = evaluation.oos_predictions(panel, feats, h, "ridge", q, pipeline._model_params("ridge"),
                                         cfg["model"]["cv_splits"])
        rnd = evaluation.random_test(oos, h, top_k, 0, n_perm)
        rows.append({"horizon": h, "ic": rnd["ic"], "p_value_ic": rnd["p_value_ic"]})
        log.info("[%s] prueba rápida h=%d ic=%.3f p=%.3f", market_key, h, rnd["ic"], rnd["p_value_ic"])
    ok = any(r["p_value_ic"] < g["max_p_ic"] and r["ic"] >= g.get("min_ic", 0) for r in rows)
    return ok, rows


def evaluate(market_key: str, force: bool = False) -> dict:
    """Evaluación por etapas de un mercado (lo que corre la tarea semanal).

    1. prueba rápida → si falla, descartado (o en pausa si es núcleo) y se detiene.
    2. compare completo → 3. ablación → 4. compare otra vez si la ablación descartó variables.
    5. el gate fija la evidencia y se actualiza el registro de experimentos.
    """
    from stockcast import ablation, experiments, gate

    core = market_key in config.load()["gate"]["core_markets"]
    if not force and not experiments.due_for_evaluation(market_key):
        e = experiments.get(market_key)
        log.info("[%s] descartado el %s (%s): se re-prueba hasta cumplir el plazo", market_key,
                 e["updated"][:10], e["reason"])
        return e
    ok, rows = screen(market_key)
    if not ok:
        best = max(rows, key=lambda r: r["ic"])
        reason = f"no ordena mejor que el azar en ningún plazo (mejor IC {best['ic']:.3f}, p={best['p_value_ic']:.3f})"
        return experiments.record(market_key, "pausa" if core else "descartado", reason, "prueba rápida")
    run(market_key)
    before = set(ablation.dropped_groups(market_key))
    ablation.run(market_key)
    if set(ablation.dropped_groups(market_key)) != before:
        run(market_key)                      # revalidar con las variables finales
    d = gate.decide(market_key)
    good = {h: v for h, v in d["horizons"].items() if v.get("evidence", "ninguna") != "ninguna"}
    if d["active"] and good:
        txt = ", ".join(f"{h} d ({v['evidence']})" for h, v in sorted(good.items()))
        return experiments.record(market_key, "activo", f"evidencia en {txt}", "evaluación completa",
                                  horizons=sorted(good))
    best = max(d["horizons"].values(), key=lambda v: v["sharpe_long_short"]) if d["horizons"] else None
    reason = ("ordena mejor que el azar pero la estrategia no gana de forma significativa con costos"
              + (f" (mejor Sharpe {best['sharpe_long_short']:.2f})" if best else ""))
    return experiments.record(market_key, "pausa" if core else "descartado", reason, "evaluación completa")
