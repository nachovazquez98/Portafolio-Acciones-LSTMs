"""Pipeline completo: datos -> features -> validación -> modelo final -> pronóstico -> artefactos.

Artefactos en artifacts/<mercado>/:
  forecasts.parquet         último pronóstico por ticker y horizonte (con bandas p10-p90)
  forecast_history.parquet  todos los pronósticos emitidos (para medir precisión real después)
  metrics.json              métricas fuera de muestra vs. caminata aleatoria
  backtest.parquet          curva de capital de la estrategia top-k
  importance.parquet        importancia de features (LightGBM)
  model_h<h>.joblib         modelos finales para inferencia (stockcast predict)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from stockcast import config, evaluation, signals
from stockcast.data import central_banks, fred, market, news, universe
from stockcast.features import build_panel, feature_columns
from stockcast.models import make_model

log = logging.getLogger(__name__)
MIN_NEWS_DAYS = 250  # el sentimiento sólo entra al modelo cuando hay ~1 año de histórico propio


def artifacts_dir(market_key: str):
    d = config.path("artifacts") / market_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def point_in_time_universe(stocks: pd.DataFrame, top_n: int | None, window: int,
                           min_dollar_volume: float = 0.0) -> pd.MultiIndex:
    """(fecha, ticker) que pertenecen al universo: mediana de volumen operado (precio x volumen) de los
    `window` días previos >= min_dollar_volume y, si se pide, dentro del top_n. El shift(1) garantiza que la
    decisión del día t sólo use datos hasta t-1."""
    dv = (stocks.assign(dv=stocks["close"] * stocks["volume"])
          .pivot_table(index="date", columns="ticker", values="dv").sort_index())
    liq = dv.rolling(window, min_periods=window // 2).median().shift(1)
    member = liq >= min_dollar_volume
    if top_n:
        member &= liq.rank(axis=1, ascending=False) <= top_n
    member = member.stack()
    return member[member].index.set_names(["date", "ticker"])


def load_inputs(market_key: str, refresh: bool = True, with_news: bool = True,
                drop_irrelevant: bool = True) -> dict:
    cfg = config.load()
    m = config.market(market_key)
    prices = market.load_prices(sorted(set(m.tickers + m.macro + [m.benchmark])), refresh=refresh)
    stocks = prices[prices["ticker"].isin(m.tickers)]
    pit_members = None
    if m.pit_top_n or m.pit_min_dollar_volume > 0:
        pit_members = point_in_time_universe(stocks, m.pit_top_n, m.pit_window, m.pit_min_dollar_volume)
        log.info("[%s] universo punto en el tiempo (top %s, mínimo %.0f diario): %d candidatas, %d distintas en la "
                 "historia", market_key, m.pit_top_n or "todas", m.pit_min_dollar_volume, stocks["ticker"].nunique(),
                 pit_members.get_level_values("ticker").nunique())
    if m.min_dollar_volume > 0:
        keep = universe.liquid(stocks, m.min_dollar_volume)
        log.info("[%s] %d de %d tickers pasan el filtro de liquidez", market_key, len(keep), stocks["ticker"].nunique())
        stocks = stocks[stocks["ticker"].isin(keep)]
    macro_close = market.close_matrix(prices[prices["ticker"].isin(m.macro + [m.benchmark])])

    try:
        cb = central_banks.load_policy_rates(refresh=refresh)
    except Exception as e:
        log.warning("Sin tasas de bancos centrales: %s", e)
        cb = None
    try:
        fr = fred.load_fred(refresh=refresh)
    except Exception as e:
        log.warning("Sin series FRED: %s", e)
        fr = None

    news_daily = None
    if with_news and m.news:
        try:
            hist = news.update_news_history(m.tickers, market_key) if refresh else \
                pd.read_parquet(config.path("data") / "news.parquet")
            hist = hist[hist["ticker"].isin(m.tickers)]
            # días distintos con noticias DENTRO del último año y medio: Google devuelve notas viejas sueltas
            # repartidas en años que no equivalen a una serie diaria utilizable
            recent = hist[hist["published"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=548)]
            days = recent["published"].dt.normalize().nunique() if not recent.empty else 0
            if days >= MIN_NEWS_DAYS:
                news_daily = news.daily_sentiment(hist)
            else:
                log.info("[%s] noticias en %d días distintos (<%d): aún no entran al modelo",
                         market_key, days, MIN_NEWS_DAYS)
        except Exception as e:
            log.warning("Noticias no disponibles: %s", e)

    panel = build_panel(stocks, macro_close, cb, cfg["horizons"], news_daily, fr,
                        cfg["central_banks"]["breadth_window_days"])
    panel = panel[panel["ret_126"].notna()]
    if pit_members is not None:
        panel = panel[panel.index.isin(pit_members)]
    if drop_irrelevant:
        from stockcast.ablation import dropped_groups, group_of
        drop = dropped_groups(market_key)
        if drop:
            cols = [c for c in panel.columns if not c.startswith("y_") and c != "close" and group_of(c) in drop]
            panel = panel.drop(columns=cols)
            log.info("[%s] sin aporte en la ablación, se descartan: %s (%d variables)",
                     market_key, ", ".join(drop), len(cols))
    return {"market": m, "panel": panel, "macro_close": macro_close}


def _calibrate(pred: pd.DataFrame, quantiles: list[float], k: float) -> pd.DataFrame:
    """Aplica el factor conformal k a las bandas alrededor de la mediana."""
    out = pred.copy()
    for q in quantiles:
        if q != 0.5:
            out[f"q{q}"] = pred["q0.5"] + k * (pred[f"q{q}"] - pred["q0.5"])
    return out


def _model_params(engine: str) -> dict:
    m = config.load()["model"]
    if engine == "ensemble":
        return {"lightgbm": dict(m.get("lightgbm", {})), "ridge": dict(m.get("ridge", {}))}
    return dict(m.get(engine, {}))


def run(market_key: str, refresh: bool = True, engine: str | None = None,
        evaluate: bool = True, with_news: bool = True) -> dict:
    t0 = time.time()
    cfg = config.load()
    from stockcast import experiments, gate
    decision = gate.decide(market_key)
    engine = engine or cfg["model"]["engine"]
    quantiles, horizons = cfg["quantiles"], cfg["horizons"]
    if decision.get("evaluated") and cfg["gate"].get("train_only_with_evidence", False):
        # sólo se entrenan los plazos con evidencia; el resto no generaría señales de todos modos
        horizons = [h for h in horizons
                    if decision["horizons"].get(h, {}).get("evidence", "ninguna") != "ninguna"]
        if not horizons:
            out_dir = artifacts_dir(market_key)
            (out_dir / "forecasts.parquet").unlink(missing_ok=True)   # evita mostrar pronósticos viejos
            info = {"market": market_key, "status": "pausa", "gate": decision,
                    "run_at": datetime.now().isoformat(timespec="seconds"),
                    "metrics": json.loads((out_dir / "metrics.json").read_text()).get("metrics", {})
                    if (out_dir / "metrics.json").exists() else {}}
            (out_dir / "metrics.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            if experiments.get(market_key)["status"] in ("activo", "pendiente"):   # conserva el motivo de la evaluación
                experiments.record(market_key, "pausa", "ningún plazo con evidencia: no se entrena", "entrenamiento")
            log.info("[%s] en pausa: ningún plazo con evidencia, no se entrena", market_key)
            return info
        log.info("[%s] plazos con evidencia a entrenar: %s", market_key, horizons)
    inputs = load_inputs(market_key, refresh=refresh, with_news=with_news)
    m, panel = inputs["market"], inputs["panel"]
    feats = feature_columns(panel)
    out_dir = artifacts_dir(market_key)
    log.info("[%s] panel %s, %d features, motor=%s", market_key, panel.shape, len(feats), engine)

    all_metrics, forecasts, importances = {}, [], []
    prev_k = {}  # sin validación, se reutiliza la calibración conformal de la última corrida completa
    if (out_dir / "metrics.json").exists():
        prev = json.loads((out_dir / "metrics.json").read_text()).get("metrics", {})
        prev_k = {int(h): v.get("conformal_k", 1.0) for h, v in prev.items() if h.isdigit()}
    bt_h = cfg["backtest"]["horizon"]
    bt = pd.DataFrame()
    for h in horizons:
        target = f"y_{h}"
        g_h = decision["horizons"].get(h, {})
        eng_h = g_h.get("engine", engine) if engine == cfg["model"]["engine"] else engine
        params = _model_params(eng_h)
        if evaluate:
            n_splits = cfg["model"]["cv_splits"]
            oos = evaluation.oos_predictions(panel, feats, h, eng_h, quantiles, params, n_splits)
            naive = evaluation.oos_predictions(panel, feats, h, "naive", quantiles, {}, n_splits)
            all_metrics[h] = {"engine": eng_h, eng_h: evaluation.metrics(oos, quantiles),
                              "naive": evaluation.metrics(naive, quantiles),
                              "conformal_k": evaluation.conformal_scale(oos, quantiles)}
            log.info("[%s] h=%d skill=%.3f hit=%.3f ic=%.3f", market_key, h,
                     all_metrics[h][eng_h]["skill_vs_naive"], all_metrics[h][eng_h]["hit_rate"],
                     all_metrics[h][eng_h]["ic"])
            if h == bt_h:
                bench = inputs["macro_close"].get(m.benchmark)
                bt = evaluation.backtest(oos, h, (m.top_k or cfg["backtest"]["top_k"]),
                                         m.cost_bps if m.cost_bps is not None else cfg["backtest"]["cost_bps"], bench,
                                         cfg["backtest"]["borrow_bps_annual"])
                all_metrics["backtest"] = evaluation.summarize_backtest(bt, h)

        train = panel.dropna(subset=[target])
        model = make_model(eng_h, quantiles, params).fit(train[feats], train[target], h)
        joblib.dump({"model": model, "features": feats, "horizon": h, "engine": eng_h,
                     "conformal_k": all_metrics.get(h, {}).get("conformal_k", prev_k.get(h, 1.0))},
                    out_dir / f"model_h{h}.joblib")
        if hasattr(model, "feature_importance"):
            importances.append(model.feature_importance().rename(f"h{h}"))

        # Pronóstico desde la última fecha disponible de cada ticker
        dates = panel.index.get_level_values("date")
        ctx = panel[dates > dates.max() - pd.Timedelta(days=200)]
        pred = model.predict(ctx[feats])
        last = pred.groupby(level="ticker").tail(1)
        k = all_metrics.get(h, {}).get("conformal_k", prev_k.get(h, 1.0))
        last = _calibrate(last, quantiles, k)
        for (d, t), row in last.iterrows():
            close = panel.loc[(d, t), "close"]
            rec = {"ticker": t, "as_of": d, "horizon": h, "close": close, "engine": eng_h,
                   "validated": bool(g_h.get("validated", False)),
                   "evidence": g_h.get("evidence", "ninguna"),
                   "target_date": d + pd.offsets.BDay(h)}
            rec["conformal_k"] = k
            for q in quantiles:
                rec[f"ret_q{q}"] = float(np.expm1(row[f"q{q}"]))
                rec[f"price_q{q}"] = float(close * np.exp(row[f"q{q}"]))
            forecasts.append(rec)

    fc = signals.add_signals(pd.DataFrame(forecasts), quantiles).assign(run_at=datetime.now())
    fc.to_parquet(out_dir / "forecasts.parquet", index=False)
    hist_path = out_dir / "forecast_history.parquet"
    hist = pd.concat([pd.read_parquet(hist_path), fc]) if hist_path.exists() else fc
    hist.drop_duplicates(["ticker", "as_of", "horizon", "engine"], keep="last").to_parquet(hist_path, index=False)
    if importances:
        pd.concat(importances, axis=1).to_parquet(out_dir / "importance.parquet")
    if not bt.empty:
        bt.to_parquet(out_dir / "backtest.parquet")
    info = {"market": market_key, "engine": engine, "gate": decision, "status": "activo", "run_at": datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(panel)), "n_features": len(feats), "tickers": sorted(panel.index.unique("ticker")),
            "seconds": round(time.time() - t0, 1),
            "metrics": {str(k): v for k, v in all_metrics.items()} if evaluate else
            (json.loads((out_dir / "metrics.json").read_text()).get("metrics", {})
             if (out_dir / "metrics.json").exists() else {})}
    (out_dir / "metrics.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log.info("[%s] listo en %.0fs", market_key, info["seconds"])
    return info


def predict(market_key: str, refresh: bool = True) -> pd.DataFrame:
    """Inferencia sin reentrenar: carga model_h*.joblib y pronostica con datos actualizados."""
    out_dir = artifacts_dir(market_key)
    inputs = load_inputs(market_key, refresh=refresh, with_news=False)
    panel = inputs["panel"]
    dates = panel.index.get_level_values("date")
    ctx = panel[dates > dates.max() - pd.Timedelta(days=200)]
    rows = []
    for path in sorted(out_dir.glob("model_h*.joblib")):
        art = joblib.load(path)
        X = ctx.reindex(columns=art["features"])  # columnas nuevas/faltantes -> NaN
        last = art["model"].predict(X).groupby(level="ticker").tail(1)
        last = _calibrate(last, config.load()["quantiles"], art.get("conformal_k", 1.0))
        for (d, t), row in last.iterrows():
            close = panel.loc[(d, t), "close"]
            rows.append({"ticker": t, "as_of": d, "horizon": art["horizon"], "close": close,
                         **{f"price_{c}": close * np.exp(v) for c, v in row.items()},
                         **{f"ret_{c}": np.expm1(v) for c, v in row.items()}})
    return pd.DataFrame(rows)
