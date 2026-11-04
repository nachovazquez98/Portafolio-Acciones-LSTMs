"""Señales COMPRA / VENTA EN CORTO / NEUTRAL y órdenes para un broker.

La señal es *relativa* (ranking entre acciones del mismo día), no absoluta: la
validación mostró que el modelo sirve para ordenar acciones (IC > 0) mucho más
que para acertar el precio exacto. Se compra el quintil superior y se vende en
corto el inferior, que es lo que evalúa el backtest long-short.

Órdenes: `stockcast orders` genera un CSV. Con `--submit` las envía a Alpaca
(acciones de EE. UU., API gratuita con cuenta *paper* para probar sin dinero real).
Nada se envía sin ese flag explícito.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests


log = logging.getLogger(__name__)
ALPACA_PAPER = "https://paper-api.alpaca.markets"


def prob_up_vec(qvals: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """P(retorno > 0) interpolando linealmente la CDF entre cuantiles. qvals: (n, len(quantiles))."""
    qs = np.asarray(quantiles, dtype=float)
    x = np.sort(np.asarray(qvals, dtype=float), axis=1)
    cdf0 = np.full(len(x), np.nan)
    cdf0[0 <= x[:, 0]] = qs[0] / 2                      # 0 debajo del cuantil más bajo
    cdf0[0 > x[:, -1]] = 1 - (1 - qs[-1]) / 2            # 0 arriba del más alto
    for i in range(len(qs) - 1):
        lo, hi = x[:, i], x[:, i + 1]
        m = np.isnan(cdf0) & (lo <= 0) & (0 < hi)
        cdf0[m] = qs[i] + (qs[i + 1] - qs[i]) * (0 - lo[m]) / (hi[m] - lo[m])
    return 1 - np.nan_to_num(cdf0, nan=0.5)


def add_signals(fc: pd.DataFrame, quantiles: list[float], quantile_cut: float = 0.2,
                long_only: bool = False) -> pd.DataFrame:
    fc = fc.copy()
    fc["prob_up"] = prob_up_vec(fc[[f"ret_q{q}" for q in quantiles]].to_numpy(), quantiles)
    pct = fc.groupby("horizon")["ret_q0.5"].rank(pct=True)
    fc["rank_pct"] = pct
    fc["signal"] = np.select([pct > 1 - quantile_cut, pct <= quantile_cut], ["COMPRA", "VENTA EN CORTO"], "NEUTRAL")
    if long_only:  # p. ej. CETES: sólo se recomienda en qué invertir, no qué vender en corto
        fc.loc[fc["signal"] == "VENTA EN CORTO", "signal"] = "NEUTRAL"
    if "evidence" in fc:  # plazo sin ventaja demostrada fuera de muestra -> no se recomienda operar
        fc.loc[fc["evidence"] == "ninguna", "signal"] = "SIN SEÑAL"
    elif "validated" in fc:
        fc.loc[~fc["validated"].astype(bool), "signal"] = "SIN SEÑAL"
    from stockcast.data.universe import exchange_of
    ex = fc["ticker"].map(exchange_of)
    fc["exchange"], fc["country"] = ex.str[0], ex.str[1]
    return fc


def build_orders(market_key: str, capital: float, horizon: int, top_k: int, allow_short: bool = True) -> pd.DataFrame:
    from stockcast import pipeline  # evita import circular
    fc = pd.read_parquet(pipeline.artifacts_dir(market_key) / "forecasts.parquet")
    f = fc[fc["horizon"] == horizon].sort_values("ret_q0.5")
    ev = f["evidence"].iloc[0] if "evidence" in f and len(f) else ("fuerte" if f.get("validated", pd.Series([True])).any() else "ninguna")
    if ev == "ninguna":
        raise SystemExit(f"El horizonte {horizon} d de '{market_key}' no tiene evidencia (gate): "
                         "no se generan órdenes. Revisa `stockcast compare`.")
    if ev == "moderada":
        log.warning("Evidencia MODERADA en %s a %d d: ordena mejor que el azar, pero la ganancia no sobrevive "
                    "la corrección por pruebas múltiples. Opera con montos pequeños o en paper.", market_key, horizon)
    legs = [("buy", f.tail(top_k))]
    if allow_short:
        legs.append(("sell_short", f.head(top_k)))
    notional = capital / (len(legs) * top_k)
    rows = []
    for side, df in legs:
        for _, r in df.iterrows():
            rows.append({"ticker": r["ticker"], "side": side, "notional": round(notional, 2),
                         # cripto admite fracciones; acciones, enteros
                         "qty": round(notional / r["close"], 6) if r["ticker"].endswith("-USD")
                         else int(notional // r["close"]), "last_close": r["close"],
                         "expected_ret": r["ret_q0.5"], "p10": r["ret_q0.1"], "p90": r["ret_q0.9"],
                         "hold_until": r["target_date"]})
    return pd.DataFrame(rows)


def _alpaca(method: str, path: str, **kw):
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Define ALPACA_API_KEY y ALPACA_SECRET_KEY (cuenta paper gratuita en alpaca.markets)")
    base = os.environ.get("ALPACA_BASE_URL", ALPACA_PAPER)
    r = requests.request(method, base + path, timeout=30,
                         headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, **kw)
    r.raise_for_status()
    return r.json()


def submit_alpaca(orders: pd.DataFrame) -> pd.DataFrame:
    """Envía órdenes a Alpaca (paper por defecto). Verifica que el activo sea operable/vendible en corto."""
    results = []
    for _, o in orders.iterrows():
        sym = o["ticker"]
        status = ""
        try:
            asset = _alpaca("GET", f"/v2/assets/{sym}")
            if not asset.get("tradable"):
                status = "no operable en Alpaca"
            elif o["side"] == "sell_short" and not (asset.get("shortable") and asset.get("easy_to_borrow")):
                status = "no disponible para venta en corto"
            elif o["qty"] <= 0:
                status = "cantidad 0 (capital insuficiente)"
            else:
                side = "buy" if o["side"] == "buy" else "sell"
                res = _alpaca("POST", "/v2/orders", json={"symbol": sym, "qty": str(o["qty"]), "side": side,
                                                          "type": "market", "time_in_force": "day"})
                status = f"enviada ({res.get('id', '')[:8]})"
        except requests.HTTPError as e:
            status = f"error: {e.response.status_code} {e.response.text[:80]}"
        results.append(status)
        log.info("%s %s x%s -> %s", o["side"], sym, o["qty"], status)
    return orders.assign(status=results)


def orders_cli(market_key: str, capital: float, horizon: int, top_k: int, no_short: bool, submit: bool) -> pd.DataFrame:
    from stockcast import pipeline
    orders = build_orders(market_key, capital, horizon, top_k, allow_short=not no_short)
    out = pipeline.artifacts_dir(market_key) / f"orders_h{horizon}.csv"
    orders.to_csv(out, index=False)
    log.info("Órdenes guardadas en %s", out)
    if submit:
        us_only = orders[~orders["ticker"].str.contains(r"\.|-USD|\^", regex=True)]
        if len(us_only) < len(orders):
            log.warning("Alpaca sólo opera acciones de EE. UU.; se omiten %d órdenes", len(orders) - len(us_only))
        orders = submit_alpaca(us_only)
    return orders
