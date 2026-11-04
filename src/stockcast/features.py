"""Ingeniería de features en formato panel (fecha, ticker).

Diferencias clave respecto al notebook original:
- Se modelan *retornos logarítmicos* (estacionarios), no precios en nivel.
- Todas las features usan sólo información disponible al cierre del día t.
- Los objetivos y_h son el retorno de t a t+h (pronóstico directo multi-horizonte),
  así no hace falta alimentar al modelo con sus propias predicciones ni congelar
  las variables macro durante 30 días.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores técnicos normalizados para un solo ticker (df ordenado por fecha)."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    out = pd.DataFrame(index=df.index)
    for n in (1, 5, 21, 63, 126):
        out[f"ret_{n}"] = logc.diff(n)
    r1 = logc.diff()
    for n in (5, 21, 63):
        out[f"vol_{n}"] = r1.rolling(n).std() * np.sqrt(252)
    out["vol_ratio"] = out["vol_5"] / out["vol_63"]
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pos"] = (c - sma20) / (2 * std20)                      # posición en Bandas de Bollinger
    out["sma50_ratio"] = c / c.rolling(50).mean() - 1
    out["sma200_ratio"] = c / c.rolling(200).mean() - 1
    out["rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / c
    out["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1
    lv = np.log1p(v.replace(0, np.nan))
    out["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    return out


def macro_features(macro_close: pd.DataFrame) -> pd.DataFrame:
    """Retornos/cambios de índices, VIX, tasas y divisas (en calendario diario, forward-filled)."""
    m = macro_close.sort_index()
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="D")).ffill()
    out = pd.DataFrame(index=m.index)
    for col in m.columns:
        name = col.replace("^", "").replace("=", "").replace("-", "").replace(".", "")
        s = m[col]
        if col in ("^TNX", "^VIX", "^IRX", "^FVX"):  # niveles con significado propio
            out[f"m_{name}"] = s
            out[f"m_{name}_chg21"] = s - s.shift(30)
        else:
            ls = np.log(s.where(s > 0))  # el petróleo cotizó negativo en abril de 2020
            out[f"m_{name}_ret5"] = ls.diff(7)
            out[f"m_{name}_ret21"] = ls.diff(30)
    return out.rename_axis("date")


def central_bank_features(rates: pd.DataFrame, breadth_days: int = 90) -> pd.DataFrame:
    """Ciclo de política monetaria de todos los bancos centrales del BIS.

    Sólo se usan transformaciones estacionarias: con niveles de tasa los árboles
    "memorizan" épocas (2021 = tasa 0) y empeoran fuera de muestra.
    - cb_<AREA>_chg90: cambio en 90 días (pb) -> fase de alzas/recortes de cada banco
    - cb_breadth: (# bancos que recortaron - # que subieron) / total en la ventana
    - cb_spread_<MX|XM|GB|JP>_US: diferencial de tasas vs. la Fed (carry)
    """
    out = pd.DataFrame(index=rates.index)
    chg = (rates - rates.shift(breadth_days)) * 100
    for a in rates.columns:
        out[f"cb_{a}_chg90"] = chg[a]
    valid = chg.notna().sum(axis=1).replace(0, np.nan)
    out["cb_breadth"] = ((chg < 0).sum(axis=1) - (chg > 0).sum(axis=1)) / valid
    if "US" in rates:
        out["cb_US_chg30"] = (rates["US"] - rates["US"].shift(30)) * 100
        for a in ("MX", "XM", "GB", "JP"):  # diferencial de tasas vs. la Fed (carry / presión cambiaria)
            if a in rates:
                out[f"cb_spread_{a}_US"] = rates[a] - rates["US"]
    return out.rename_axis("date")


def fred_features(fred: pd.DataFrame) -> pd.DataFrame:
    """Niveles suavizados y cambios de 30 días de las series de estrés/incertidumbre de FRED."""
    out = pd.DataFrame(index=fred.index)
    for c in fred.columns:
        s = fred[c].rolling(30, min_periods=1).mean() if c == "USEPUINDXD" else fred[c]
        if c in ("USEPUINDXD", "BAA10Y"):  # niveles de estrés con rango estable (tasas: sólo cambios)
            out[f"f_{c}"] = s
        out[f"f_{c}_chg30"] = s - s.shift(30)
    return out.rename_axis("date")


def build_panel(prices: pd.DataFrame, macro_close: pd.DataFrame | None = None,
                cb_rates: pd.DataFrame | None = None, horizons: list[int] = (1, 5, 10, 21),
                news_daily: pd.DataFrame | None = None, fred: pd.DataFrame | None = None,
                breadth_days: int = 90) -> pd.DataFrame:
    """Panel con features X y objetivos y_<h> indexado por (date, ticker)."""
    frames = []
    for t, g in prices.sort_values("date").groupby("ticker"):
        g = g.set_index("date")
        f = ticker_features(g)
        logc = np.log(g["close"])
        for h in horizons:
            f[f"y_{h}"] = logc.shift(-h) - logc
        f["close"] = g["close"]
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames).reset_index()

    # Momentum relativo: retorno del ticker vs. promedio del universo ese día
    for n in (21, 63):
        panel[f"rel_ret_{n}"] = panel[f"ret_{n}"] - panel.groupby("date")[f"ret_{n}"].transform("mean")

    calendar_extra = []
    if macro_close is not None and not macro_close.empty:
        calendar_extra.append(macro_features(macro_close))
    if cb_rates is not None and not cb_rates.empty:
        calendar_extra.append(central_bank_features(cb_rates, breadth_days))
    if fred is not None and not fred.empty:
        calendar_extra.append(fred_features(fred))
    for extra in calendar_extra:
        panel = panel.merge(extra, left_on="date", right_index=True, how="left")

    if news_daily is not None and not news_daily.empty:
        panel = panel.merge(news_daily, on=["date", "ticker"], how="left")
        panel[["news_sent", "news_count"]] = panel[["news_sent", "news_count"]].fillna(0)

    # float32: la mitad de memoria (importa con ~1600 acciones) sin pérdida práctica de precisión
    num = panel.select_dtypes("float64").columns.difference(["close"])
    panel[num] = panel[num].astype("float32")
    return panel.set_index(["date", "ticker"]).sort_index()


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if not c.startswith("y_") and c != "close"]
