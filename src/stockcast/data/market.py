"""Descarga de precios OHLCV con yfinance y caché local en parquet.

La caché evita volver a descargar 8 años de historia cada día: sólo se piden las
últimas semanas y se combinan con lo guardado.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from stockcast import config

log = logging.getLogger(__name__)
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _download(tickers: list[str], start: str, batch: int = 100) -> pd.DataFrame:
    """Descarga en lotes para no saturar a Yahoo con universos grandes."""
    if len(tickers) > batch:
        parts = []
        for i in range(0, len(tickers), batch):
            parts.append(_download(tickers[i:i + batch], start, batch))
            time.sleep(2)
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    raw = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    frames = []
    for t in ([] if raw.empty else tickers):
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t][FIELDS].dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.assign(ticker=t)
        frames.append(df)
    got = {f["ticker"].iloc[0] for f in frames}
    missing = [t for t in tickers if t not in got]
    if missing and pd.Timestamp(start) < pd.Timestamp.today() - pd.Timedelta(days=800):
        # yfinance falla con algunas zonas horarias (p. ej. horario de verano de Brasil 2018);
        # reintentar desde una fecha posterior recupera esos tickers con algo menos de historia.
        later = (pd.Timestamp(start) + pd.Timedelta(days=400)).date().isoformat()
        retry = _download(missing, later, batch)
        if not retry.empty:
            return pd.concat([_to_long(frames), retry], ignore_index=True) if frames else retry
    if not frames:
        return pd.DataFrame()
    return _to_long(frames)


def _to_long(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames).rename(columns=str.lower).rename_axis("date").reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out


def load_prices(tickers: list[str], refresh: bool = True, years: int | None = None) -> pd.DataFrame:
    """Precios en formato largo: date, ticker, open, high, low, close, volume.

    Desde `history_start` (config). Si un ticker en caché se descargó con un inicio posterior al pedido
    (p. ej. caché de 8 años y ahora se piden 26), se vuelve a bajar su historia completa una sola vez.
    """
    cfg = config.load()
    start = cfg.get("history_start") or (date.today() - timedelta(days=365 * (years or cfg["history_years"]))).isoformat()
    cache = config.path("data") / "prices.parquet"
    meta_path = config.path("data") / "prices_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cached = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["date", "ticker"])
    have = set(cached["ticker"].unique()) if not cached.empty else set()

    if refresh:
        # pendiente de historia completa: no está, o se pidió con un inicio más tardío que el actual
        missing = [t for t in tickers if t not in have or meta.get(t, "9999") > start]
        known = [t for t in tickers if t not in missing]
        parts = [cached[~cached["ticker"].isin(missing)]] if not cached.empty else []
        if missing:
            log.info("Descargando historia completa (desde %s) de %d tickers", start, len(missing))
            parts.append(_download(missing, start))
            meta.update({t: start for t in missing})
        if known:
            last = cached.loc[cached["ticker"].isin(known), "date"].max()
            inc_start = (pd.Timestamp(last) - pd.Timedelta(days=10)).date().isoformat()
            log.info("Actualización incremental desde %s", inc_start)
            parts.append(_download(known, inc_start))
        merged = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        merged = merged.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
        merged.to_parquet(cache, index=False)
        meta_path.write_text(json.dumps(meta))
        cached = merged

    out = cached[cached["ticker"].isin(tickers) & (cached["date"] >= pd.Timestamp(start))].copy()
    lost = sorted(set(tickers) - set(out["ticker"]))
    if lost:
        log.warning("Sin datos para: %s", ", ".join(lost))
    return out.reset_index(drop=True)


def close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Matriz fecha x ticker de precios de cierre."""
    return prices.pivot(index="date", columns="ticker", values="close").sort_index()
