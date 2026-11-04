"""Ratios fundamentales y señal de ventaja competitiva ("moat").

Basado en Jiménez-Preciado, Venegas-Martínez & Ramírez-García (2022),
"Stock portfolio optimization with competitive advantages (moat): A machine
learning approach", Mathematics 10(23):4449. Ahí los ratios más importantes para
detectar empresas con moat fueron deuda de largo plazo / utilidad neta y
D&A / utilidad bruta (ambos: más bajo = mejor). Se muestran como filtro de
calidad en el dashboard; no entran al modelo diario porque yfinance sólo da los
últimos 4 años de estados financieros (no hay versión point-in-time gratuita).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _get(df: pd.DataFrame, *names: str) -> float:
    for n in names:
        if n in df.index:
            v = df.loc[n].dropna()
            if not v.empty:
                return float(v.iloc[0])
    return np.nan


def ratios(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    row = {"Ticker": ticker}
    try:
        inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
        net_income = _get(inc, "Net Income")
        gross = _get(inc, "Gross Profit")
        lt_debt = _get(bal, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
        da = _get(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion")
        revenue = _get(inc, "Total Revenue")
        equity = _get(bal, "Stockholders Equity", "Common Stock Equity")
        row.update({
            "Deuda LP / Utilidad neta": lt_debt / net_income if net_income and net_income > 0 else np.nan,
            "D&A / Utilidad bruta": da / gross if gross and gross > 0 else np.nan,
            "Margen bruto %": 100 * gross / revenue if revenue else np.nan,
            "ROE %": 100 * net_income / equity if equity and equity > 0 else np.nan,
        })
        info = t.info or {}
        row["P/E"] = info.get("trailingPE")
        row["P/B"] = info.get("priceToBook")
    except Exception as e:
        log.warning("Fundamentales no disponibles para %s: %s", ticker, e)
    return row


def moat_table(tickers: list[str]) -> pd.DataFrame:
    """Ratios + bandera de moat (heurística: deuda LP < 4x utilidad y D&A < 15% de utilidad bruta)."""
    df = pd.DataFrame([ratios(t) for t in tickers]).set_index("Ticker")
    for col in ("Deuda LP / Utilidad neta", "D&A / Utilidad bruta"):
        if col not in df:
            df[col] = np.nan
    df["Moat probable"] = (df["Deuda LP / Utilidad neta"] < 4) & (df["D&A / Utilidad bruta"] < 0.15)
    return df
