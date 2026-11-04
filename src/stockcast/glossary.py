"""Diccionario de variables: nombre legible y explicación de cada columna del modelo.

Se usa en las gráficas del dashboard y para generar docs/VARIABLES.md (`stockcast report`).
"""
from __future__ import annotations

import re

import pandas as pd

from stockcast import config
from stockcast.data.central_banks import BANKS

TECH = {
    "ret_1": ("Retorno de 1 día", "Cambio porcentual (logarítmico) del precio respecto al cierre anterior."),
    "vol_ratio": ("Volatilidad reciente vs. trimestral",
                  "Volatilidad de 5 días dividida entre la de 63 días. > 1: la acción está más agitada de lo normal."),
    "bb_pos": ("Posición en Bandas de Bollinger",
               "Dónde está el precio dentro de las bandas de 20 días: -1 = banda inferior, 0 = media, +1 = superior."),
    "sma50_ratio": ("Distancia a la media de 50 días",
                    "Precio / media móvil de 50 días − 1. Positivo: cotiza por encima de su tendencia de ~2 meses."),
    "sma200_ratio": ("Distancia a la media de 200 días",
                     "Precio / media móvil de 200 días − 1. Tendencia de largo plazo (~10 meses)."),
    "rsi_14": ("RSI 14 días", "Índice de fuerza relativa (0–100). > 70 sobrecompra, < 30 sobreventa."),
    "macd_hist": ("Histograma MACD", "MACD (EMA 12 − EMA 26) menos su señal de 9 días, dividido entre el precio. "
                                     "Mide si el impulso de corto plazo se acelera o se frena."),
    "stoch_k": ("Estocástico %K", "Dónde cerró el precio dentro del rango máximo-mínimo de 14 días (0–100)."),
    "atr_pct": ("Rango verdadero promedio (%)", "ATR de 14 días dividido entre el precio: tamaño típico del movimiento diario."),
    "dist_52w_high": ("Distancia al máximo de 52 semanas", "Precio / máximo del último año − 1. 0 = en máximos."),
    "volume_z": ("Volumen anormal", "Volumen de hoy en desviaciones estándar respecto a su promedio de 20 días."),
    "news_sent": ("Sentimiento de noticias", "Promedio del sentimiento (-1 a +1) de las noticias del día."),
    "news_count": ("Número de noticias", "Cantidad de noticias publicadas ese día sobre la acción."),
}

MACRO_NAMES = {
    "GSPC": "S&P 500", "VIX": "VIX (volatilidad esperada del S&P 500)", "TNX": "Tasa del Tesoro de EE. UU. a 10 años",
    "IRX": "Tasa del Tesoro a 3 meses", "FVX": "Tasa del Tesoro a 5 años", "DXYNYB": "Índice del dólar (DXY)",
    "CLF": "Petróleo WTI", "GCF": "Oro", "MXX": "IPC (Bolsa Mexicana)", "MXNX": "Tipo de cambio USD/MXN",
    "BVSP": "Bovespa (Brasil)", "EWW": "ETF de México (EWW)", "GDAXI": "DAX (Alemania)", "STOXX50E": "Euro Stoxx 50",
    "EURUSDX": "Tipo de cambio EUR/USD", "N225": "Nikkei 225 (Japón)", "JPYX": "Tipo de cambio USD/JPY",
    "HSI": "Hang Seng (Hong Kong)", "FTSE": "FTSE 100 (Reino Unido)", "GBPUSDX": "Tipo de cambio GBP/USD",
    "URTH": "MSCI World (ETF)", "NDX": "Nasdaq-100", "MOVE": "MOVE (volatilidad esperada de bonos del Tesoro)",
    "TIP": "Bonos ligados a inflación (ETF TIP)", "EEM": "Mercados emergentes (ETF EEM)",
}

FRED_LEVEL_NOTE = {
    "USEPUINDXD": "Promedio de 30 días; más alto = más incertidumbre de política económica",
    "BAA10Y": "Más alto = más estrés crediticio (inversionistas exigen más premio por riesgo)",
}


def describe(col: str) -> dict:
    """{'variable', 'nombre', 'descripcion', 'grupo'} para una columna del panel."""
    from stockcast.ablation import GROUPS, group_of

    group = GROUPS.get(group_of(col), group_of(col)).split(" (")[0]
    name, desc = col, ""

    if col in TECH:
        name, desc = TECH[col]
    elif m := re.fullmatch(r"ret_(\d+)", col):
        n = int(m[1])
        name = f"Momentum {n} días"
        desc = f"Retorno logarítmico de la acción en los últimos {n} días hábiles (~{round(n / 21, 1)} meses)."
    elif m := re.fullmatch(r"vol_(\d+)", col):
        name = f"Volatilidad {m[1]} días"
        desc = f"Desviación estándar anualizada de los retornos diarios de los últimos {m[1]} días."
    elif m := re.fullmatch(r"rel_ret_(\d+)", col):
        name = f"Momentum relativo {m[1]} días"
        desc = (f"Retorno de {m[1]} días de la acción menos el promedio de todas las acciones del mercado ese día. "
                "Positivo: le va mejor que al resto.")
    elif m := re.fullmatch(r"m_([A-Z0-9]+?)(_ret5|_ret21|_chg21)?", col):
        base = MACRO_NAMES.get(m[1], m[1])
        suffix = m[2] or ""
        if suffix == "_ret5":
            name, desc = f"{base}: retorno 1 semana", f"Retorno logarítmico de {base} en los últimos 7 días naturales."
        elif suffix == "_ret21":
            name, desc = f"{base}: retorno 1 mes", f"Retorno logarítmico de {base} en los últimos 30 días naturales."
        elif suffix == "_chg21":
            name, desc = f"{base}: cambio 1 mes", f"Cambio en puntos de {base} en los últimos 30 días naturales."
        else:
            name, desc = f"{base}: nivel", f"Valor actual de {base}."
    elif m := re.fullmatch(r"cb_([A-Z]{2})_chg(\d+)", col):
        bank = BANKS.get(m[1], m[1])
        name = f"{bank}: cambio de tasa {m[2]} d"
        desc = (f"Cambio de la tasa de política de {bank} en {m[2]} días, en puntos base. "
                "Negativo = recortes (política más laxa), positivo = alzas.")
    elif col == "cb_breadth":
        name = "Ciclo monetario global"
        desc = ("(Bancos centrales que recortaron − los que subieron) / total, en 90 días. "
                "+1 = todos recortando, -1 = todos subiendo.")
    elif m := re.fullmatch(r"cb_spread_([A-Z]{2})_US", col):
        bank = BANKS.get(m[1], m[1])
        name = f"Diferencial de tasas: {bank} − Fed"
        desc = f"Tasa de {bank} menos la de la Fed (puntos porcentuales). Afecta tipo de cambio y flujos de capital."
    elif m := re.fullmatch(r"f_(slope_10y3m|curvature)(_chg30)?", col):
        base = ("Pendiente de la curva (10 años − 3 meses)" if m[1] == "slope_10y3m"
                else "Curvatura de la curva (2×5a − 1a − 30a)")
        name = base + (": cambio 30 d" if m[2] else "")
        desc = ("Diferencia entre tasas largas y cortas: anticipa la prima por plazo (Fama y Bliss, 1987)."
                if m[1] == "slope_10y3m" else
                "Qué tan 'jorobada' está la curva; resume expectativas de política a mediano plazo "
                "(Litterman y Scheinkman, 1991).") + (" Cambio en 30 días." if m[2] else "")
    elif m := re.fullmatch(r"f_([A-Z0-9]+?)(_chg30)?", col):
        cfgd = config.load()
        label = cfgd.get("fred", {}).get(m[1]) or cfgd.get("fred_labels", {}).get(m[1], m[1])
        if m[2]:
            name, desc = f"{label}: cambio 30 d", f"Cambio en 30 días de «{label}» (FRED, serie {m[1]})."
        else:
            note = FRED_LEVEL_NOTE.get(m[1], "")
            name, desc = f"{label}: nivel", f"Valor actual de «{label}» (FRED, serie {m[1]}). {note}".strip()
    return {"variable": col, "nombre": name, "descripcion": desc, "grupo": group}


def label(col: str) -> str:
    return describe(col)["nombre"]


def table(cols) -> pd.DataFrame:
    return pd.DataFrame([describe(c) for c in cols])


def write_markdown(cols, path=None) -> str:
    df = table(sorted(set(cols))).sort_values(["grupo", "variable"])
    lines = ["# Diccionario de variables", "",
             "Variables que puede usar el modelo (cada mercado descarta los grupos sin aporte; ver la "
             "pestaña *Validación* o `stockcast ablation`). Todas se calculan sólo con información disponible "
             "al cierre del día.", ""]
    for g, sub in df.groupby("grupo", sort=False):
        lines += [f"## {g}", "", "| Variable | Nombre | Qué mide |", "|---|---|---|"]
        lines += [f"| `{r.variable}` | {r.nombre} | {r.descripcion} |" for r in sub.itertuples()]
        lines.append("")
    text = "\n".join(lines)
    (path or config.ROOT / "docs" / "VARIABLES.md").write_text(text, encoding="utf-8")
    return text
