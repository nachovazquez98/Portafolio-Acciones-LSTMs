"""Dashboard local de stockcast.  Ejecutar:  stockcast app   (o  streamlit run app/streamlit_app.py)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain as X  # noqa: E402
from stockcast import config, experiments, glossary, pipeline, tracking  # noqa: E402
from stockcast.ablation import GROUPS, group_of  # noqa: E402
from stockcast.data import central_banks, fred, fundamentals, market, news  # noqa: E402
from stockcast.features import _rsi  # noqa: E402

st.set_page_config(page_title="stockcast", page_icon="📈", layout="wide")
CFG = config.load()
HORIZON_LABEL = {1: "1 día", 5: "1 semana", 10: "2 semanas", 21: "1 mes"}
GREEN, RED, BLUE, GRAY = "#1e9e5a", "#d0453a", "#2a6fdb", "#9aa5b1"


def hl(h) -> str:
    return HORIZON_LABEL.get(int(h), f"{h} d")


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f}%"


def how_to(text: str, title: str = "¿Cómo leer esta gráfica?"):
    with st.expander(f"💡 {title}"):
        st.markdown(text)


def compact(fig, h: int):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


# ---------------- carga con caché ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_artifacts(mk: str, _stamp: float):
    d = pipeline.artifacts_dir(mk)
    out = {"info": json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None}
    ab = d / "ablation.json"
    out["ablation"] = json.loads(ab.read_text()) if ab.exists() else None
    cmp = d / "comparison.json"
    out["comparison"] = pd.DataFrame(json.loads(cmp.read_text())) if cmp.exists() else pd.DataFrame()
    for name in ("forecasts", "backtest", "importance"):
        p = d / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...]):
    return market.load_prices(list(tickers), refresh=False)


@st.cache_data(ttl=6 * 3600, show_spinner="Descargando tasas de bancos centrales (BIS)…")
def load_rates():
    return central_banks.load_policy_rates(refresh=not (config.path("data") / "central_banks.parquet").exists())


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_fred():
    return fred.load_fred(refresh=not (config.path("data") / "fred.parquet").exists())


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(ticker: str, mk: str):
    return news.fetch_news(ticker, mk)


@st.cache_data(ttl=3600, show_spinner=False)
def load_tracking(mk: str, _stamp: float):
    return tracking.summary(mk)


@st.cache_data(ttl=24 * 3600, show_spinner="Descargando estados financieros…")
def load_moat(ticker: str):
    return fundamentals.moat_table([ticker])


# ---------------- barra lateral ----------------
st.sidebar.title("📈 stockcast")
st.sidebar.caption("Ranking de acciones con machine learning, bancos centrales y macro. 100% local.")
REG = experiments.load()
avail = [k for k in config.markets()
         if REG.get(k, {}).get("status") in ("activo", "pausa")
         or (k not in REG and (pipeline.artifacts_dir(k) / "forecasts.parquet").exists())] or config.markets()
mk = st.sidebar.radio("Mercado", avail, format_func=lambda k: CFG["markets"][k]["name"] + (
    "  ⏸️" if REG.get(k, {}).get("status") == "pausa" else ""))
M = config.market(mk)
adir = pipeline.artifacts_dir(mk)
stamp = (adir / "metrics.json").stat().st_mtime if (adir / "metrics.json").exists() else 0
A = load_artifacts(mk, stamp)
INFO = A["info"] or {}
GATE = {int(h): v for h, v in INFO.get("gate", {}).get("horizons", {}).items()}
EV_BADGE = {"fuerte": "✅ evidencia fuerte", "moderada": "🟡 evidencia moderada", "ninguna": "sin señal"}


def evidence(h) -> str:
    return GATE.get(int(h), {}).get("evidence", "fuerte" if GATE.get(int(h), {}).get("validated") else "ninguna")
DROPPED = set((A["ablation"] or {}).get("drop_groups", []))
SECTOR = M.sectors

st.sidebar.divider()
if INFO:
    st.sidebar.markdown(f"**Última actualización**  \n{INFO['run_at'].replace('T', ' ')}")
if st.sidebar.button("🔄 Actualizar datos y pronósticos", width="stretch",
                     help="Descarga precios nuevos y reentrena (1–5 min). La validación completa se hace con "
                          "`stockcast compare` / la corrida de los lunes."):
    with st.spinner("Actualizando… (1–5 min)"):
        pipeline.run(mk, refresh=True, evaluate=False)
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("¿Primera vez? Empieza por la pestaña **📘 Cómo funciona**.")
st.sidebar.caption("⚠️ Uso educativo. No es recomendación de inversión.")

def experiments_table():
    st.subheader("🧪 Estado de los experimentos")
    st.dataframe(experiments.summary_table(), hide_index=True, width="stretch")
    st.caption("Evaluación por etapas cada lunes: prueba rápida (Ridge, segundos) → evaluación completa sólo si "
               "pasa → el filtro decide la evidencia. Los descartados no se recalculan; se re-prueban cada "
               f"{CFG['gate'].get('recheck_days', 180)} días. Sólo se entrenan a diario los plazos con evidencia.")


if INFO.get("status") == "pausa" or A["forecasts"].empty:
    st.title(M.name)
    e = REG.get(mk, {})
    if INFO.get("status") == "pausa":
        st.warning(f"**Mercado en pausa:** {e.get('reason', 'ningún plazo con evidencia')}. No se entrena ni se "
                   "emiten señales; se vuelve a evaluar automáticamente cada lunes.")
        rows = [{"Plazo": hl(h), "IC": v["ic"], "p IC": v["p_value_ic"], "Sharpe long-short": v["sharpe_long_short"],
                 "p Sharpe": v["p_value_sharpe"], "Deflated Sharpe": v.get("dsr"), "Evidencia": v.get("evidence", "-")}
                for h, v in sorted(GATE.items())]
        if rows:
            st.markdown("**Resultado de la última validación** (selección anidada):")
            st.dataframe(pd.DataFrame(rows).style.format({"IC": "{:.3f}", "p IC": "{:.3f}", "Sharpe long-short": "{:.2f}",
                                                          "p Sharpe": "{:.3f}", "Deflated Sharpe": "{:.0%}"}, na_rep="–"),
                         hide_index=True, width="stretch")
            how_to("- **IC** alto y **p IC** < 0.05: el modelo sí ordena las acciones mejor que el azar.\n"
                   "- Pero si el **Sharpe long-short** es bajo o negativo, esa ventaja no alcanza a pagar los costos "
                   "de operar (en la BMV se asumen 25 pb por operación).\n"
                   "- Por eso el mercado queda en pausa: hay señal, pero no es rentable aprovecharla.",
                   "¿Por qué está en pausa si ordena bien?")
    else:
        st.info("Aún no hay pronósticos para este mercado. Ejecuta `stockcast evaluate --market " + mk + "`.")
    experiments_table()
    st.stop()

FC = A["forecasts"].copy()
FC["sector"] = FC["ticker"].map(SECTOR).fillna("Otro")
stock_dates = pd.to_datetime(FC.loc[~FC["ticker"].str.endswith("-USD"), "as_of"])
AS_OF = (stock_dates if len(stock_dates) else pd.to_datetime(FC["as_of"])).max().date()

st.title(M.name)
st.markdown(X.INTRO)
if M.pit_top_n:
    st.caption(f"Universo dinámico: de {len(M.tickers)} acciones candidatas (S&P 500 + Nasdaq-100), en cada fecha "
               f"entran sólo las {M.pit_top_n} con mayor volumen operado de los {M.pit_window} días previos. "
               "Las señales de hoy se refieren a esas acciones.")

tabs = st.tabs(["🎯 Señales", "🔍 Acción", "🌎 Entorno económico", "📰 Noticias", "🧪 ¿Qué tan confiable es?",
                "📘 Cómo funciona"])


# =====================================================================
# 1. SEÑALES
# =====================================================================
with tabs[0]:
    hs = sorted(FC["horizon"].unique())
    valid = [h for h in hs if evidence(h) != "ninguna"]
    c = st.columns(4)
    c[0].metric("Acciones analizadas", FC["ticker"].nunique())
    c[1].metric("Datos al cierre de", str(AS_OF))
    n_strong = sum(evidence(h) == "fuerte" for h in hs)
    c[2].metric("Plazos con señal", f"{len(valid)} de {len(hs)}",
                f"{n_strong} con evidencia fuerte" if n_strong else "evidencia moderada" if valid else None,
                delta_color="off",
                help="Plazos en los que el modelo, en datos que no vio, ordenó mejor que el azar y ganó con una "
                     "estrategia long-short. 'Fuerte' = además sobrevive a la corrección por pruebas múltiples.")
    c[3].metric("Sectores", FC["sector"].nunique())

    h_sig = st.segmented_control(
        "Plazo de inversión", hs, default=valid[0] if valid else hs[-1],
        format_func=lambda h: f"{hl(h)} " + {"fuerte": "✅", "moderada": "🟡"}.get(evidence(h), "(sin señal)"))
    h_sig = h_sig or (valid[0] if valid else hs[-1])
    cur = FC[FC["horizon"] == h_sig].set_index("ticker").sort_values("ret_q0.5", ascending=False)
    is_valid = h_sig in valid

    if is_valid:
        g = GATE[h_sig]
        base = (f"En la validación, el modelo (*{g['engine']}*) ordenó las acciones mejor que el azar "
                f"(IC {g['ic']:.3f}) y la estrategia long-short logró un Sharpe de {g['sharpe_long_short']:.2f} con costos.")
        if evidence(h_sig) == "fuerte":
            st.success(f"**Plazo {hl(h_sig)}: ✅ evidencia fuerte.** {base} Además sobrevive a la corrección por "
                       "haber probado muchas configuraciones.")
        else:
            dsr = g.get("dsr")
            st.warning(f"**Plazo {hl(h_sig)}: 🟡 evidencia moderada.** {base} Pero al descontar todas las "
                       f"configuraciones probadas, la probabilidad de que la ganancia sea real es "
                       f"{dsr:.0%} (se exige 95% para 'fuerte'). Úsalo con cautela: montos pequeños o en *paper*."
                       if dsr is not None else f"**Plazo {hl(h_sig)}: 🟡 evidencia moderada.** {base}")
    else:
        st.warning(f"**Plazo {hl(h_sig)}: sin señal.** En este plazo el modelo no demostró ventaja sobre el azar. "
                   "Los pronósticos se muestran sólo como referencia; no se recomienda operar con ellos.")

    # --- tarjetas: mejores y peores perspectivas ---
    n_show = 5
    left, right = st.columns(2)
    for col, title, df, color in (
            (left, "🟢 Mejores perspectivas" + (" · COMPRA" if is_valid else ""), cur.head(n_show), GREEN),
            (right, "🔴 Peores perspectivas" + (" · VENTA EN CORTO" if is_valid else ""), cur.tail(n_show)[::-1], RED)):
        with col.container(border=True):
            st.markdown(f"#### {title}")
            for t, r in df.iterrows():
                a, b, cc = st.columns([2.2, 1.2, 2])
                a.markdown(f"**{t}**  \n<span style='color:{GRAY};font-size:0.85em'>{r['sector']}</span>",
                           unsafe_allow_html=True)
                b.markdown(f"<span style='color:{color};font-size:1.25em;font-weight:600'>{pct(r['ret_q0.5'])}</span>",
                           unsafe_allow_html=True)
                cc.markdown(f"<span style='font-size:0.85em'>rango {pct(r['ret_q0.1'])} a {pct(r['ret_q0.9'])}"
                            f"<br>precio {r['close']:,.2f} → {r['price_q0.5']:,.2f}</span>", unsafe_allow_html=True)
    how_to(X.HOW_TO_READ_SIGNALS, "¿Qué significa cada señal y cada número?")

    # --- por sector ---
    st.subheader(f"Retorno esperado por sector · {hl(h_sig)}")
    by_sec = cur.groupby("sector")["ret_q0.5"].agg(["mean", "count"]).sort_values("mean")
    fig = go.Figure(go.Bar(x=by_sec["mean"], y=by_sec.index, orientation="h",
                           marker_color=[GREEN if v > by_sec["mean"].median() else RED for v in by_sec["mean"]],
                           text=[f"{n} acc." for n in by_sec["count"]], textposition="outside",
                           hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
    fig.update_layout(xaxis_tickformat="+.1%", xaxis_title="Retorno esperado promedio (mediana)")
    st.plotly_chart(compact(fig, max(260, 34 * len(by_sec))), width="stretch")
    how_to("Promedio del retorno esperado de las acciones de cada sector. Verde = por encima de la mediana de "
           "sectores. Sirve para ver **qué tipo de empresas** favorece el modelo en el entorno actual (p. ej. "
           "defensivas como consumo básico y salud vs. cíclicas como tecnología o materiales).")

    # --- tabla completa ---
    st.subheader("Todas las acciones")
    f1, f2 = st.columns(2)
    sectors = sorted(cur["sector"].unique())
    sec_sel = f1.multiselect("Sector", sectors, placeholder="Todos los sectores")
    sig_opts = ["COMPRA", "VENTA EN CORTO", "NEUTRAL", "SIN SEÑAL"]
    sig_sel = f2.multiselect("Señal", sig_opts, placeholder="Todas las señales")
    view = cur.copy()
    if sec_sel:
        view = view[view["sector"].isin(sec_sel)]
    if sig_sel:
        view = view[view["signal"].isin(sig_sel)]
    other = FC.pivot(index="ticker", columns="horizon", values="ret_q0.5")
    table = pd.DataFrame({
        "Sector": view["sector"], "Señal": view["signal"], "Precio": view["close"],
        f"Esperado {hl(h_sig)}": view["ret_q0.5"], "Pesimista (p10)": view["ret_q0.1"],
        "Optimista (p90)": view["ret_q0.9"],
        **{f"Esperado {hl(h)}": other.loc[view.index, h] for h in hs if h != h_sig},
    })
    sig_color = {"COMPRA": "background-color: rgba(30,158,90,.22)", "VENTA EN CORTO": "background-color: rgba(208,69,58,.22)"}
    pcols = [c for c in table.columns if c not in ("Sector", "Señal", "Precio")]
    st.dataframe(table.style.format({"Precio": "{:,.2f}", **{c: "{:+.2%}" for c in pcols}})
                 .map(lambda v: sig_color.get(v, ""), subset=["Señal"])
                 .background_gradient(cmap="RdYlGn", subset=[f"Esperado {hl(h_sig)}"], vmin=-0.06, vmax=0.06),
                 width="stretch", height=min(45 + 35 * len(table), 640))
    st.caption(f"{len(table)} acciones. Para generar órdenes: `stockcast orders --market {mk} --horizon {h_sig}`.")

    # --- rango ---
    st.subheader(f"Rango esperado · {hl(h_sig)}")
    rng = view.sort_values("ret_q0.5")
    if len(rng) > 40:
        rng = pd.concat([rng.head(20), rng.tail(20)])
        st.caption("Mostrando las 20 con mejores y las 20 con peores perspectivas (filtra por sector para ver otras).")
    colors = [GREEN if s == "COMPRA" else RED if s == "VENTA EN CORTO" else BLUE for s in rng["signal"]]
    fig = go.Figure(go.Scatter(
        x=rng["ret_q0.5"], y=rng.index, mode="markers", marker=dict(size=10, color=colors),
        error_x=dict(type="data", symmetric=False, array=rng["ret_q0.9"] - rng["ret_q0.5"],
                     arrayminus=rng["ret_q0.5"] - rng["ret_q0.1"], color="#c5cfdb", thickness=2),
        customdata=np.column_stack([rng["ret_q0.1"], rng["ret_q0.9"], rng["sector"]]),
        hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>Esperado %{x:+.2%}<br>"
                      "Rango %{customdata[0]:+.2%} a %{customdata[1]:+.2%}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=GRAY)
    fig.update_layout(xaxis_tickformat="+.0%", showlegend=False, hovermode="closest")
    st.plotly_chart(compact(fig, max(320, 24 * len(rng))), width="stretch")
    how_to(X.HOW_TO_READ_RANGE)


# =====================================================================
# 2. ACCIÓN
# =====================================================================
with tabs[1]:
    s1, s2 = st.columns([1, 3])
    sec_pick = s1.selectbox("Sector", ["Todos"] + sorted(FC["sector"].unique()))
    opts = sorted(FC.loc[FC["sector"] == sec_pick, "ticker"].unique()) if sec_pick != "Todos" else sorted(FC["ticker"].unique())
    tick = s2.selectbox("Acción", opts)
    px_all = load_prices(tuple(M.tickers))
    g = px_all[px_all["ticker"] == tick].set_index("date").sort_index()
    f_t = FC[FC["ticker"] == tick].sort_values("horizon")
    close = g["close"]
    st.header(f"{tick} · {SECTOR.get(tick, '')}")

    cols = st.columns(len(f_t) + 1)
    cols[0].metric("Último cierre", f"{close.iloc[-1]:,.2f} {M.currency}")
    for col, (_, r) in zip(cols[1:], f_t.iterrows()):
        badge = {"fuerte": " ✅", "moderada": " 🟡"}.get(evidence(r["horizon"]), " · sin señal")
        col.metric(f"{hl(r['horizon'])}{badge}", f"{r['price_q0.5']:,.2f}", pct(r["ret_q0.5"], 2),
                   help=f"Rango p10–p90: {r['price_q0.1']:,.2f} a {r['price_q0.9']:,.2f}. Señal: {r['signal']}")

    # lectura técnica en lenguaje simple
    sma50, sma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
    rsi = _rsi(close).iloc[-1]
    ret6m = close.iloc[-1] / close.iloc[-127] - 1 if len(close) > 127 else np.nan
    vol_now = np.log(close).diff().tail(21).std() * np.sqrt(252)
    vol_hist = np.log(close).diff().tail(756).std() * np.sqrt(252)
    dd52 = close.iloc[-1] / close.tail(252).max() - 1
    trend = ("📈 Alcista" if close.iloc[-1] > sma50 > sma200 else "📉 Bajista" if close.iloc[-1] < sma50 < sma200
             else "↔️ Mixta")
    readings = [
        ("Tendencia", trend, f"Precio vs. medias de 50 y 200 días ({close.iloc[-1] / sma200 - 1:+.1%} sobre la de 200)."),
        ("Momentum 6 meses", pct(ret6m), "Ganadoras de 3–12 meses tienden a seguir ganando (efecto momentum)."),
        ("RSI 14", f"{rsi:.0f} · " + ("sobrecompra" if rsi > 70 else "sobreventa" if rsi < 30 else "neutral"),
         "> 70 subió muy rápido (riesgo de corrección); < 30 cayó muy rápido (posible rebote)."),
        ("Volatilidad", f"{vol_now:.0%} anual · " + ("más alta de lo normal" if vol_now > 1.2 * vol_hist
                                                    else "más baja de lo normal" if vol_now < 0.8 * vol_hist else "normal"),
         f"Último mes vs. su promedio de 3 años ({vol_hist:.0%})."),
        ("Distancia al máximo anual", pct(dd52), "0% = en máximos del año; muy negativo = lejos de sus máximos."),
    ]
    rc = st.columns(len(readings))
    for col, (name, val, tip) in zip(rc, readings):
        with col.container(border=True):
            st.caption(name)
            st.markdown(f"**{val}**")
            st.caption(tip)

    gg = g.iloc[-400:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=gg.index, open=gg["open"], high=gg["high"], low=gg["low"], close=gg["close"],
                                 name="Precio", increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_trace(go.Scatter(x=gg.index, y=close.rolling(50).mean().loc[gg.index], name="Media 50 días",
                             line=dict(width=1.3, color="#f0a202")), row=1, col=1)
    fig.add_trace(go.Scatter(x=gg.index, y=close.rolling(200).mean().loc[gg.index], name="Media 200 días",
                             line=dict(width=1.3, color="#7d4cdb")), row=1, col=1)
    x0 = [gg.index[-1]] + list(pd.to_datetime(f_t["target_date"]))
    y0 = gg["close"].iloc[-1]
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.9"]), line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.1"]), fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(42,111,219,0.18)", name="Rango p10–p90"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.5"]), name="Pronóstico (escenario central)",
                             line=dict(color=BLUE, dash="dash", width=2), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=gg.index, y=_rsi(close).loc[gg.index], name="RSI 14", line=dict(color="#8e44ad")),
                  row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.07, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=GREEN, opacity=0.07, line_width=0, row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(compact(fig, 640), width="stretch")
    how_to(X.HOW_TO_READ_STOCK_CHART)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Noticias recientes")
        nd = load_news(tick, mk)
        if nd.empty:
            st.caption("Sin noticias disponibles.")
        else:
            st.caption(f"Tono promedio {nd['sentiment'].mean():+.2f} (−1 negativo · +1 positivo). "
                       "Contexto para ti: aún no entra al modelo.")
            for _, r in nd.head(8).iterrows():
                icon = "🟢" if r["sentiment"] > 0.15 else "🔴" if r["sentiment"] < -0.15 else "⚪"
                when = r["published"].strftime("%d-%b") if pd.notna(r["published"]) else ""
                st.markdown(f"{icon} [{r['title']}]({r['url']}) <span style='color:{GRAY};font-size:0.8em'>"
                            f"· {r['source']} · {when}</span>", unsafe_allow_html=True)
    with right:
        st.subheader("Calidad del negocio (moat)")
        if tick.endswith("-USD"):
            st.caption("No aplica a criptomonedas.")
        else:
            mt = load_moat(tick)
            if not mt.empty:
                r = mt.iloc[0]
                fmt = {"Deuda LP / Utilidad neta": "{:.1f}x", "D&A / Utilidad bruta": "{:.0%}",
                       "Margen bruto %": "{:.0f}%", "ROE %": "{:.0f}%", "P/E": "{:.1f}", "P/B": "{:.1f}"}
                rows = [(k, (f.format(r[k]) if pd.notna(r.get(k)) and r.get(k) is not None else "–"))
                        for k, f in fmt.items()]
                st.dataframe(pd.DataFrame(rows, columns=["Indicador", "Valor"]), hide_index=True, width="stretch")
                st.markdown("**Moat probable** ✅" if r.get("Moat probable") else "**Moat probable** ✗")
                st.caption("Ventaja competitiva duradera según Jiménez-Preciado et al. (2022): poca deuda de largo "
                           "plazo respecto a utilidades y poca depreciación respecto a la utilidad bruta. Contexto; "
                           "no entra al modelo.")


# =====================================================================
# 3. ENTORNO ECONÓMICO
# =====================================================================
with tabs[2]:
    if "bancos_centrales" in DROPPED:
        st.info("En este mercado las tasas de bancos centrales **no mejoraron** la predicción (se midió quitándolas), "
                "así que el modelo no las usa. Se muestran como contexto económico.")
    else:
        st.info("El modelo **sí usa** estas variables: el cambio de tasa a 90 días de ~40 bancos centrales, el ciclo "
                "monetario global, los diferenciales de tasas contra la Fed y los indicadores de estrés de FRED.")
    rates = load_rates()
    since = st.slider("Mostrar desde", 2010, pd.Timestamp.today().year, 2018)

    st.subheader("Bancos centrales clave")
    key = [a for a in ["US", "XM", "MX", "GB", "JP", "BR"] if a in rates.columns]
    kc = st.columns(len(key))
    for col, a in zip(kc, key):
        s = rates[a].dropna()
        ch90 = (s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=90))) * 100
        moves = s[s.diff().fillna(0) != 0]
        col.metric(central_banks.BANKS.get(a, a).split(" (")[0], f"{s.iloc[-1]:.2f}%",
                   f"{ch90:+.0f} pb en 90 d" if ch90 else "sin cambio en 90 d", delta_color="inverse",
                   help=f"Último movimiento: {moves.index[-1].date() if len(moves) else 'n/d'}")

    sel = st.multiselect("Comparar tasas de política", list(rates.columns), default=key,
                         format_func=lambda a: central_banks.BANKS.get(a, a))
    if sel:
        r = rates.loc[str(since):, sel].resample("W").last()
        fig = go.Figure([go.Scatter(x=r.index, y=r[a], name=central_banks.BANKS.get(a, a).split(" (")[0],
                                    line_shape="hv") for a in sel])
        fig.update_layout(yaxis_title="Tasa de política (%)")
        st.plotly_chart(compact(fig, 400), width="stretch")
    how_to(X.HOW_TO_READ_RATES)

    st.subheader("Ciclo monetario global")
    chg = (rates - rates.shift(90)) * 100
    breadth = (((chg < 0).sum(axis=1) - (chg > 0).sum(axis=1)) / chg.notna().sum(axis=1)).loc[str(since):]
    bnow = breadth.dropna().iloc[-1]
    st.markdown(f"**Hoy: {bnow:+.0%}** → " + (
        "la mayoría de los bancos centrales está **recortando** tasas (entorno de liquidez creciente)." if bnow > 0.15
        else "la mayoría está **subiendo** tasas (endurecimiento global)." if bnow < -0.15
        else "bancos centrales **divididos o en pausa**."))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breadth.index, y=breadth.clip(lower=0), fill="tozeroy", line=dict(width=0),
                             fillcolor="rgba(30,158,90,.35)", name="Recortando"))
    fig.add_trace(go.Scatter(x=breadth.index, y=breadth.clip(upper=0), fill="tozeroy", line=dict(width=0),
                             fillcolor="rgba(208,69,58,.35)", name="Subiendo"))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[-1, 1])
    st.plotly_chart(compact(fig, 260), width="stretch")
    how_to(X.HOW_TO_READ_CYCLE)

    with st.expander("📋 Tabla de todos los bancos centrales"):
        summ = central_banks.summary(rates)
        st.dataframe(summ.style.format({"Tasa actual %": "{:.2f}", "Cambio 3m (pb)": "{:+.0f}", "Cambio 12m (pb)": "{:+.0f}"})
                     .background_gradient(cmap="RdYlGn_r", subset=["Cambio 12m (pb)"], vmin=-200, vmax=200),
                     width="stretch", hide_index=True, height=420)
        st.caption("Fuente: BIS (Bank for International Settlements). Verde = recortes en 12 meses; rojo = alzas.")

    st.subheader("Termómetros de estrés e incertidumbre (FRED)")
    fr = load_fred()
    if not fr.empty:
        cols2 = st.columns(2)
        for i, sid in enumerate([s for s in X.FRED_SERIES if s in fr.columns]):
            meta = X.FRED_SERIES[sid]
            s = fr[sid].dropna()
            s = s.rolling(30).mean().dropna() if sid == "USEPUINDXD" else s
            now = s.iloc[-1]
            pctile = (s.loc["2010":] < now).mean()
            level = "alto" if pctile > 0.7 else "bajo" if pctile < 0.3 else "intermedio"
            if sid == "T10Y2Y" and now < 0:
                level = "invertida"
            with cols2[i % 2].container(border=True):
                st.markdown(f"**{meta['titulo']}**")
                st.markdown(f"Hoy: **{now:.2f}** · nivel **{level}** (más alto que el {pctile:.0%} de los días desde 2010)")
                ss = s.loc[str(since):]
                fig = go.Figure(go.Scatter(x=ss.index, y=ss, line=dict(color="#e67e22", width=1.6), name=sid))
                if sid == "T10Y2Y":
                    fig.add_hline(y=0, line_dash="dot", line_color=RED)
                fig.add_hline(y=float(s.loc["2010":].median()), line_dash="dot", line_color=GRAY,
                              annotation_text="mediana", annotation_position="top left")
                st.plotly_chart(compact(fig, 200), width="stretch")
                with st.expander("💡 Qué es y cómo interpretarlo"):
                    st.markdown(f"{meta['que_es']}\n\n- **Alto:** {meta['alto']}\n- **Bajo:** {meta['bajo']}")


# =====================================================================
# 4. NOTICIAS (por sector → acción)
# =====================================================================
with tabs[3]:
    st.info("Las noticias son **contexto para ti**: todavía no entran al modelo porque las fuentes gratuitas no dan "
            "historia suficiente. Se guardan a diario y, al juntar ~1 año, se medirá si mejoran la predicción.")
    sec_n = st.selectbox("Sector", sorted(FC["sector"].unique()), key="news_sector")
    tick_n = sorted(FC.loc[FC["sector"] == sec_n, "ticker"].unique())
    if st.button(f"Cargar noticias de {sec_n} ({len(tick_n)} acciones)", type="primary"):
        st.session_state[f"news_{mk}_{sec_n}"] = {t: load_news(t, mk) for t in tick_n}
    loaded = st.session_state.get(f"news_{mk}_{sec_n}")
    if loaded:
        agg = pd.DataFrame({t: {"tono": d["sentiment"].mean() if not d.empty else np.nan, "notas": len(d)}
                            for t, d in loaded.items()}).T.sort_values("tono")
        fig = go.Figure(go.Bar(x=agg["tono"], y=agg.index, orientation="h",
                               marker_color=[GREEN if v > 0.05 else RED if v < -0.05 else GRAY for v in agg["tono"]],
                               text=[f"{int(n)} notas" for n in agg["notas"]], textposition="outside"))
        fig.update_layout(xaxis_title="Tono promedio de los titulares (−1 negativo · +1 positivo)", xaxis_range=[-1, 1])
        st.plotly_chart(compact(fig, max(220, 34 * len(agg))), width="stretch")
        for t in agg.index[::-1]:
            d = loaded[t]
            sig = FC[(FC["ticker"] == t) & (FC["horizon"] == (valid[0] if valid else hs[-1]))]["signal"]
            with st.expander(f"{t} · tono {agg.loc[t, 'tono']:+.2f} · señal del modelo: {sig.iloc[0] if len(sig) else '–'}"):
                if d.empty:
                    st.caption("Sin noticias.")
                for _, r in d.iterrows():
                    icon = "🟢" if r["sentiment"] > 0.15 else "🔴" if r["sentiment"] < -0.15 else "⚪"
                    when = r["published"].strftime("%d-%b-%Y") if pd.notna(r["published"]) else ""
                    st.markdown(f"{icon} [{r['title']}]({r['url']}) <span style='color:{GRAY};font-size:0.8em'>"
                                f"· {r['source']} · {when}</span>", unsafe_allow_html=True)
        how_to("El **tono** se calcula con VADER, un analizador de sentimiento léxico que puntúa cada titular de −1 "
               "(muy negativo) a +1 (muy positivo). Es una lectura rápida del ánimo de la prensa, no un análisis "
               "profundo: titulares neutros o técnicos quedan cerca de 0.", "¿Cómo se calcula el tono?")
    else:
        st.caption("Elige un sector y presiona el botón (se consultan las noticias en ese momento).")


# =====================================================================
# 5. ¿QUÉ TAN CONFIABLE ES?
# =====================================================================
with tabs[4]:
    st.markdown("Todo lo de esta pestaña se midió con datos que el modelo **no vio al entrenar** (validación "
                "*walk-forward*), simulando cómo habría funcionado en el pasado reciente.")
    experiments_table()
    if GATE:
        st.subheader("Veredicto por plazo")
        gc = st.columns(len(GATE))
        for col, (h, v) in zip(gc, sorted(GATE.items())):
            with col.container(border=True):
                ev = evidence(h)
                st.markdown(f"### {hl(h)} {({'fuerte': '✅', 'moderada': '🟡'}).get(ev, '❌')}")
                st.markdown(f"**{({'fuerte': 'Evidencia fuerte', 'moderada': 'Evidencia moderada'}).get(ev, 'Sin ventaja demostrada')}**")
                lines = [f"- Ordena mejor que el azar: **{'sí' if v['p_value_ic'] < 0.05 else 'no'}** "
                         f"(IC {v['ic']:.3f}, p={v['p_value_ic']:.3f})",
                         f"- Long-short gana: **{'sí' if v['sharpe_long_short'] >= 0.3 else 'no'}** "
                         f"(Sharpe {v['sharpe_long_short']:.2f}, p={v['p_value_sharpe']:.3f})"]
                if v.get("dsr") is not None:
                    lines.append(f"- Sobrevive a {v.get('n_trials', '?')} pruebas: "
                                 f"**{'sí' if v['dsr'] >= CFG['gate'].get('min_dsr', 0.95) else 'no'}** "
                                 f"(Deflated Sharpe {v['dsr']:.0%})")
                lines.append(f"- Modelo: {v['engine']}" + (" · selección anidada" if v.get("method") == "nested" else ""))
                st.markdown("\n".join(lines))
        how_to("- **IC**: qué tan bien ordena las acciones (0 = como el azar; 0.05–0.10 ya es valioso en finanzas).\n"
               "- **Sharpe long-short**: ganancia por unidad de riesgo de comprar las mejores y vender en corto las "
               "peores, con costos. > 0.5 aceptable, > 1 muy bueno.\n"
               "- **p**: probabilidad de que el resultado sea suerte. < 0.05 → evidencia real.\n"
               "- **Deflated Sharpe**: probabilidad de que la ganancia sea real *después de descontar* que se probaron "
               "muchas configuraciones (modelos × plazos). Se exige ≥ 95%.\n"
               "- **Selección anidada**: el modelo de cada periodo de prueba se eligió sólo con periodos anteriores, "
               "así que el resultado no está inflado por haber escogido 'el mejor' mirando todo.\n"
               "- ✅ **Fuerte**: cumple todo lo anterior. 🟡 **Moderada**: ordena mejor que el azar y gana, pero no "
               "llega al 95% de Deflated Sharpe. ❌ **Ninguna**: sin señal.",
               "¿Qué significan estos números?")

    st.subheader("📡 Seguimiento en vivo")
    live = load_tracking(mk, stamp)
    if live.empty:
        st.info("Todavía no vence ningún pronóstico guardado. Cada corrida diaria guarda sus pronósticos; cuando "
                "llega la fecha objetivo se comparan aquí con lo que realmente pasó. Es la prueba más honesta "
                "porque no involucra ninguna selección posterior.")
    else:
        st.dataframe(live.assign(horizon=live["horizon"].map(hl)).rename(columns={
            "horizon": "Plazo", "fechas": "Fechas de pronóstico", "pronosticos": "Pronósticos vencidos",
            "ic_real": "IC real", "long_short_real": "Long-short real (prom.)", "cobertura_real": "Dentro del rango p10–p90"})
            .style.format({"IC real": "{:.3f}", "Long-short real (prom.)": "{:+.2%}", "Dentro del rango p10–p90": "{:.0%}"}),
            hide_index=True, width="stretch")
        st.caption("Con pocas fechas el resultado es muy ruidoso; se vuelve informativo después de varios meses.")

    bt = A["backtest"]
    if not bt.empty:
        hb = CFG["backtest"]["horizon"]
        st.subheader(f"¿Cuánto habría ganado? (rebalanceo cada {hl(hb)}, con costos)")
        names = {"long_short": ("Estrategia long-short (compra mejores, vende peores)", BLUE, 3),
                 "estrategia": ("Sólo comprar las mejores", GREEN, 1.6),
                 "equiponderado": ("Comprar todas por igual", GRAY, 1.4),
                 "benchmark": (f"Índice {M.benchmark}", "#f0a202", 1.4)}
        fig = go.Figure()
        for k, (label, color, w) in names.items():
            if f"{k}_equity" in bt:
                fig.add_trace(go.Scatter(x=bt.index, y=(bt[f"{k}_equity"] - 1), name=label, line=dict(color=color, width=w)))
        fig.add_hline(y=0, line_color=GRAY, line_dash="dot")
        fig.update_layout(yaxis_tickformat="+.0%", yaxis_title="Ganancia acumulada")
        st.plotly_chart(compact(fig, 380), width="stretch")
        summ = INFO.get("metrics", {}).get("backtest", {})
        if summ:
            label_map = {k: v[0] for k, v in names.items()}
            tb = pd.DataFrame(summ).T.rename(index=label_map).rename(columns={
                "retorno_total": "Ganancia total", "cagr": "Por año", "sharpe": "Sharpe", "max_drawdown": "Peor caída"})
            st.dataframe(tb.style.format({"Ganancia total": "{:+.1%}", "Por año": "{:+.1%}", "Sharpe": "{:.2f}",
                                          "Peor caída": "{:.1%}"}), width="stretch")
        how_to("- La **línea azul gruesa** es lo que el modelo sabe hacer: ganar con la *diferencia* entre las "
               "mejores y las peores, sin depender de que el mercado suba.\n"
               "- **Sólo comprar las mejores** depende también de la dirección del mercado.\n"
               "- Si la línea azul sube de forma constante, el ranking funciona; si es plana o cae, no.\n"
               "- **Peor caída** (drawdown): la mayor pérdida desde un máximo que habrías tenido que aguantar.\n"
               "- Incluye 10 pb de costo por operación y 50 pb anuales por pedir prestadas las acciones en corto.")

    if A["ablation"]:
        st.subheader("¿Qué información le sirve al modelo?")
        contrib = pd.DataFrame(A["ablation"]["contribution"]).set_index("grupo")
        contrib = contrib.rename(index=lambda g: GROUPS.get(g, g).split(" (")[0])
        v = contrib["aporte_sharpe"].sort_values()
        fig = go.Figure(go.Bar(x=v, y=v.index, orientation="h",
                               marker_color=[GREEN if x > 0 else "#cfd6de" for x in v],
                               customdata=contrib.loc[v.index, "aporte_ic"],
                               hovertemplate="%{y}<br>Aporte al Sharpe: %{x:+.2f}<br>Aporte al IC: "
                                             "%{customdata:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color=GRAY)
        fig.update_layout(xaxis_title="Cuánto empeora el Sharpe long-short si se quita el grupo", hovermode="closest")
        st.plotly_chart(compact(fig, 240), width="stretch")
        if DROPPED:
            st.caption("Descartado(s) en este mercado por no aportar: **"
                       + ", ".join(GROUPS.get(g, g).split(" (")[0] for g in DROPPED) + "**.")
        how_to("Se reentrenó el modelo quitando un grupo de variables a la vez. Barra **verde** = sin ese grupo la "
               "estrategia gana menos, o sea, el grupo **sirve**. Barra **gris** (≤ 0) = quitarlo da igual o "
               "mejora: el grupo se descarta automáticamente para no saturar al modelo con ruido.")

    imp = A["importance"]
    if not imp.empty:
        st.subheader("Variables con más peso")
        h_sel = st.segmented_control("Plazo", list(imp.columns), default=imp.columns[-1],
                                     format_func=lambda c: hl(c.lstrip("h")), key="imp_h") or imp.columns[-1]
        top = imp[h_sel].sort_values().tail(15)
        info = glossary.table(top.index).set_index("variable")
        fig = go.Figure(go.Bar(
            x=top / imp[h_sel].sum(), y=[info.loc[c, "nombre"] for c in top.index], orientation="h",
            marker_color=[{"Técnicos de la acción": BLUE, "Mercado y macro": "#f0a202",
                           "Tasas de bancos centrales": "#7d4cdb"}.get(info.loc[c, "grupo"], "#e67e22") for c in top.index],
            customdata=[[c, info.loc[c, "descripcion"], info.loc[c, "grupo"]] for c in top.index],
            hovertemplate="<b>%{y}</b><br>%{customdata[1]}<br><i>%{customdata[2]}</i> · peso %{x:.1%}<extra></extra>"))
        fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Peso en el modelo", hovermode="closest")
        st.plotly_chart(compact(fig, 480), width="stretch")
        st.caption("Color = familia: azul técnicos · naranja macro · morado bancos centrales · ocre estrés (FRED). "
                   "Pasa el mouse para ver qué mide cada variable.")

    with st.expander("🔬 Detalle técnico: comparación de todos los modelos"):
        cmp = A["comparison"]
        if cmp.empty:
            st.caption("Ejecuta `stockcast compare` para generarla.")
        else:
            show = cmp.rename(columns={"horizon": "Plazo (d)", "engine": "Modelo", "ic": "IC", "p_value_ic": "p IC",
                                       "dsr": "Deflated Sharpe",
                                       "sharpe_long_short": "Sharpe L/S", "p_value_sharpe": "p Sharpe",
                                       "hit_rate": "Acierto dirección", "hit_always_up": "'Siempre sube'",
                                       "coverage": "Cobertura p10–p90", "seconds": "Segundos"})
            keep = [c for c in ["Plazo (d)", "Modelo", "IC", "p IC", "Sharpe L/S", "p Sharpe", "Deflated Sharpe",
                                "Acierto dirección",
                                "'Siempre sube'", "Cobertura p10–p90", "Segundos"] if c in show]
            st.dataframe(show[keep].style.format(precision=3, na_rep="–"), hide_index=True, width="stretch")
            st.caption("naive = caminata aleatoria (retorno esperado 0); momentum = sólo retorno de 6 meses; ridge = "
                       "lineal; lightgbm = árboles; ensemble = promedio de lightgbm y ridge; nested = selección anidada (lo que de "
                       "verdad se usa). Ningún modelo predice la "
                       "dirección mejor que 'siempre sube'; el valor está en el orden relativo (IC).")


# =====================================================================
# 6. CÓMO FUNCIONA
# =====================================================================
with tabs[5]:
    st.markdown(X.METHOD)
    st.subheader("Glosario")
    gcols = st.columns(2)
    for i, (k, v) in enumerate(X.CONCEPTS.items()):
        with gcols[i % 2].expander(k):
            st.markdown(v)
    st.subheader("📖 Diccionario de variables de este mercado")
    cols_used = list(A["importance"].index) if not A["importance"].empty else []
    if cols_used:
        dic = glossary.table(cols_used)
        q = st.text_input("Buscar variable", placeholder="p. ej. volatilidad, Fed, dólar, momentum…")
        if q:
            dic = dic[dic.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        st.dataframe(dic.rename(columns={"variable": "Código", "nombre": "Nombre", "descripcion": "Qué mide",
                                         "grupo": "Familia"})[["Familia", "Nombre", "Qué mide", "Código"]],
                     hide_index=True, width="stretch", height=min(45 + 35 * len(dic), 560))
        st.caption(f"{len(dic)} variables en este mercado. Lista completa: docs/VARIABLES.md")
