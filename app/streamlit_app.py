"""Dashboard local de stockcast.  Ejecutar:  stockcast app   (o  streamlit run app/streamlit_app.py)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stockcast import config, pipeline  # noqa: E402
from stockcast.data import cb_tone, central_banks, fred, fundamentals, market, news  # noqa: E402
from stockcast.features import _rsi  # noqa: E402

st.set_page_config(page_title="stockcast", page_icon="📈", layout="wide")
CFG = config.load()
HORIZON_LABEL = {1: "1 día", 5: "1 semana", 10: "2 semanas", 21: "1 mes"}


# ---------- carga con caché ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_artifacts(mk: str, _stamp: float):
    d = pipeline.artifacts_dir(mk)
    out = {"info": json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None}
    cmp = d / "comparison.json"
    out["comparison"] = pd.DataFrame(json.loads(cmp.read_text())) if cmp.exists() else pd.DataFrame()
    for name in ("forecasts", "backtest", "importance", "forecast_history"):
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


@st.cache_data(ttl=6 * 3600, show_spinner="Leyendo comunicados de la Fed…")
def load_tone():
    return cb_tone.fed_statements(refresh=True)


@st.cache_data(ttl=1800, show_spinner="Buscando noticias…")
def load_news(ticker: str, mk: str):
    return news.fetch_news(ticker, mk)


@st.cache_data(ttl=24 * 3600, show_spinner="Descargando estados financieros…")
def load_moat(tickers: tuple[str, ...]):
    return fundamentals.moat_table(list(tickers))


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


# ---------- barra lateral ----------
st.sidebar.title("📈 stockcast")
_avail = [k for k in config.markets() if (pipeline.artifacts_dir(k) / "forecasts.parquet").exists()] or config.markets()
mk = st.sidebar.radio("Mercado", _avail, format_func=lambda k: CFG["markets"][k]["name"])
M = config.market(mk)
adir = pipeline.artifacts_dir(mk)
stamp = (adir / "metrics.json").stat().st_mtime if (adir / "metrics.json").exists() else 0
A = load_artifacts(mk, stamp)

st.sidebar.markdown("### Actualizar")
full_eval = st.sidebar.checkbox("Incluir validación walk-forward (más lento)", value=False)
if st.sidebar.button("🔄 Descargar datos y reentrenar", width="stretch"):
    with st.spinner("Ejecutando pipeline… (1–5 min en CPU)"):
        pipeline.run(mk, refresh=True, evaluate=full_eval)
    st.cache_data.clear()
    st.rerun()
if A["info"]:
    st.sidebar.caption(f"Último entrenamiento: {A['info']['run_at']}  \n"
                       f"Motor: {A['info']['engine']} · {A['info']['n_features']} features · "
                       f"{A['info']['seconds']} s")
st.sidebar.caption("Uso educativo. No es recomendación de inversión.")

if A["forecasts"].empty:
    st.title(M.name)
    st.info("Aún no hay pronósticos para este mercado. Presiona **Descargar datos y reentrenar** "
            "o ejecuta `stockcast run --market " + mk + "` en la terminal.")
    st.stop()

fc = A["forecasts"]
tabs = st.tabs(["📊 Pronósticos", "🔍 Acción", "🏦 Bancos centrales", "📰 Noticias", "🧪 Validación"])

# ---------- 1. Pronósticos ----------
with tabs[0]:
    st.header(f"Pronósticos · {M.name}")
    metrics = (A["info"] or {}).get("metrics", {})
    h_main = max(CFG["horizons"])
    mh = metrics.get(str(h_main), {}) if metrics else {}
    m_main = mh.get(mh.get("engine", A["info"]["engine"]))
    c = st.columns(4)
    c[0].metric("Acciones", fc["ticker"].nunique())
    stock_dates = pd.to_datetime(fc.loc[~fc["ticker"].str.endswith("-USD"), "as_of"])
    c[1].metric("Datos al", str((stock_dates if len(stock_dates) else pd.to_datetime(fc["as_of"])).max().date()))
    if m_main:
        c[2].metric(f"Acierto de dirección ({HORIZON_LABEL.get(h_main, h_main)})", f"{m_main['hit_rate']:.1%}")
        c[3].metric("IC (ranking)", f"{m_main['ic']:.3f}",
                    help="Correlación de Spearman entre retorno pronosticado y real entre acciones, por día. "
                         "> 0.03 ya es útil para ordenar acciones.")
        if m_main["skill_vs_naive"] < 0:
            st.warning("Fuera de muestra el modelo **no** reduce el error absoluto frente a la caminata "
                       "aleatoria. Úsalo para *ordenar* acciones (IC), no como precio objetivo exacto.")

    gate_info = (A["info"] or {}).get("gate", {}).get("horizons", {})
    if gate_info:
        ok = [HORIZON_LABEL.get(int(h), h) for h, v in gate_info.items() if v["validated"]]
        if ok:
            st.success("Horizontes con ventaja demostrada fuera de muestra: **" + ", ".join(ok) + "**. "
                       "En los demás la tabla muestra el pronóstico pero sin señal de operación.")
        else:
            st.error("Ningún horizonte pasó la validación en este mercado: los pronósticos se muestran "
                     "sólo como referencia, sin señales.")
    hs = sorted(fc["horizon"].unique())
    valid_h = [h for h in hs if gate_info.get(str(h), gate_info.get(h, {})).get("validated")]
    h_sig = st.radio("Horizonte de la señal", hs, index=hs.index(valid_h[0]) if valid_h else len(hs) - 1,
                     format_func=lambda h: HORIZON_LABEL.get(h, f"{h} d"), horizontal=True)
    wide = fc.pivot(index="ticker", columns="horizon", values="ret_q0.5")
    wide.columns = [f"Mediana {HORIZON_LABEL.get(h, h)}" for h in wide.columns]
    last = fc[fc["horizon"] == h_sig].set_index("ticker")
    lbl = HORIZON_LABEL.get(h_sig, f"{h_sig} d")
    cols_sig = ["signal", "prob_up"] if "signal" in last else []
    table = pd.concat([last[cols_sig + ["close"]].rename(columns={"close": "Precio", "signal": "Señal",
                                                                   "prob_up": f"P(sube) {lbl}"}), wide,
                       last[["ret_q0.1", "ret_q0.9"]].rename(columns={"ret_q0.1": f"p10 {lbl}", "ret_q0.9": f"p90 {lbl}"})],
                      axis=1)
    table = table.sort_values(f"Mediana {lbl}", ascending=False)
    if "Señal" in table:
        fcol = st.columns(2)
        f_sel = fcol[0].multiselect("Señal", ["COMPRA", "VENTA EN CORTO", "NEUTRAL", "SIN SEÑAL"],
                                    default=["COMPRA", "VENTA EN CORTO", "NEUTRAL", "SIN SEÑAL"])
        table = table[table["Señal"].isin(f_sel)]
        if "country" in last and last["country"].nunique() > 1:
            countries = sorted(last["country"].dropna().unique())
            c_sel = fcol[1].multiselect("País / bolsa", countries, default=countries)
            table = table[last.loc[table.index, "country"].isin(c_sel)]
            table.insert(1, "País", last.loc[table.index, "country"])
    pct_cols = [c for c in table.columns if c not in ("Precio", "Señal", "País")]
    med_cols = [c for c in pct_cols if c.startswith("Mediana")]
    sig_color = {"COMPRA": "background-color: rgba(39,174,96,.25)",
                 "VENTA EN CORTO": "background-color: rgba(192,57,43,.25)"}
    sty = table.style.format({"Precio": "{:,.2f}", **{c: "{:+.2%}" for c in pct_cols if not c.startswith("P(")},
                              **{c: "{:.0%}" for c in pct_cols if c.startswith("P(")}})
    sty = sty.background_gradient(cmap="RdYlGn", subset=med_cols, vmin=-0.08, vmax=0.08)
    if "Señal" in table:
        sty = sty.map(lambda v: sig_color.get(v, ""), subset=["Señal"])
    st.dataframe(sty, width="stretch", height=min(40 + 35 * len(table), 700))
    st.caption("La señal es **relativa**: COMPRA = 20% de acciones con mayor retorno esperado; VENTA EN CORTO = 20% "
               "con menor. **P(sube)** es sólo orientativa: en validación no fue más precisa que la frecuencia "
               "histórica de subidas (Brier en *Validación*), así que no indica qué acciones subirán *seguro*; "
               "lo validado es el **orden** relativo. Genera órdenes con `stockcast orders --market " + mk + " --horizon " + str(h_sig) + "`.")

    st.subheader(f"Rango esperado a {lbl} (p10 – mediana – p90)")
    srt = last.sort_values("ret_q0.5").tail(40) if len(last) > 40 else last.sort_values("ret_q0.5")
    if len(last) > 40:
        st.caption("Mostrando las 40 con mayor retorno esperado.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=srt["ret_q0.5"], y=srt.index, mode="markers", name="Mediana",
                             marker=dict(size=10, color="#2a6fdb"),
                             error_x=dict(type="data", symmetric=False, array=srt["ret_q0.9"] - srt["ret_q0.5"],
                                          arrayminus=srt["ret_q0.5"] - srt["ret_q0.1"], color="#9db7e6")))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(300, 28 * len(srt)), xaxis_tickformat="+.0%", margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ---------- 2. Acción ----------
with tabs[1]:
    tick = st.selectbox("Acción", sorted(fc["ticker"].unique()))
    px = load_prices(tuple(M.tickers))
    g = px[px["ticker"] == tick].set_index("date").sort_index().iloc[-400:]
    f_t = fc[fc["ticker"] == tick].sort_values("horizon")
    st.header(tick)
    cols = st.columns(len(f_t) + 1)
    cols[0].metric("Último cierre", f"{g['close'].iloc[-1]:,.2f} {M.currency}")
    for col, (_, r) in zip(cols[1:], f_t.iterrows()):
        col.metric(HORIZON_LABEL.get(r["horizon"], r["horizon"]), f"{r['price_q0.5']:,.2f}", pct(r["ret_q0.5"]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=g.index, open=g["open"], high=g["high"], low=g["low"], close=g["close"],
                                 name="Precio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["close"].rolling(50).mean(), name="SMA 50",
                             line=dict(width=1, color="#f0a202")), row=1, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["close"].rolling(200).mean(), name="SMA 200",
                             line=dict(width=1, color="#7d4cdb")), row=1, col=1)
    x0 = [g.index[-1]] + list(pd.to_datetime(f_t["target_date"]))
    y0 = g["close"].iloc[-1]
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.9"]), line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.1"]), fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(42,111,219,0.2)", name="Banda p10–p90"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x0, y=[y0] + list(f_t["price_q0.5"]), name="Pronóstico (mediana)",
                             line=dict(color="#2a6fdb", dash="dash", width=2), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=_rsi(g["close"]), name="RSI 14", line=dict(color="#8e44ad")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#c0392b", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#27ae60", row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Noticias recientes")
        nd = load_news(tick, mk)
        if nd.empty:
            st.caption("Sin noticias.")
        else:
            st.metric("Sentimiento promedio", f"{nd['sentiment'].mean():+.2f}", help="-1 muy negativo · +1 muy positivo")
            for _, r in nd.iterrows():
                icon = "🟢" if r["sentiment"] > 0.15 else "🔴" if r["sentiment"] < -0.15 else "⚪"
                when = r["published"].strftime("%d-%m-%Y") if pd.notna(r["published"]) else ""
                st.markdown(f"{icon} [{r['title']}]({r['url']})  \n<small>{r['source']} · {when} · "
                            f"{r['sentiment']:+.2f}</small>", unsafe_allow_html=True)
    with right:
        st.subheader("Fundamentales y moat")
        mt = load_moat((tick,))
        if not mt.empty:
            st.dataframe(mt.T.rename(columns={tick: "Valor"}).astype(str), width="stretch")
            st.caption("Moat según Jiménez-Preciado et al. (2022): deuda LP/utilidad neta y D&A/utilidad bruta "
                       "bajos indican ventaja competitiva duradera.")

# ---------- 3. Bancos centrales ----------
with tabs[2]:
    st.header("Bancos centrales")
    rates = load_rates()
    summ = central_banks.summary(rates)
    c = st.columns(3)
    recent = rates.iloc[-1] - rates.asof(rates.index[-1] - pd.Timedelta(days=90))
    c[0].metric("Bancos con datos", len(summ))
    c[1].metric("Recortaron (90 d)", int((recent < 0).sum()))
    c[2].metric("Subieron (90 d)", int((recent > 0).sum()))
    st.dataframe(summ.style.format({"Tasa actual %": "{:.2f}", "Cambio 3m (pb)": "{:+.0f}", "Cambio 12m (pb)": "{:+.0f}"})
                 .background_gradient(cmap="RdYlGn_r", subset=["Cambio 12m (pb)"], vmin=-200, vmax=200),
                 width="stretch", height=400)
    default = [a for a in ["US", "XM", "GB", "JP", "MX", "BR", "CA"] if a in rates.columns]
    sel = st.multiselect("Comparar tasas", list(rates.columns), default=default,
                         format_func=lambda a: f"{a} · {central_banks.BANKS.get(a, a)}")
    since = st.slider("Desde", 2010, pd.Timestamp.today().year, 2018)
    if sel:
        r = rates.loc[str(since):, sel].resample("W").last()
        fig = go.Figure([go.Scatter(x=r.index, y=r[a], name=a, line_shape="hv") for a in sel])
        fig.update_layout(height=420, yaxis_title="Tasa de política %", hovermode="x unified",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    chg = (rates - rates.shift(90)) * 100
    breadth = (((chg < 0).sum(axis=1) - (chg > 0).sum(axis=1)) / chg.notna().sum(axis=1)).loc[str(since):]
    st.subheader("Ciclo monetario global")
    st.caption("(# bancos que recortaron − # que subieron) / total, ventana de 90 días. Positivo = relajación global.")
    fig = go.Figure(go.Scatter(x=breadth.index, y=breadth, fill="tozeroy", line=dict(color="#2a6fdb")))
    fig.update_layout(height=260, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Tono de la Fed (hawkish / dovish)")
        tone = load_tone()
        if tone.empty:
            st.caption("No se pudieron leer los comunicados.")
        else:
            fig = go.Figure(go.Bar(x=tone["date"], y=tone["tone"],
                                   marker_color=["#c0392b" if v > 0 else "#27ae60" for v in tone["tone"]],
                                   hovertext=tone["title"]))
            fig.update_layout(height=280, yaxis_range=[-1, 1], margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="← dovish · hawkish →")
            st.plotly_chart(fig, width="stretch")
            st.caption("Léxico local inspirado en Jiménez-Preciado et al. (2025), *Hawkish or Dovish?*, Mathematics 13(20).")
    with right:
        st.subheader("Estrés e incertidumbre (FRED)")
        fr = load_fred()
        if not fr.empty:
            col = st.selectbox("Serie", list(fr.columns), format_func=lambda s: CFG.get("fred", {}).get(s, s))
            s = fr.loc[str(since):, col]
            s = s.rolling(30).mean() if col == "USEPUINDXD" else s
            fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color="#e67e22")))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

# ---------- 4. Noticias ----------
with tabs[3]:
    st.header("Noticias y sentimiento")
    st.caption("Fuentes: Yahoo Finance y Google News (RSS). Sentimiento con "
               + ("FinBERT" if CFG["news"]["sentiment"] == "finbert" else "VADER (ligero, local)") + ".")
    if st.button("Cargar noticias de todas las acciones"):
        allnews = pd.concat([load_news(t, mk) for t in M.tickers], ignore_index=True)
        st.session_state[f"news_{mk}"] = allnews
    allnews = st.session_state.get(f"news_{mk}")
    if allnews is not None and not allnews.empty:
        agg = allnews.groupby("ticker")["sentiment"].agg(["mean", "count"]).sort_values("mean")
        fig = go.Figure(go.Bar(x=agg["mean"], y=agg.index, orientation="h",
                               marker_color=["#c0392b" if v < 0 else "#27ae60" for v in agg["mean"]],
                               text=agg["count"].astype(str) + " notas", textposition="outside"))
        fig.update_layout(height=max(300, 24 * len(agg)), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Sentimiento promedio")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(allnews[["ticker", "published", "source", "sentiment", "title", "url"]],
                     width="stretch", column_config={"url": st.column_config.LinkColumn("Enlace")})
    hist_path = config.path("data") / "news.parquet"
    if hist_path.exists():
        h = pd.read_parquet(hist_path)
        st.caption(f"Histórico local: {len(h):,} noticias. Cuando cubra ≥ {pipeline.MIN_NEWS_DAYS} días, "
                   "el sentimiento se usará automáticamente como feature del modelo.")

# ---------- 5. Validación ----------
with tabs[4]:
    st.header("Validación fuera de muestra")
    metrics = (A["info"] or {}).get("metrics", {})
    rows = []
    for h, v in metrics.items():
        if h == "backtest":
            continue
        for model_name in (v.get("engine", A["info"]["engine"]), "naive"):
            m = v.get(model_name)
            if m:
                rows.append({"Horizonte": HORIZON_LABEL.get(int(h), h), "Modelo": model_name, "MAE": m["mae"],
                             "Skill vs. naive": m["skill_vs_naive"], "Acierto dirección": m["hit_rate"],
                             "IC": m["ic"], "Cobertura p10–p90": m["coverage"]})
    if not rows:
        st.info("Ejecuta el pipeline con validación (`stockcast run`) para ver métricas.")
    else:
        st.dataframe(pd.DataFrame(rows).style.format({"MAE": "{:.4f}", "Skill vs. naive": "{:+.3f}",
                                                       "Acierto dirección": "{:.1%}", "IC": "{:.3f}",
                                                       "Cobertura p10–p90": "{:.1%}"}),
                     width="stretch", hide_index=True)
        st.caption("Skill > 0: menor error que la caminata aleatoria. IC: capacidad de ordenar acciones. "
                   "Las bandas se calibran con conformal para acercarse al 80% de cobertura.")
    bt = A["backtest"]
    if not bt.empty:
        st.subheader(f"Backtest: top {CFG['backtest']['top_k']} cada {CFG['backtest']['horizon']} días "
                     f"(costo {CFG['backtest']['cost_bps']} pb por lado)")
        fig = go.Figure()
        names = {"estrategia": "Long top-k", "long_short": "Long-short (neutral al mercado)",
                 "equiponderado": "Equiponderado universo",
                 "benchmark": f"Benchmark {M.benchmark}"}
        for k, label in names.items():
            if f"{k}_equity" in bt:
                fig.add_trace(go.Scatter(x=bt.index, y=bt[f"{k}_equity"], name=label))
        fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        summ = metrics.get("backtest", {})
        if summ:
            st.dataframe(pd.DataFrame(summ).T.rename(columns={"retorno_total": "Retorno total", "cagr": "CAGR",
                                                              "sharpe": "Sharpe", "max_drawdown": "Máx. drawdown"})
                         .style.format({"Retorno total": "{:+.1%}", "CAGR": "{:+.1%}", "Sharpe": "{:.2f}",
                                        "Máx. drawdown": "{:.1%}"}), width="stretch")
    imp = A["importance"]
    if not imp.empty:
        st.subheader("Variables más importantes")
        h_sel = st.selectbox("Horizonte", imp.columns, index=len(imp.columns) - 1)
        top = imp[h_sel].sort_values().tail(20)
        fig = go.Figure(go.Bar(x=top, y=top.index, orientation="h", marker_color="#2a6fdb"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    cmp = A["comparison"]
    if not cmp.empty:
        st.subheader("Comparación de modelos (`stockcast compare`)")
        st.caption("Todo fuera de muestra (walk-forward). **p-valor** = probabilidad de obtener ese resultado con un "
                   "modelo aleatorio (permutación de pronósticos entre acciones). p < 0.05 → mejor que el azar. "
                   "El acierto de dirección se compara contra *siempre sube*, no contra 50%.")
        show = cmp.rename(columns={"horizon": "Horizonte", "engine": "Modelo", "hit_rate": "Acierto",
                                   "hit_always_up": "Acierto 'siempre sube'", "p_value_hit": "p acierto",
                                   "ic": "IC", "p_value_ic": "p IC", "sharpe_long_short": "Sharpe L/S",
                                   "p_value_sharpe": "p Sharpe", "sharpe_long": "Sharpe long",
                                   "sharpe_equal_weight": "Sharpe equiponderado", "coverage": "Cobertura",
                                   "seconds": "Segundos"})
        keep = [c for c in ["Horizonte", "Modelo", "Acierto", "Acierto 'siempre sube'", "p acierto", "IC", "p IC",
                            "Sharpe L/S", "p Sharpe", "Sharpe long", "Sharpe equiponderado", "Cobertura", "Segundos"]
                if c in show]
        pcols = [c for c in keep if c.startswith("p ")]
        st.dataframe(show[keep].style.format(precision=3, na_rep="–")
                     .map(lambda v: "color: #27ae60; font-weight: 600" if isinstance(v, float) and v < 0.05 else "",
                          subset=pcols),
                     width="stretch", hide_index=True)
