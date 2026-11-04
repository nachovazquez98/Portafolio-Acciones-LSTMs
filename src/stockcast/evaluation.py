"""Validación walk-forward, métricas fuera de muestra y backtest de una estrategia simple.

- Walk-forward con *purga*: entre el último día de entrenamiento y el primero de
  prueba se dejan h días para que el objetivo y_h del train no vea el periodo de test.
- Todo modelo se compara contra la caminata aleatoria (retorno esperado 0). Si no le
  gana, el dashboard lo dice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockcast.models import make_model


def walk_forward_splits(dates: pd.DatetimeIndex, n_splits: int, horizon: int, test_frac: float = 0.4):
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    test_len = int(len(dates) * test_frac) // n_splits
    first_test = len(dates) - test_len * n_splits
    for k in range(n_splits):
        a = first_test + k * test_len
        b = a + test_len if k < n_splits - 1 else len(dates)
        train_end = dates[a - horizon - 1]
        yield train_end, dates[a], dates[b - 1]


def oos_predictions(panel: pd.DataFrame, features: list[str], horizon: int, engine: str,
                    quantiles: list[float], params: dict, n_splits: int) -> pd.DataFrame:
    """Predicciones fuera de muestra concatenadas de todos los folds."""
    target = f"y_{horizon}"
    dates = panel.index.get_level_values("date")
    out = []
    for train_end, test_start, test_end in walk_forward_splits(dates, n_splits, horizon):
        train = panel[(dates <= train_end)].dropna(subset=[target])
        # contexto previo (features ya usan sólo el pasado; esto mantiene la interfaz por fecha)
        ctx = panel[(dates > test_start - pd.Timedelta(days=200)) & (dates <= test_end)]
        model = make_model(engine, quantiles, params).fit(train[features], train[target], horizon)
        pred = model.predict(ctx[features])
        test_mask = ctx.index.get_level_values("date") >= test_start
        pred = pred[test_mask].join(ctx.loc[test_mask, [target]].rename(columns={target: "y"}))
        out.append(pred.assign(fold=str(test_start.date())))
    return pd.concat(out).dropna(subset=["y"])


def metrics(pred: pd.DataFrame, quantiles: list[float]) -> dict:
    lo, med, hi = (f"q{q}" for q in (min(quantiles), 0.5, max(quantiles)))
    p = pred.dropna(subset=[med])
    err = p["y"] - p[med]
    nz = p["y"] != 0
    ic = p.groupby(level="date").apply(
        lambda g: g[med].corr(g["y"], method="spearman") if len(g) > 2 and g[med].nunique() > 1 else np.nan)
    return {
        "n": int(len(p)),
        "mae": float(err.abs().mean()),
        "mae_naive": float(p["y"].abs().mean()),           # caminata aleatoria: predice 0
        "skill_vs_naive": float(1 - err.abs().mean() / p["y"].abs().mean()),
        # acierto de dirección (NaN si el modelo nunca toma postura, p. ej. naive)
        "hit_rate": float((np.sign(p.loc[nz, med]) == np.sign(p.loc[nz, "y"])).mean())
        if (p[med] != 0).any() else np.nan,
        # referencia honesta: en un periodo alcista "siempre sube" ya acierta > 50%
        "hit_always_up": float((p.loc[nz, "y"] > 0).mean()),
        "ic": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
        "coverage": float(((p["y"] >= p[lo]) & (p["y"] <= p[hi])).mean()),
        **_prob_calibration(p, quantiles),
        "coverage_target": max(quantiles) - min(quantiles),
    }


def _prob_calibration(p: pd.DataFrame, quantiles: list[float]) -> dict:
    """¿P(sube) está calibrada? Brier score vs. usar siempre la frecuencia histórica de subidas,
    y tasa real de subida entre los casos con P(sube) >= 0.65 / <= 0.35 (las señales "más seguras")."""
    from stockcast.signals import prob_up_vec
    pu = prob_up_vec(p[[f"q{q}" for q in quantiles]].to_numpy(), quantiles)
    up = (p["y"] > 0).to_numpy().astype(float)
    base = up.mean()
    hi, lo = pu >= 0.65, pu <= 0.35
    return {
        "brier": float(np.mean((pu - up) ** 2)),
        "brier_base": float(np.mean((base - up) ** 2)),
        "n_confident_up": int(hi.sum()), "up_rate_when_confident_up": float(up[hi].mean()) if hi.any() else np.nan,
        "n_confident_down": int(lo.sum()), "up_rate_when_confident_down": float(up[lo].mean()) if lo.any() else np.nan,
    }


def conformal_scale(pred: pd.DataFrame, quantiles: list[float]) -> float:
    """Factor k para ensanchar/estrechar las bandas y lograr la cobertura nominal.

    Versión simple de Conformalized Quantile Regression (Romano, Patterson & Candès, 2019):
    intervalo calibrado = mediana ± k·(distancia al cuantil), con k elegido sobre
    residuos fuera de muestra.
    """
    lo, med, hi = (f"q{q}" for q in (min(quantiles), 0.5, max(quantiles)))
    p = pred.dropna(subset=[lo, med, hi])
    target = max(quantiles) - min(quantiles)
    for k in np.arange(0.5, 4.01, 0.05):
        inside = (p["y"] >= p[med] - k * (p[med] - p[lo])) & (p["y"] <= p[med] + k * (p[hi] - p[med]))
        if inside.mean() >= target:
            return float(round(k, 2))
    return 4.0


def backtest(pred: pd.DataFrame, horizon: int, top_k: int, cost_bps: float,
             benchmark: pd.Series | None = None, borrow_bps_annual: float = 50.0,
             y_step: pd.Series | None = None, step: int = 5) -> pd.DataFrame:
    """Cada `horizon` días rebalancea según el retorno mediano pronosticado.

    Si se pasa `y_step` (retorno a `step` días de cada acción) y horizon > step, usa portafolios
    traslapados (Jegadeesh & Titman 1993): ver `backtest_overlapping`.

    - estrategia: compra (long) las top_k acciones.
    - long_short: compra top_k y vende en corto las bottom_k (neutral al mercado),
      descontando costos de ambos lados y el costo de préstamo de los títulos.
    - equiponderado / benchmark: referencias pasivas.
    """
    if y_step is not None and horizon > step:
        return backtest_overlapping(pred, horizon, top_k, cost_bps, y_step, step, borrow_bps_annual)
    med = "q0.5"
    # sólo días con suficientes acciones (excluye fines de semana donde sólo cotiza cripto)
    counts = pred.groupby(level="date").size()
    dates = list(counts[counts >= 2 * top_k].index)[::horizon]
    borrow = borrow_bps_annual / 1e4 * horizon / 252
    rows = []
    for d in dates:
        g = pred.xs(d, level="date")
        if len(g) < 2 * top_k:
            continue
        ranked = g.sort_values(med)
        longs, shorts = ranked.tail(top_k), ranked.head(top_k)
        r_long = np.expm1(longs["y"]).mean()
        r_short = np.expm1(shorts["y"]).mean()
        rows.append({"date": d,
                     "estrategia": r_long - 2 * cost_bps / 1e4,
                     "long_short": 0.5 * (r_long - r_short) - 2 * cost_bps / 1e4 - 0.5 * borrow,
                     "equiponderado": np.expm1(g["y"]).mean(),
                     "picks": ", ".join(longs.index[::-1]), "shorts": ", ".join(shorts.index)})
    bt = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if bt.empty:
        return bt
    if benchmark is not None:
        b = benchmark.sort_index()
        fwd = b.shift(-horizon) / b - 1
        bt["benchmark"] = fwd.reindex(bt.index, method="ffill").fillna(0).to_numpy()
    for col in STRATS:
        if col in bt:
            bt[f"{col}_equity"] = (1 + bt[col]).cumprod()
    bt.attrs["periods_per_year"] = 252 / horizon
    return bt


def backtest_overlapping(pred: pd.DataFrame, horizon: int, top_k: int, cost_bps: float,
                         y_step: pd.Series, step: int = 5, borrow_bps_annual: float = 50.0) -> pd.DataFrame:
    """Portafolios traslapados (Jegadeesh & Titman 1993).

    Cada `step` días se forma un portafolio (top_k largo, bottom_k corto) que se mantiene
    n = horizon // step periodos. El rendimiento de cada periodo es el promedio de los n portafolios vivos,
    medido con el retorno a `step` días. Sólo 1/n del capital rota por periodo, así que el costo por
    periodo es 1/n del de una rotación completa. Resultado: una serie de `step` días (≈ semanal) en vez de
    una observación cada `horizon` días → ~n veces más observaciones para estimar el Sharpe.
    """
    from collections import deque

    med = "q0.5"
    counts = pred.groupby(level="date").size()
    dates = list(counts[counts >= 2 * top_k].index)[::step]
    n = max(horizon // step, 1)
    ys = y_step.dropna()
    ys_dates = set(ys.index.get_level_values("date"))
    live: deque = deque(maxlen=n)
    borrow = borrow_bps_annual / 1e4 * step / 252
    rows = []
    for d in dates:
        g = pred.xs(d, level="date")
        ranked = g.sort_values(med)
        live.append((list(ranked.index[-top_k:]), list(ranked.index[:top_k])))
        if d not in ys_dates:
            continue
        yd = np.expm1(ys.xs(d, level="date"))
        r_l = np.nanmean([yd.reindex(L).mean() for L, _ in live])
        r_s = np.nanmean([yd.reindex(S).mean() for _, S in live])
        turn = 2 * cost_bps / 1e4 / n
        rows.append({"date": d, "estrategia": r_l - turn,
                     "long_short": 0.5 * (r_l - r_s) - turn - 0.5 * borrow,
                     "equiponderado": yd.reindex(g.index).mean(),
                     "picks": ", ".join(live[-1][0][::-1]), "shorts": ", ".join(live[-1][1])})
    bt = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    for col in STRATS:
        if col in bt:
            bt[f"{col}_equity"] = (1 + bt[col]).cumprod()
    bt.attrs["periods_per_year"] = 252 / step
    return bt


STRATS = ("estrategia", "long_short", "equiponderado", "benchmark")


def summarize_backtest(bt: pd.DataFrame, horizon: int) -> dict:
    periods = bt.attrs.get("periods_per_year", 252 / horizon)
    out = {}
    for col in STRATS:
        if col not in bt:
            continue
        r = bt[col]
        eq = (1 + r).cumprod()
        out[col] = {
            "retorno_total": float(eq.iloc[-1] - 1),
            "cagr": float(eq.iloc[-1] ** (periods / len(r)) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else np.nan,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
        }
    return out


def _zrank(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    r = rankdata(v)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def random_test(pred: pd.DataFrame, horizon: int, top_k: int, cost_bps: float,
                n_perm: int = 300, seed: int = 0, y_step: pd.Series | None = None, step: int = 5,
                borrow_bps_annual: float = 50.0) -> dict:
    """¿El modelo es mejor que el azar?

    Prueba de permutación: dentro de cada fecha se barajan las predicciones entre acciones (un "modelo
    aleatorio" con la misma distribución de pronósticos) y se recalculan IC y Sharpe long-short con la MISMA
    construcción de portafolio que el backtest (traslapada si se pasa `y_step`). p-valor = fracción de
    permutaciones que igualan o superan al modelo real. También prueba binomial del acierto de dirección.

    Vectorizado: las `n_perm` permutaciones se evalúan a la vez como una matriz (fila 0 = modelo real), en vez
    de repetir todo el cálculo en un bucle por permutación (~50x más rápido).
    """
    from scipy.stats import binomtest

    rng = np.random.default_rng(seed)
    p = pred.dropna(subset=["q0.5"])[["q0.5", "y"]]
    p = p[p.groupby(level="date")["q0.5"].transform("size") >= 2 * top_k]
    y = p["y"].to_numpy()
    s0 = p["q0.5"].to_numpy()
    groups = list(p.groupby(level="date").indices.values())
    overlapping = y_step is not None and horizon > step
    every = step if overlapping else horizon
    rebal_ids = set(range(0, len(groups), every))
    n_live = max(horizon // step, 1) if overlapping else 1
    # mismos costos que el backtest: rotación + préstamo de títulos de la pata corta
    cost = 2 * cost_bps / 1e4 / n_live + 0.5 * borrow_bps_annual / 1e4 * every / 252
    ppy = 252 / step if overlapping else 252 / horizon
    P = n_perm + 1

    if overlapping:
        tickers = p.index.get_level_values("ticker").to_numpy()
        rebal_dates = [p.index[groups[i][0]][0] for i in sorted(rebal_ids)]
        ymat = np.expm1(y_step.dropna()).unstack("ticker").reindex(index=rebal_dates)
        col = {t: i for i, t in enumerate(ymat.columns)}
        ymat_v = np.column_stack([ymat.to_numpy(), np.full(len(ymat), np.nan)])  # col extra = sin dato
        miss = ymat_v.shape[1] - 1
        R = len(rebal_dates)
        acc = {leg: (np.zeros((R, P)), np.zeros((R, P))) for leg in ("L", "S")}
    ls_cols = []
    ic_sum, ic_n = np.zeros(P), 0
    ri = -1
    for gi, g in enumerate(groups):
        m = len(g)
        perm = np.argsort(rng.random((P, m)), axis=1)
        perm[0] = np.arange(m)                                   # fila 0 = predicciones reales
        zs, zy = _zrank(s0[g]), _zrank(y[g])
        if zs.any() and zy.any():
            ic_sum += (zs[perm] * zy).mean(axis=1)
            ic_n += 1
        if gi not in rebal_ids:
            continue
        ri += 1
        order0 = np.argsort(s0[g], kind="stable")
        is_top = np.isin(perm, order0[-top_k:])                  # posiciones que el modelo "barajado" compra
        is_bot = np.isin(perm, order0[:top_k])
        if not overlapping:
            ey = np.expm1(y[g])
            ls_cols.append(0.5 * ((is_top * ey).sum(1) - (is_bot * ey).sum(1)) / top_k - cost)
            continue
        codes = np.array([col.get(t, miss) for t in tickers[g]])
        for leg, mask in (("L", is_top), ("S", is_bot)):
            pos = np.argsort(~mask, axis=1, kind="stable")[:, :top_k]
            sel = codes[pos]                                    # P x top_k tickers elegidos
            for j in range(n_live):                             # el portafolio vive n_live periodos
                t = ri + j
                if t >= R:
                    break
                v = np.nanmean(ymat_v[t][sel], axis=1)
                ok = np.isfinite(v)
                acc[leg][0][t, ok] += v[ok]
                acc[leg][1][t, ok] += 1
    if overlapping:
        with np.errstate(invalid="ignore", divide="ignore"):
            r_l = acc["L"][0] / acc["L"][1]
            r_s = acc["S"][0] / acc["S"][1]
        ls = 0.5 * (r_l - r_s) - cost
        ls = ls[np.isfinite(ls).all(axis=1)]
    else:
        ls = np.array(ls_cols)
    if len(ls) > 1:
        sd = ls.std(axis=0)
        sharpe = np.where(sd > 0, ls.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(ppy), 0.0)
    else:
        sharpe = np.zeros(P)
    ic = ic_sum / ic_n if ic_n else np.full(P, np.nan)
    real_ic, real_sh = float(ic[0]), float(sharpe[0])
    perm_ic, perm_sh = ic[1:], sharpe[1:]
    nz = p["y"] != 0
    hits = int((np.sign(p.loc[nz, "q0.5"]) == np.sign(p.loc[nz, "y"])).sum())
    return {
        "ic": real_ic, "ic_random_p95": float(np.nanpercentile(perm_ic, 95)),
        "p_value_ic": float((np.sum(perm_ic >= real_ic) + 1) / (n_perm + 1)),
        "sharpe_ls": real_sh, "sharpe_random_p95": float(np.nanpercentile(perm_sh, 95)),
        "p_value_sharpe": float((np.sum(perm_sh >= real_sh) + 1) / (n_perm + 1)),
        "hit_rate": hits / int(nz.sum()),
        # H0: el acierto no supera al de "siempre sube" (no a 50%, que sería engañoso)
        "p_value_hit": float(binomtest(hits, int(nz.sum()), max(0.5, float((p.loc[nz, "y"] > 0).mean())),
                                       alternative="greater").pvalue),
    }


# ---------------------------------------------------------------------------
# Corrección por haber probado muchas configuraciones
# ---------------------------------------------------------------------------

def _fold_ic(pred: pd.DataFrame) -> pd.Series:
    """IC promedio de cada fold de validación."""
    def ic(g):
        d = g.groupby(level="date").apply(
            lambda x: x["q0.5"].corr(x["y"], method="spearman") if x["q0.5"].nunique() > 1 else np.nan)
        return d.mean()
    return pred.groupby("fold").apply(ic)


def nested_selection(oos_by_engine: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict], str]:
    """Selección anidada (nested walk-forward) del modelo.

    En cada fold k se elige el motor con mejor IC promedio en los folds ANTERIORES y se usan sus
    predicciones del fold k. El primer fold no tiene historia para elegir y se excluye. El resultado mide
    el desempeño del *procedimiento de selección*, sin el sesgo de elegir "el mejor" mirando todo.
    Devuelve (predicciones concatenadas, elecciones por fold, motor elegido con todos los folds).
    """
    scores = pd.DataFrame({e: _fold_ic(p) for e, p in oos_by_engine.items()}).sort_index()
    folds = list(scores.index)
    parts, choices = [], []
    for k in range(1, len(folds)):
        best = scores.iloc[:k].mean().idxmax()
        p = oos_by_engine[best]
        parts.append(p[p["fold"] == folds[k]])
        choices.append({"fold": folds[k], "engine": best})
    final = scores.mean().idxmax()
    return (pd.concat(parts) if parts else pd.DataFrame()), choices, final


def deflated_sharpe(returns: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probabilidad de que el Sharpe verdadero sea > 0 descontando que se eligió entre N pruebas.
    `returns`: retornos por periodo de la estrategia evaluada. `trial_sharpes`: Sharpe por periodo de
    todas las configuraciones probadas (incluida ésta). 0.95 = 95% de confianza tras la corrección.
    """
    from scipy.stats import kurtosis, norm, skew

    r = pd.Series(returns).dropna()
    t = len(r)
    if t < 5 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    trials = np.asarray([x for x in trial_sharpes if np.isfinite(x)])
    n = max(len(trials), 1)
    euler = 0.5772156649
    sr0 = 0.0 if n == 1 else np.sqrt(trials.var(ddof=1)) * (
        (1 - euler) * norm.ppf(1 - 1 / n) + euler * norm.ppf(1 - 1 / (n * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
