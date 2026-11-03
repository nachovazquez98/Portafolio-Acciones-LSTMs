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
        # contexto previo para modelos secuenciales (LSTM)
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
             benchmark: pd.Series | None = None, borrow_bps_annual: float = 50.0) -> pd.DataFrame:
    """Cada `horizon` días rebalancea según el retorno mediano pronosticado.

    - estrategia: compra (long) las top_k acciones.
    - long_short: compra top_k y vende en corto las bottom_k (neutral al mercado),
      descontando costos de ambos lados y el costo de préstamo de los títulos.
    - equiponderado / benchmark: referencias pasivas.
    """
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
    return bt


STRATS = ("estrategia", "long_short", "equiponderado", "benchmark")


def summarize_backtest(bt: pd.DataFrame, horizon: int) -> dict:
    periods = 252 / horizon
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


def random_test(pred: pd.DataFrame, horizon: int, top_k: int, cost_bps: float,
                n_perm: int = 300, seed: int = 0) -> dict:
    """¿El modelo es mejor que el azar?

    Prueba de permutación: dentro de cada fecha se barajan las predicciones entre
    acciones (un "modelo aleatorio" con la misma distribución de pronósticos) y se
    recalculan IC y Sharpe long-short. p-valor = fracción de permutaciones que
    igualan o superan al modelo real. También prueba binomial del acierto de dirección.
    """
    from scipy.stats import binomtest

    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    p = pred.dropna(subset=["q0.5"])[["q0.5", "y"]]
    p = p[p.groupby(level="date")["q0.5"].transform("size") >= 2 * top_k]
    y = p["y"].to_numpy()
    groups = list(p.groupby(level="date").indices.values())
    rebal = groups[::horizon]
    ranks_y = [rankdata(y[g]) for g in groups]
    cost = 2 * cost_bps / 1e4

    def stats(score: np.ndarray) -> tuple[float, float]:
        ics = []
        for g, ry in zip(groups, ranks_y):
            rs = rankdata(score[g])
            if rs.std() > 0 and ry.std() > 0:
                ics.append(np.corrcoef(rs, ry)[0, 1])
        ls = []
        for g in rebal:
            order = g[np.argsort(score[g])]
            ls.append(0.5 * (np.expm1(y[order[-top_k:]]).mean() - np.expm1(y[order[:top_k]]).mean()) - cost)
        ls = np.array(ls)
        sharpe = ls.mean() / ls.std() * np.sqrt(252 / horizon) if ls.std() > 0 else 0.0
        return float(np.mean(ics)) if ics else np.nan, float(sharpe)

    base = p["q0.5"].to_numpy()
    real_ic, real_sh = stats(base)
    perm_ic, perm_sh = [], []
    for _ in range(n_perm):
        shuffled = base.copy()
        for g in groups:
            shuffled[g] = rng.permutation(base[g])
        ic, sh = stats(shuffled)
        perm_ic.append(ic)
        perm_sh.append(sh)
    nz = p["y"] != 0
    hits = int((np.sign(p.loc[nz, "q0.5"]) == np.sign(p.loc[nz, "y"])).sum())
    return {
        "ic": real_ic, "ic_random_p95": float(np.nanpercentile(perm_ic, 95)),
        "p_value_ic": float((np.sum(np.array(perm_ic) >= real_ic) + 1) / (n_perm + 1)),
        "sharpe_ls": real_sh, "sharpe_random_p95": float(np.nanpercentile(perm_sh, 95)),
        "p_value_sharpe": float((np.sum(np.array(perm_sh) >= real_sh) + 1) / (n_perm + 1)),
        "hit_rate": hits / int(nz.sum()),
        # H0: el acierto no supera al de "siempre sube" (no a 50%, que sería engañoso)
        "p_value_hit": float(binomtest(hits, int(nz.sum()), max(0.5, float((p.loc[nz, "y"] > 0).mean())),
                                       alternative="greater").pvalue),
    }
