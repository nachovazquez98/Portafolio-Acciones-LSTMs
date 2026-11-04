"""Pruebas sin red: datos sintéticos para features, validación, modelos y backtest."""
import numpy as np
import pandas as pd
import pytest

from stockcast import evaluation
from stockcast.features import build_panel, central_bank_features, feature_columns
from stockcast.models import make_model

Q = [0.1, 0.5, 0.9]


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=900)
    rows = []
    for t in ["AAA", "BBB", "CCC", "DDD"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "open": close, "high": close * 1.01,
                                  "low": close * 0.99, "close": close, "volume": rng.integers(1e5, 1e6, len(dates))}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def panel(prices):
    rates = pd.DataFrame({"US": 2.0, "MX": 7.0}, index=pd.date_range("2017-01-01", "2022-01-01"))
    rates.loc["2019-06-01":, "US"] = 1.5
    p = build_panel(prices, cb_rates=rates, horizons=[1, 5])
    return p[p["ret_126"].notna()]


def test_targets_are_future_returns(prices, panel):
    g = prices[prices.ticker == "AAA"].set_index("date")["close"]
    d = panel.xs("AAA", level="ticker").index[10]
    nxt = g.index[g.index.get_loc(d) + 5]
    assert panel.loc[(d, "AAA"), "y_5"] == pytest.approx(np.log(g[nxt] / g[d]))


def test_features_do_not_use_future(prices):
    """Cambiar precios futuros no debe alterar features pasadas."""
    base = build_panel(prices, horizons=[1])
    cut = prices["date"].sort_values().unique()[600]
    shocked = prices.copy()
    shocked.loc[shocked["date"] > cut, "close"] *= 3
    after = build_panel(shocked, horizons=[1])
    cols = [c for c in feature_columns(base) if c != "y_1"]
    past = base.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(base.loc[past, cols], after.loc[past, cols])


def test_walk_forward_purges_horizon(panel):
    dates = panel.index.get_level_values("date")
    udates = pd.DatetimeIndex(sorted(dates.unique()))
    for train_end, test_start, _ in evaluation.walk_forward_splits(dates, 3, 5):
        assert udates.get_loc(test_start) - udates.get_loc(train_end) > 5


def test_central_bank_breadth():
    rates = pd.DataFrame({"US": [5.0] * 100 + [4.75] * 100, "XM": [4.0] * 200},
                         index=pd.date_range("2020-01-01", periods=200))
    f = central_bank_features(rates, breadth_days=90)
    assert f["cb_breadth"].iloc[150] == pytest.approx(0.5)  # 1 de 2 bancos recortó
    assert f["cb_US_chg90"].iloc[150] == pytest.approx(-25)
    assert "cb_US" not in f  # sin niveles (no estacionarios)


@pytest.mark.parametrize("engine", ["lightgbm", "naive"])
def test_oos_and_backtest(panel, engine):
    feats = feature_columns(panel)
    params = {"n_estimators": 30, "num_leaves": 7} if engine == "lightgbm" else {}
    oos = evaluation.oos_predictions(panel, feats, 5, engine, Q, params, n_splits=2)
    assert {"q0.1", "q0.5", "q0.9", "y"} <= set(oos.columns)
    assert (oos["q0.1"] <= oos["q0.9"]).all()
    m = evaluation.metrics(oos, Q)
    assert 0 <= m["coverage"] <= 1
    k = evaluation.conformal_scale(oos, Q)
    assert 0.5 <= k <= 4
    bt = evaluation.backtest(oos, 5, top_k=2, cost_bps=10)
    assert "estrategia_equity" in bt


def test_quantile_model_predicts_all_rows(panel):
    feats = feature_columns(panel)
    train = panel.dropna(subset=["y_1"])
    m = make_model("lightgbm", Q, {"n_estimators": 20}).fit(train[feats], train["y_1"])
    pred = m.predict(panel[feats])
    assert len(pred) == len(panel) and not pred.isna().any().any()


@pytest.mark.parametrize("engine", ["momentum", "ridge"])
def test_simple_models(panel, engine):
    feats = feature_columns(panel)
    train = panel.dropna(subset=["y_5"])
    pred = make_model(engine, Q).fit(train[feats], train["y_5"], 5).predict(panel[feats])
    assert (pred["q0.1"] <= pred["q0.5"]).all() and (pred["q0.5"] <= pred["q0.9"]).all()


def test_signals_rank_and_prob():
    from stockcast.signals import add_signals
    fc = pd.DataFrame({"ticker": list("ABCDEFGHIJ"), "horizon": 5,
                       "ret_q0.5": np.linspace(-0.05, 0.05, 10)})
    fc["ret_q0.1"], fc["ret_q0.9"] = fc["ret_q0.5"] - 0.04, fc["ret_q0.5"] + 0.04
    out = add_signals(fc, Q).set_index("ticker")
    assert out.loc["J", "signal"] == "COMPRA" and out.loc["A", "signal"] == "VENTA EN CORTO"
    assert out.loc["J", "prob_up"] > 0.5 > out.loc["A", "prob_up"]


def test_random_test_detects_signal(panel):
    """Un 'modelo' que conoce el futuro debe tener p-valor bajo; uno aleatorio, no."""
    df = panel.dropna(subset=["y_5"])[["y_5"]].rename(columns={"y_5": "y"})
    oracle = df.assign(**{"q0.5": df["y"]})
    rng = np.random.default_rng(1)
    noise = df.assign(**{"q0.5": rng.normal(size=len(df))})
    assert evaluation.random_test(oracle, 5, 1, 0, n_perm=50)["p_value_ic"] < 0.05
    assert evaluation.random_test(noise, 5, 1, 0, n_perm=50)["p_value_ic"] > 0.05


def test_gate_requires_significant_profit(tmp_path, monkeypatch):
    import json
    from stockcast import gate, pipeline
    rows = [  # h=5: lightgbm pasa; h=21: nadie pasa (Sharpe bajo o p alto)
        {"horizon": 5, "engine": "lightgbm", "ic": .06, "p_value_ic": .003, "sharpe_long_short": 1.0, "p_value_sharpe": .01},
        {"horizon": 5, "engine": "ridge", "ic": .03, "p_value_ic": .003, "sharpe_long_short": .7, "p_value_sharpe": .02},
        {"horizon": 5, "engine": "momentum", "ic": .09, "p_value_ic": .003, "sharpe_long_short": 2.0, "p_value_sharpe": .01},
        {"horizon": 21, "engine": "ensemble", "ic": .10, "p_value_ic": .003, "sharpe_long_short": .4, "p_value_sharpe": .2},
    ]
    (tmp_path / "comparison.json").write_text(json.dumps(rows))
    monkeypatch.setattr(pipeline, "artifacts_dir", lambda mk: tmp_path)
    d = gate.decide("de")
    assert d["horizons"][5] == {**d["horizons"][5], "engine": "lightgbm", "validated": True}  # momentum no es elegible
    assert not d["horizons"][21]["validated"]
    assert d["active"]
    (tmp_path / "comparison.json").write_text(json.dumps(rows[3:]))
    assert not gate.decide("de")["active"]   # candidato sin horizontes validados -> fuera
    assert gate.decide("us")["active"]       # núcleo siempre activo


def test_ablation_contribution_sign():
    from stockcast.ablation import contribution, group_of
    assert group_of("cb_US_chg90") == "bancos_centrales" and group_of("m_VIX") == "macro"
    assert group_of("f_T10Y2Y_chg30") == "fred" and group_of("rsi_14") == "tecnicos"
    df = pd.DataFrame({"horizon": [5, 5, 5], "variant": ["todas", "sin_macro", "sin_fred"],
                       "ic": [.06, .02, .07], "sharpe_long_short": [1.0, .2, 1.1]})
    df["delta_ic"] = df["ic"] - .06
    df["delta_sharpe"] = df["sharpe_long_short"] - 1.0
    c = contribution(df)
    assert c.loc["macro", "aporte_ic"] > 0 and c.loc["fred", "aporte_ic"] < 0  # fred no aporta


def test_glossary_describes_every_feature(panel):
    from stockcast import glossary
    t = glossary.table(feature_columns(panel))
    assert (t["descripcion"] != "").all(), t[t["descripcion"] == ""]["variable"].tolist()
    assert glossary.label("cb_MX_chg90").startswith("Banco de México")
    assert glossary.label("m_DXYNYB_ret21") == "Índice del dólar (DXY): retorno 1 mes"


def test_point_in_time_universe_uses_only_past(prices):
    from stockcast.pipeline import point_in_time_universe
    p = prices.copy()
    big = p["ticker"] == "AAA"
    p.loc[big, "volume"] *= 100                      # AAA siempre la más líquida
    late = p["date"] > p["date"].sort_values().unique()[700]
    p.loc[late & (p["ticker"] == "BBB"), "volume"] *= 10_000   # BBB se vuelve líquida sólo al final
    mem = point_in_time_universe(p, top_n=1, window=20)
    d_early = p["date"].sort_values().unique()[400]
    d_late = p["date"].sort_values().unique()[-1]
    assert (d_early, "AAA") in mem and (d_early, "BBB") not in mem   # antes no "sabe" que BBB será líquida
    assert (d_late, "BBB") in mem


def test_nested_selection_uses_only_previous_folds():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2024-01-01", periods=40), list("ABCDE")], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    y = rng.normal(size=len(idx))
    folds = np.repeat(["f1", "f2", "f3", "f4"], len(idx) // 4)
    good = pd.DataFrame({"q0.5": y, "y": y, "fold": folds}, index=idx)          # ordena perfecto
    bad = pd.DataFrame({"q0.5": rng.normal(size=len(idx)), "y": y, "fold": folds}, index=idx)
    nested, choices, final = evaluation.nested_selection({"good": good, "bad": bad})
    assert final == "good" and all(c["engine"] == "good" for c in choices)
    assert set(nested["fold"]) == {"f2", "f3", "f4"}                           # el primer fold no se usa


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.003, 0.02, 200))                                 # Sharpe moderado
    alone = evaluation.deflated_sharpe(r, np.array([r.mean() / r.std()]))
    many = evaluation.deflated_sharpe(r, rng.normal(0, 0.1, 50))               # elegido entre 50 pruebas
    assert alone > many


def test_evidence_levels_control_signals():
    from stockcast.signals import add_signals
    fc = pd.DataFrame({"ticker": list("ABCDEFGHIJ") * 2, "horizon": [5] * 10 + [21] * 10,
                       "ret_q0.5": np.tile(np.linspace(-0.05, 0.05, 10), 2),
                       "evidence": ["moderada"] * 10 + ["ninguna"] * 10})
    fc["ret_q0.1"], fc["ret_q0.9"] = fc["ret_q0.5"] - 0.04, fc["ret_q0.5"] + 0.04
    out = add_signals(fc, Q)
    assert set(out.loc[out["horizon"] == 5, "signal"]) >= {"COMPRA", "VENTA EN CORTO"}
    assert set(out.loc[out["horizon"] == 21, "signal"]) == {"SIN SEÑAL"}


def test_overlapping_backtest_gives_more_observations(panel):
    feats = feature_columns(panel)
    oos = evaluation.oos_predictions(panel, feats, 5, "ridge", Q, {}, n_splits=2)
    oos21 = oos.rename(columns={})  # mismas predicciones, evaluadas como plazo 21
    plain = evaluation.backtest(oos21, 21, 1, 10)
    over = evaluation.backtest(oos21, 21, 1, 10, y_step=panel["y_5"])
    assert len(over) > 3 * len(plain)
    assert over.attrs["periods_per_year"] == pytest.approx(252 / 5)
    rnd = evaluation.random_test(oos21, 21, 1, 10, n_perm=20, y_step=panel["y_5"])
    assert 0 < rnd["p_value_sharpe"] <= 1


def test_dsr_uses_evaluated_frequency():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.004, 0.02, 400))
    ann = np.array([1.0, 0.5, 0.2, -0.3])
    weekly = evaluation.deflated_sharpe(r, ann / np.sqrt(52))
    wrong = evaluation.deflated_sharpe(r, ann / np.sqrt(np.array([252, 52, 25, 12])))  # mezcla de frecuencias
    assert weekly != wrong and 0 <= weekly <= 1


def test_experiment_registry_skips_recent_discards(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    from stockcast import experiments
    monkeypatch.setattr(experiments, "_path", lambda: tmp_path / "experiments.json")
    assert experiments.due_for_evaluation("de")                       # nunca evaluado → toca
    experiments.record("de", "descartado", "sin señal", "prueba rápida")
    assert not experiments.due_for_evaluation("de")                   # recién descartado → no se recalcula
    reg = experiments.load()
    reg["de"]["updated"] = (datetime.now() - timedelta(days=400)).isoformat()
    experiments.save(reg)
    assert experiments.due_for_evaluation("de")                       # pasó el plazo → se re-prueba
    experiments.record("mx", "pausa", "sin ganancia", "evaluación completa")
    assert experiments.due_for_evaluation("mx")                       # en pausa se sigue evaluando
