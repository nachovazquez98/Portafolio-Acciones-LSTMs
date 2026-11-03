"""Pruebas sin red: datos sintéticos para features, validación, modelos y backtest."""
import numpy as np
import pandas as pd
import pytest

from stockcast import evaluation
from stockcast.data.cb_tone import tone_score
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


def test_tone_score():
    hawk, _, _ = tone_score("The Committee decided to raise the target range. Inflation remains elevated.")
    dove, _, _ = tone_score("The Committee decided to lower the target range given downside risks.")
    assert hawk > 0 > dove


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


@pytest.mark.parametrize("engine", ["lstm", "lstm_legacy"])
def test_lstm_engines(panel, engine):
    pytest.importorskip("torch")
    feats = feature_columns(panel) + (["close"] if engine == "lstm_legacy" else [])
    params = {"epochs": 1, "window": 20}
    oos = evaluation.oos_predictions(panel, feats, 5, engine, Q, params, n_splits=2)
    assert oos["q0.5"].notna().mean() > 0.9


def test_gate_requires_significant_profit(tmp_path, monkeypatch):
    import json
    from stockcast import gate, pipeline
    rows = [  # h=5: lightgbm pasa; h=21: nadie pasa (Sharpe bajo o p alto)
        {"horizon": 5, "engine": "lightgbm", "ic": .06, "p_value_ic": .003, "sharpe_long_short": 1.0, "p_value_sharpe": .01},
        {"horizon": 5, "engine": "ridge", "ic": .03, "p_value_ic": .003, "sharpe_long_short": .7, "p_value_sharpe": .02},
        {"horizon": 5, "engine": "lstm", "ic": .09, "p_value_ic": .003, "sharpe_long_short": 2.0, "p_value_sharpe": .01},
        {"horizon": 21, "engine": "ensemble", "ic": .10, "p_value_ic": .003, "sharpe_long_short": .4, "p_value_sharpe": .2},
    ]
    (tmp_path / "comparison.json").write_text(json.dumps(rows))
    monkeypatch.setattr(pipeline, "artifacts_dir", lambda mk: tmp_path)
    d = gate.decide("de")
    assert d["horizons"][5] == {**d["horizons"][5], "engine": "lightgbm", "validated": True}  # lstm no es elegible
    assert not d["horizons"][21]["validated"]
    assert d["active"]
    (tmp_path / "comparison.json").write_text(json.dumps(rows[3:]))
    assert not gate.decide("de")["active"]   # candidato sin horizontes validados -> fuera
    assert gate.decide("us")["active"]       # núcleo siempre activo
