"""Línea de comandos.

  stockcast run --market us            # descarga, valida, entrena y pronostica
  stockcast run --market all --no-eval # sólo reentrena y pronostica (más rápido)
  stockcast predict --market mx        # inferencia con modelos guardados, sin reentrenar
  stockcast news --market us           # actualiza el histórico de noticias
  stockcast data                       # sólo refresca precios, bancos centrales y FRED
  stockcast compare --market us        # compara modelos fuera de muestra y contra el azar
  stockcast orders --market us --capital 10000           # CSV de compras/ventas en corto
  stockcast orders --market us --capital 10000 --submit  # envía a Alpaca (paper)
  stockcast app                        # abre el dashboard de Streamlit
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from stockcast import config
from stockcast.models import ENGINES


def _markets(arg: str) -> list[str]:
    """'all' = mercados núcleo + candidatos que pasaron la validación (gate)."""
    if arg != "all":
        return [arg]
    from stockcast import gate
    return gate.active_markets()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stockcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="pipeline completo")
    r.add_argument("--market", default="all")
    r.add_argument("--engine", choices=ENGINES, default=None)
    r.add_argument("--no-eval", action="store_true", help="omite la validación walk-forward")
    r.add_argument("--offline", action="store_true", help="usa sólo la caché local")
    r.add_argument("--no-news", action="store_true")

    pr = sub.add_parser("predict", help="pronóstico con modelos guardados")
    pr.add_argument("--market", default="all")
    pr.add_argument("--offline", action="store_true")

    n = sub.add_parser("news", help="actualiza noticias y sentimiento")
    n.add_argument("--market", default="all")

    c = sub.add_parser("compare", help="compara modelos fuera de muestra y contra el azar")
    c.add_argument("--market", default="all")
    c.add_argument("--engines", nargs="*", choices=ENGINES, default=None)
    c.add_argument("--refresh", action="store_true", help="descarga datos antes (por defecto usa caché)")

    o = sub.add_parser("orders", help="genera (y opcionalmente envía) órdenes long/short")
    o.add_argument("--market", default="us")
    o.add_argument("--capital", type=float, default=10_000)
    o.add_argument("--horizon", type=int, default=5)
    o.add_argument("--top-k", type=int, default=None)
    o.add_argument("--no-short", action="store_true", help="sólo compras")
    o.add_argument("--submit", action="store_true", help="envía a Alpaca (paper por defecto)")

    sub.add_parser("report", help="escribe docs/RESULTS.md con la comparación de modelos")
    sub.add_parser("data", help="refresca precios, bancos centrales y FRED")
    sub.add_parser("app", help="abre el dashboard")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.cmd == "run":
        from stockcast import pipeline
        for mk in _markets(a.market):
            pipeline.run(mk, refresh=not a.offline, engine=a.engine,
                         evaluate=not a.no_eval, with_news=not a.no_news)
    elif a.cmd == "predict":
        from stockcast import pipeline
        for mk in _markets(a.market):
            print(pipeline.predict(mk, refresh=not a.offline).to_string(index=False))
    elif a.cmd == "compare":
        from stockcast import compare
        cols = ["horizon", "engine", "hit_rate", "hit_always_up", "p_value_hit", "ic", "p_value_ic",
                "sharpe_long_short", "p_value_sharpe", "sharpe_long", "sharpe_equal_weight", "coverage", "seconds"]
        for mk in (config.markets() if a.market == "all" else [a.market]):
            df = compare.run(mk, refresh=a.refresh, engines=a.engines)
            print(f"\n=== {mk} ===")
            print(df[cols].round(3).to_string(index=False))
    elif a.cmd == "orders":
        from stockcast import signals
        top_k = a.top_k or config.load()["backtest"]["top_k"]
        df = signals.orders_cli(a.market, a.capital, a.horizon, top_k, a.no_short, a.submit)
        print(df.to_string(index=False))
    elif a.cmd == "report":
        from stockcast import compare
        print(compare.report())
    elif a.cmd == "news":
        from stockcast.data import news
        for mk in _markets(a.market):
            df = news.update_news_history(config.market(mk).tickers, mk)
            print(f"{mk}: {len(df)} noticias en el histórico")
    elif a.cmd == "data":
        from stockcast.data import central_banks, fred, market
        tickers = sorted({t for mk in config.markets() for t in
                          config.market(mk).tickers + config.market(mk).macro + [config.market(mk).benchmark]})
        market.load_prices(tickers)
        central_banks.load_policy_rates()
        fred.load_fred()
        print("Datos actualizados en", config.path("data"))
    elif a.cmd == "app":
        app = config.ROOT / "app" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
