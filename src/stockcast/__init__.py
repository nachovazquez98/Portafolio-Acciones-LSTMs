"""stockcast: pronóstico local de acciones con datos macro, bancos centrales y noticias."""
__version__ = "0.2.0"

import warnings as _w

# yfinance todavía usa `inplace=` (deprecado en pandas 3.x) y llena el log; no afecta resultados.
_w.filterwarnings("ignore", message=".*inplace keyword.*")
_w.filterwarnings("ignore", category=FutureWarning, module="yfinance")
_w.filterwarnings("ignore", message="An input array is constant")
