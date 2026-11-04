# stockcast: pronóstico de acciones con bancos centrales, noticias y ML, 100% local

Pronóstico a 1 día, 1 semana, 2 semanas y 1 mes con bandas de confianza.

**Mercados núcleo** (siempre activos):

- **EE. UU.**: 11 acciones grandes y BTC.
- **México (BMV)**: 26 emisoras del IPC.

**Mercados candidatos** (se activan solo si la validación demuestra que la predicción gana dinero mejor que el azar):

- **Alemania (DAX)**, **Japón (Nikkei)** y **Reino Unido (FTSE 100)**, con 30 acciones líquidas cada uno.

Genera señales de **COMPRA / VENTA EN CORTO** únicamente en los horizontes validados. Incluye un dashboard en Streamlit y órdenes opcionales para un broker con API. Funciona en una laptop: no requiere nube ni API keys de pago.

> Proyecto educativo y de investigación. No es asesoría de inversión.

## Inicio rápido

```bash
make install                      # crea .venv (Python 3.12) e instala dependencias
.venv/bin/stockcast run --market us   # descarga datos, valida, entrena y pronostica (~1 min)
make app                          # dashboard en http://localhost:8501
make schedule                     # (opcional) diario: solo mercados activos; lunes: evaluación por etapas
```

Sin `make`: `uv venv --python 3.12 .venv && uv pip install -e ".[dev]"` (o `python -m venv .venv && pip install -e .`).

| Comando | Qué hace |
|---|---|
| `stockcast run --market us\|mx\|de\|jp\|uk\|all` | Pipeline completo. `all` = núcleo + candidatos validados. `--no-eval` omite la validación (más rápido). |
| `stockcast evaluate --market all` | **Evaluación por etapas** (la corrida de los lunes). Prueba rápida (Ridge, segundos) → evaluación completa solo si pasa → ablación → registro de experimentos. Lo que no funciona queda descartado y no se recalcula; se re-prueba cada 180 días. |
| `stockcast compare --market all` | Compara modelos fuera de muestra y contra predicciones aleatorias, y decide qué mercados y horizontes se activan. |
| `stockcast ablation --market all` | Mide cuánto aporta cada grupo de variables y descarta los irrelevantes. |
| `stockcast report` | Escribe `docs/RESULTS.md` (comparación) y `docs/VARIABLES.md` (diccionario de variables). |
| `stockcast predict --market mx` | Pronostica con los modelos guardados, sin reentrenar. |
| `stockcast orders --market us --capital 10000` | Genera un CSV con compras y ventas en corto. `--submit` lo envía a Alpaca (*paper* por defecto). |
| `stockcast news` / `stockcast data` | Actualiza solo noticias o solo datos. |

## Dashboard

| Pestaña | Contenido |
|---|---|
| **Pronósticos** | Ranking con señal, P(sube), mediana a cada horizonte y rango p10–p90. |
| **Acción** | Velas, SMA 50/200, abanico de pronóstico, RSI, noticias con sentimiento y ratios de *moat*. |
| **Entorno económico** | Tasas de ~40 bancos centrales (BIS), ciclo global de recortes/alzas e indicadores de estrés (FRED), cada uno con su interpretación. |
| **Noticias** | Sentimiento por acción (Yahoo Finance + Google News). |
| **Validación** | Métricas fuera de muestra, backtest long y long-short, comparación de modelos con p-valores e importancia de variables. |

## Fuentes de datos (todas gratuitas, sin API key)

| Dato | Fuente |
|---|---|
| Precios OHLCV, índices, VIX, divisas, materias primas | Yahoo Finance (`yfinance`) |
| Tasas de política de ~40 bancos centrales | BIS, dataset WS_CBPOL |
| Incertidumbre de política (EPU), spread high-yield, curva 10a-2a, tasa real | FRED (CSV público) |
| Noticias | Yahoo Finance + Google News RSS (inglés y español) |
| Componentes de índices (universo global) | Wikipedia |
| Estados financieros (moat) | Yahoo Finance |

## Cómo funciona

```
data/ (caché parquet)                     artifacts/<mercado>/
 precios ─┐                               ├ forecasts.parquet      ← dashboard
 BIS ─────┼─► features.py ─► modelo ─────►├ metrics.json / comparison.json
 FRED ────┤   (panel fecha×acción,  global ├ backtest.parquet
 noticias ┘    ~85–130 variables)   por    ├ model_h{1,5,10,21}.joblib
                                   horizonte└ forecast_history.parquet (para medir precisión real)
```

1. **Features**, todas calculadas con información disponible al cierre del día:
   - Momentum de 1 a 126 días y momentum relativo al universo.
   - Volatilidad, Bollinger, RSI, MACD, estocástico, ATR, volumen y distancia al máximo de 52 semanas.
   - Macro: índices, VIX, tasa a 10 años, dólar, petróleo, oro y Bovespa.
   - Bancos centrales: cambio de tasa de cada banco, *breadth* global y diferenciales MX–US y EUR–US.
   - Estrés FRED.
2. **Objetivo**: el retorno logarítmico de t a t+h, con pronóstico **directo** para cada horizonte.
3. **Modelo**: LightGBM cuantil (p10, p50, p90), uno global para todas las acciones.
4. **Validación**: walk-forward con purga, comparación contra la caminata aleatoria y contra un modelo aleatorio (permutación), y bandas calibradas con conformal.
5. **¿Vale la pena predecir?** (`gate.py`): un horizonte se valida solo si se cumplen tres condiciones fuera de muestra:
   - ordena acciones mejor que el azar (p < 0.05);
   - la estrategia long-short logra un Sharpe ≥ 0.3;
   - esa ganancia también le gana al azar (p < 0.10).

   Para cada horizonte se usa el modelo barato (Ridge, LightGBM o ensamble) que mejor salió. Si nada pasa, no hay señal.
6. **Señal**: en los horizontes validados, el 20% superior del ranking recibe COMPRA y el 20% inferior VENTA EN CORTO.

Qué significa cada variable: [`docs/VARIABLES.md`](docs/VARIABLES.md), también en el dashboard (pestaña *Validación*, "Diccionario de variables"). Los papers que respaldan cada decisión están en [`docs/PAPERS.md`](docs/PAPERS.md). El plan para llevarlo a la nube casi gratis está en [`docs/ROADMAP.md`](docs/ROADMAP.md), y los archivos listos (inactivos) en [`deploy/`](deploy/README.md).

## Resultados (validación fuera de muestra, 2000–2026)

Tabla completa en [`docs/RESULTS.md`](docs/RESULTS.md). Evaluación rigurosa:
- 26 años de historia;
- selección anidada del modelo;
- portafolios traslapados;
- costos por mercado (BMV 25 pb);
- Deflated Sharpe, que corrige por todas las configuraciones probadas.

| Mercado | Mejor plazo | IC | Sharpe long-short (anidado) | Deflated Sharpe | Estado |
|---|---|---|---|---|---|
| EE. UU. (12 originales) | 1 mes | 0.053 | 0.41 | 65% | ✅ activo · 🟡 evidencia moderada |
| Reino Unido (FTSE) | 1 mes | 0.040 | 0.40 (p = 0.013) | 51% | ✅ activo · 🟡 evidencia moderada |
| México (BMV) | 2 semanas | 0.065 | 0.29 | 24% | ⏸️ en pausa: ordena bien, pero no gana con costos |
| Sectores (ETFs SPDR) | 1 mes | 0.057 | 0.24 | 27% | 🗑️ descartado |
| EE. UU. dinámico (top 50 líquidas) | 1 mes | 0.035 | 0.07 | 25% | 🗑️ descartado |
| Japón | 1 mes | 0.013 | 0.20 | 18% | 🗑️ descartado |
| Alemania | – | < 0 | < 0 | ≈ 0% | 🗑️ descartado (también en la prueba rápida) |

Qué significa:
- **Ordenar acciones mejor que el azar es común** (IC significativo en casi todos los mercados). **Ganar dinero con eso, después de costos y de corregir por pruebas múltiples, es raro**: solo EE. UU. y Reino Unido a 1 mes, y con evidencia moderada.
- La señal vive en el **plazo mensual**. El plazo de 1 día se retiró porque perdió en los 6 mercados.
- Con 26 años, el universo dinámico de EE. UU. es mucho más débil que las 12 originales. Parte de la ventaja de las 12 puede venir de **sesgo de supervivencia**: son las ganadoras de hoy.
- Con 26 años, las tasas de bancos centrales **dejaron de aportar** en EE. UU. Su peso con 8 años era propio de 2018–2026.

Contra el método original (LSTM por acción, replicado y evaluado con la misma validación antes de retirarlo):

| Modelo | EE. UU. 5 d IC (p) | EE. UU. 5 d Sharpe long-short | México 21 d IC | Costo |
|---|---|---|---|---|
| LSTM original | 0.007 (0.29, **igual que el azar**) | −0.08 | 0.013 | ~50–80 s, GPU |
| LSTM mejorado | −0.001 (0.59) | −0.71 | 0.060 | ~50–100 s, GPU |
| **ensemble (LightGBM + Ridge)** | **0.062 (0.003)** | **1.26** | 0.050 | ~4–8 s, CPU |

### ¿Qué variables pesan? (`stockcast ablation`)

Se reentrena quitando cada grupo de variables. Si la predicción no empeora, el grupo se descarta automáticamente en ese mercado (`features.auto_drop`).

| Grupo | EE. UU. | México | Se usa en el modelo |
|---|---|---|---|
| Bancos centrales (40, BIS) | **Clave**: sin ellos, el IC a 5 d cae de 0.062 a 0.023 y el Sharpe long-short de 1.26 a −0.01 | sin aporte; quitarlos **mejoró** México (de 3 a 4 horizontes validados) | EE. UU. sí · México no |
| Macro (índices, divisa, VIX, petróleo…) | ≈ 0 en IC, + en Sharpe | **clave** para el Sharpe (+0.35) | sí |
| FRED (estrés, incertidumbre) | + (Sharpe +0.37) | + leve | sí |
| Noticias | sin histórico todavía | 0 | no (contexto en el dashboard; se acumulan a diario) |
| Moat (fundamentales) | no medible (sin histórico point-in-time) | – | no (contexto en la ficha de cada acción) |

Usar los 40 bancos centrales es mejor que usar solo los 12 principales: en EE. UU. a 21 días, IC 0.097 contra 0.068.

Advertencias honestas:
- **Ningún modelo predice la dirección (¿sube o baja?) mejor que "siempre sube"** en este periodo alcista. Lo que sí funciona es **ordenar** acciones: comprar las de arriba y vender en corto las de abajo.
- La señal VENTA EN CORTO es *relativa*: pensada en pareja con las compras (neutral al mercado). Sola, en un mercado alcista, puede perder.
- P(sube), que va de 40% a 66%, **no está mejor calibrada que la frecuencia histórica de subidas** (Brier del modelo ≥ Brier base). No sirve para elegir "las que seguro suben". Lo que sí está validado es el orden relativo.
- Los resultados son de un periodo y universo concretos (12 y 26 acciones). La precisión real se irá midiendo con `forecast_history.parquet`.

## Estructura

```
config/settings.yaml       mercados, horizontes, bancos centrales, hiperparámetros
src/stockcast/
  data/                    market, central_banks, fred, cb_tone, news, fundamentals, universe (listas de índices)
  features.py              ingeniería de features (panel)
  models/                  naive, momentum, ridge, lightgbm, ensemble
  evaluation.py            walk-forward, métricas, backtest, prueba vs. azar, conformal
  pipeline.py · compare.py · gate.py · signals.py · cli.py
app/streamlit_app.py       dashboard
scripts/                   run_daily.sh + timer de systemd / cron
notebooks/                 notebooks originales con LSTM (referencia histórica; el LSTM se retiró del pipeline)
tests/                     pruebas sin red (pytest)
```

## Notebooks originales

La versión anterior, con un LSTM por acción y optimización con Optuna, sigue en `notebooks/`:
[US](https://github.com/nachovazquez98/Portafolio-Acciones-LSTMs/blob/main/notebooks/multi-rnn-lstm-us-stocks.ipynb) ·
[MX](https://github.com/nachovazquez98/Portafolio-Acciones-LSTMs/blob/main/notebooks/multi-rnn-lstm-mx-stocks.ipynb).
