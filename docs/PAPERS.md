# Papers usados y cómo influyeron en el código

Revisión hecha en noviembre de 2026. Para cada paper se indica qué se tomó y en qué archivo está.

## 1. Qué modelo usar (más barato, misma o mejor precisión)

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies 33(5) — [NBER w25398](https://www.nber.org/papers/w25398) | Árboles y redes son los mejores, y la ganancia viene de interacciones no lineales. Las señales dominantes son momentum, liquidez y volatilidad. | Modelo **global** de LightGBM (un modelo para todas las acciones) con features de momentum, volatilidad y volumen. `models/gbm.py`, `features.py` |
| [LGBM on Stock Returns Prediction and Portfolio Construction (2025)](https://scitepress.org/PublishedPapers/2025/137013) | LightGBM supera a RF, XGBoost y LSTM en retornos diarios. | LightGBM queda como motor por defecto. Se confirmó con nuestros datos: el LSTM (original y mejorado) no superó a LightGBM/Ridge y **se retiró** del pipeline (`docs/RESULTS.md`). |
| [Gradient Boosting Decision Tree with LSTM for Investment Prediction (2025)](https://arxiv.org/abs/2505.23084) | Combinar LSTM con GBDT mejora sobre cada modelo por separado. | Se tomó la idea de combinar modelos distintos, pero con uno barato: `ensemble` = LightGBM + Ridge. `models/linear.py` |
| Zeng et al. (2023), [*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504) (DLinear) | Un modelo lineal simple le gana a Transformers complejos. La ventaja de esos modelos venía del **pronóstico directo multi-horizonte**, no de la arquitectura. | Se pasó del pronóstico recursivo del notebook (30 pasos alimentando su propia predicción) a pronóstico **directo** por horizonte. Se agregó `ridge` como baseline lineal. |
| Noguer i Alonso & Pereira Franklin (2026), [*Pretrained Time-Series Foundation Models for Financial Return Forecasting*](https://arxiv.org/abs/2606.27100) | Chronos, TimesFM y Moirai ganan pocas veces, y las ganancias frente a la caminata aleatoria son "pequeñas y escasas". | No se agregaron (son pesados y aportan poco). Todo modelo se compara contra la caminata aleatoria. `evaluation.py` |
| Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* (NeurIPS) | Intervalos con cobertura garantizada a partir de residuos fuera de muestra. | Las bandas p10–p90 se calibran con un factor conformal. `evaluation.conformal_scale` |
| Timmermann (2006), *Forecast Combinations*, Handbook of Economic Forecasting | Promediar modelos distintos reduce la varianza del error. | Motor `ensemble`. |

## 1b. Cómo elegir universo y modelo sin "fuerza bruta"

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Harvey, Liu y Zhu (2016), *…and the Cross-Section of Expected Returns*, Review of Financial Studies | Al probar muchas variables o estrategias, la mayoría de los "descubrimientos" con p < 0.05 son falsos positivos. | Se cuentan todas las configuraciones probadas por mercado y se corrige por ellas (ver Deflated Sharpe). |
| Bailey y López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management | El Sharpe del mejor de N backtests crece sólo por N. | `evaluation.deflated_sharpe`: el filtro exige un Deflated Sharpe ≥ 95%, fijado antes de ver resultados. |
| [Bailey, Borwein, López de Prado y Zhu (2017), *The Probability of Backtest Overfitting*](https://papers.ssrn.com/abstract=2326253) | Elegir el ganador dentro de muestra y reportarlo fuera de muestra sesga hacia arriba. | **Selección anidada** (`evaluation.nested_selection`): el modelo de cada bloque se elige sólo con bloques anteriores. |
| [Avramov, Cheng y Metzker (2023), *Machine Learning vs. Economic Restrictions*](https://papers.ssrn.com/abstract=3450322), Management Science | Las ganancias del ML vienen de acciones pequeñas o difíciles de arbitrar; con universos líquidos se reducen mucho. | Universo **punto en el tiempo por liquidez** (mercado `usliq`: top 50 por volumen de los 126 días previos, regla fijada de antemano) en vez de elegir listas a mano. |
| Gu, Kelly y Xiu (2020) | Reportan resultados por regla de tamaño (top 1000 / bottom 1000) en cada fecha. | Misma idea: la pertenencia al universo se decide con datos hasta t−1 (`pipeline.point_in_time_universe`). |

Decisión de método: entre el S&P 100 (lista fija de hoy) y el universo dinámico por liquidez se eligió el dinámico **antes de ver su resultado**, para no seleccionar universos por desempeño.

## 1c. Mejoras aplicadas tras la validación rigurosa (nov-2026)

Cada cambio se justifica con un resultado propio y con literatura, y tiene un objetivo medible.

| Cambio | Evidencia propia | Literatura | Objetivo |
|---|---|---|---|
| **Corrección del Deflated Sharpe**: los Sharpe de las pruebas se expresan en la frecuencia de la serie evaluada y sólo cuentan los motores elegibles | Se detectó que mezclaba frecuencias (diaria, semanal, mensual) y contaba baselines no elegibles | Bailey y López de Prado (2014) | Prueba correcta |
| **Sin plazo de 1 día** | Sharpe long-short negativo en los 6 mercados evaluados | Novy-Marx y Velikov (2016), *A Taxonomy of Anomalies and Their Trading Costs*; Avramov, Cheng y Metzker (2023) | Menos pruebas y menos penalización por pruebas múltiples |
| **Historia desde 2000** (26 años en vez de 8) | Con ~3 años fuera de muestra, un Sharpe de 0.7 no puede ser significativo | Harvey, Liu y Zhu (2016); Bailey y López de Prado (longitud mínima de backtest) | De ~3 a ~10 años fuera de muestra |
| **Portafolios traslapados** a 2 semanas y 1 mes | Sólo ~35 rebalanceos mensuales en validación | Jegadeesh y Titman (1993) | ~4x más observaciones semanales |
| **Costo de 25 pb por lado en la BMV** | 10 pb era optimista para México | Lesmond (2005), *Liquidity of emerging markets*, JFE | Ganancia de México no sobreestimada |
| **Filtro de liquidez punto en el tiempo en México** (≥ 20 M MXN/día en los 126 días previos) | Liverpool y Cuervo operan 9–19 M MXN/día | Avramov et al. (2023): la "ventaja" en acciones ilíquidas no es aprovechable | Señales sólo en acciones operables |
| **BAA10Y en vez del spread high yield** | El histórico público del high yield (ICE) inicia en 2023 | Gilchrist y Zakrajšek (2012), *Credit Spreads and Business Cycle Fluctuations* | Variable con historia completa |
| **Tasa a 2 años (DGS2)** | Las tasas son la familia más valiosa en EE. UU. (ablación) | Gürkaynak, Sack y Swanson (2005); Hanson y Stein (2015) | Medir mejor la política esperada (la ablación decide si se queda) |
| **Mercado de ETFs sectoriales** (candidato) | La señal de las 12 parece rotación crecimiento/defensivas guiada por tasas | Moskowitz y Grinblatt (1999), *Do Industries Explain Momentum?* | Probar la señal sin sesgo de supervivencia, desde 1998 |
| **Seguimiento en vivo** de pronósticos | Toda la evidencia hasta ahora es histórica | McLean y Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?* | Medición sin selección posterior |

Advertencia: más historia con listas de acciones de hoy **aumenta** el sesgo de supervivencia (Brown,
Goetzmann, Ibbotson y Ross, 1992). Por eso el mercado de ETFs sectoriales, que no lo tiene, es una prueba
de control importante.

## 1d. Bonos: curva de EE. UU. y CETES

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Cochrane y Piazzesi (2005), *Bond Risk Premia*, AER | Una combinación de tasas forward predice el exceso de retorno de bonos. | Variables de la curva completa del Tesoro (FRED: 3m, 1a, 5a, 10a, 30a). |
| Fama y Bliss (1987); Campbell y Shiller (1991) | La pendiente de la curva anticipa la prima por plazo. | `f_slope_10y3m` (10 años − 3 meses). |
| Litterman y Scheinkman (1991) | Nivel, pendiente y curvatura explican casi todo el movimiento de la curva. | `f_curvature` (2·5a − 1a − 30a). |
| Ludvigson y Ng (2009), *Macro Factors in Bond Risk Premia* | Factores macro (inflación, actividad) predicen bonos. | Inflación implícita a 10 años (T10YIE), MOVE, VIX y dólar. |
| Moskowitz, Ooi y Pedersen (2012), *Time Series Momentum* | Tendencia en bonos a 1–12 meses. | Momentum de cada ETF (variables técnicas existentes). |
| Duffee (2002); Diebold y Li (2006) | La dirección de las tasas es muy difícil de predecir fuera de muestra. | **Resultado propio:** la curva de EE. UU. se **descartó en la prueba rápida** (IC negativo: el modelo aprendió el régimen de tasas a la baja de 2003–2020 y falló con el cambio de 2022). |

CETES (28, 91, 182 y 364 días): datos de la API SIE de Banxico (requiere token gratuito). El rendimiento de
mantener cada plazo se aproxima con devengo + duración de un cupón cero. Mercado "solo compras": recomienda
el plazo, no ventas en corto. Queda **pendiente** hasta configurar `BANXICO_TOKEN`.

## 2. Bancos centrales y política monetaria

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Lucca & Moench (2015), [*The Pre-FOMC Announcement Drift*](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf), Journal of Finance | Gran parte del exceso de retorno del S&P 500 ocurre antes de los anuncios del FOMC. | Features del ciclo de la Fed (`cb_US_chg30`, `cb_US_chg90`). |
| Bernanke & Kuttner (2005), *What Explains the Stock Market's Reaction to Federal Reserve Policy?*, Journal of Finance | Las sorpresas de tasa mueven las bolsas. | Cambios de tasa de **todos** los bancos centrales del BIS y el *breadth* global de recortes/alzas. `data/central_banks.py`. **Medido** (`stockcast ablation`): en EE. UU. es el grupo que más aporta (sin él el IC a 5 d cae de 0.062 a 0.023), y los 40 bancos rinden más que 12. En México no aportó y se descarta. |
| **Jiménez-Preciado, Durán-Saldivar, Cruz-Aké & Venegas-Martínez (2025)**, [*Hawkish or Dovish? That Is the Question: Agentic Retrieval of FED Monetary Policy Report*](https://www.mdpi.com/2227-7390/13/20/3255), Mathematics 13(20):3255 | Mide la postura hawkish/dovish de los reportes de la Fed con NLP (RAG + GPT-4o). | Se probó una versión de costo cero (léxico hawkish/dovish sobre comunicados del FOMC). Se **retiró**: sólo hay ~12 comunicados recientes, insuficientes para medir si aporta a la predicción. Queda como idea si se reconstruye el histórico desde 2013. |

## 3. Papers de la lista de Scholar compartida (A. L. Jiménez-Preciado y coautores)

| Paper | Qué se tomó |
|---|---|
| *Stock portfolio optimization with competitive advantages (moat): A machine learning approach*, Mathematics 10(23):4449 (2022) | Los ratios más predictivos de moat fueron **deuda LP / utilidad neta** y **D&A / utilidad bruta**. Se calculan con yfinance y se muestran en el dashboard. `data/fundamentals.py` |
| *La dependencia del IPC con respecto a los principales índices bursátiles latinoamericanos*, Contaduría y Administración 63(4) (2018) | Se agregaron Bovespa (`^BVSP`) y el ETF de México (`EWW`) como variables del mercado MX. |
| *Behavioral economics and stock market sentiments in investment decisions in Mexico: Web scraping, NLP...*, IJEFI 15(2) (2025) | Sentimiento de noticias en español para acciones `.MX` (Google News, `hl=es-419`). `data/news.py` |
| *Propuesta de mapa térmico del sistema financiero mexicano con alertas tempranas de inestabilidad*, Análisis Económico 37(94) (2022) | Indicadores de estrés: spread high-yield, pendiente de la curva, tasa real (FRED). `data/fred.py` |
| *Clustering a sample of major and emerging economies regarding their economic policy uncertainty*, EconoQuantum 22(1) (2025) | Índice EPU (incertidumbre de política económica) como feature. |
| *Huelum Trading System: A Low-Frequency Algorithm Proposal* (2019) y *Trading strategies for ETFs: technical analysis* (2021) | Respaldan el uso de rebalanceo de baja frecuencia (cada 5 días) en el backtest en vez de operar diario, lo que baja costos. |

## 4. Sentimiento de noticias

| Paper | Qué se hizo |
|---|---|
| Araci (2019), *FinBERT*; [Cestari & Formentin (2024), Bayesian-enhanced FinBERT](https://arxiv.org/abs/2403.04427); [Zeng & Jiang (2023)](https://arxiv.org/abs/2306.02136) | VADER por defecto (ligero) y FinBERT opcional (`news.sentiment: finbert`). Las APIs gratuitas sólo dan noticias recientes, así que se guarda un histórico propio. Cuando cubra un año, entra automáticamente como feature. |

## Ideas pendientes (no implementadas)

- Sorpresas de política monetaria (Kuttner 2001) con futuros de fed funds: requiere datos de futuros que no son gratuitos.
- Calendario de reuniones de bancos centrales (para el *pre-FOMC drift*): no hay una API gratuita estable.
- Modelos fundacionales (Chronos-Bolt, TimesFM) como un motor más: posible en GPU local, pero la evidencia de 2026 muestra poca ganancia.
