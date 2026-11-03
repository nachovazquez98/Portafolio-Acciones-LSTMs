# Papers usados y cómo influyeron en el código

Revisión hecha en noviembre de 2026. Para cada paper se indica qué se tomó y en qué archivo está.

## 1. Qué modelo usar (más barato, misma o mejor precisión)

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies 33(5) — [NBER w25398](https://www.nber.org/papers/w25398) | Árboles y redes son los mejores, y la ganancia viene de interacciones no lineales. Las señales dominantes son momentum, liquidez y volatilidad. | Modelo **global** de LightGBM (un modelo para todas las acciones) con features de momentum, volatilidad y volumen. `models/gbm.py`, `features.py` |
| [LGBM on Stock Returns Prediction and Portfolio Construction (2025)](https://scitepress.org/PublishedPapers/2025/137013) | LightGBM supera a RF, XGBoost y LSTM en retornos diarios. | LightGBM queda como motor por defecto y LSTM pasa a opcional. |
| [Gradient Boosting Decision Tree with LSTM for Investment Prediction (2025)](https://arxiv.org/abs/2505.23084) | Combinar LSTM con GBDT mejora sobre cada modelo por separado. | Motor `ensemble` (LightGBM + Ridge) y motor `lstm` para comparar. `models/linear.py` |
| Zeng et al. (2023), [*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504) (DLinear) | Un modelo lineal simple le gana a Transformers complejos. La ventaja de esos modelos venía del **pronóstico directo multi-horizonte**, no de la arquitectura. | Se pasó del pronóstico recursivo del notebook (30 pasos alimentando su propia predicción) a pronóstico **directo** por horizonte. Se agregó `ridge` como baseline lineal. |
| Noguer i Alonso & Pereira Franklin (2026), [*Pretrained Time-Series Foundation Models for Financial Return Forecasting*](https://arxiv.org/abs/2606.27100) | Chronos, TimesFM y Moirai ganan pocas veces, y las ganancias frente a la caminata aleatoria son "pequeñas y escasas". | No se agregaron (son pesados y aportan poco). Todo modelo se compara contra la caminata aleatoria. `evaluation.py` |
| Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* (NeurIPS) | Intervalos con cobertura garantizada a partir de residuos fuera de muestra. | Las bandas p10–p90 se calibran con un factor conformal. `evaluation.conformal_scale` |
| Timmermann (2006), *Forecast Combinations*, Handbook of Economic Forecasting | Promediar modelos distintos reduce la varianza del error. | Motor `ensemble`. |

## 2. Bancos centrales y política monetaria

| Paper | Hallazgo | Qué se hizo |
|---|---|---|
| Lucca & Moench (2015), [*The Pre-FOMC Announcement Drift*](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf), Journal of Finance | Gran parte del exceso de retorno del S&P 500 ocurre antes de los anuncios del FOMC. | Features del ciclo de la Fed (`cb_US_chg30`, `cb_US_chg90`). |
| Bernanke & Kuttner (2005), *What Explains the Stock Market's Reaction to Federal Reserve Policy?*, Journal of Finance | Las sorpresas de tasa mueven las bolsas. | Cambios de tasa de **todos** los bancos centrales del BIS y el *breadth* global de recortes/alzas. `data/central_banks.py` |
| **Jiménez-Preciado, Durán-Saldivar, Cruz-Aké & Venegas-Martínez (2025)**, [*Hawkish or Dovish? That Is the Question: Agentic Retrieval of FED Monetary Policy Report*](https://www.mdpi.com/2227-7390/13/20/3255), Mathematics 13(20):3255 | Mide la postura hawkish/dovish de los reportes de la Fed con NLP (RAG + GPT-4o). | Versión de **costo cero**: léxico hawkish/dovish sobre los comunicados del FOMC (RSS oficial), mostrado en el dashboard. `data/cb_tone.py` |

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
