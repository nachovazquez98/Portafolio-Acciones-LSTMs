# Diccionario de variables

Variables que puede usar el modelo (cada mercado descarta los grupos sin aporte; ver la pestaña *Validación* o `stockcast ablation`). Todas se calculan sólo con información disponible al cierre del día.

## Estrés e incertidumbre

| Variable | Nombre | Qué mide |
|---|---|---|
| `f_BAA10Y` | Spread corporativo Baa − Tesoro 10 años (estrés crediticio): nivel | Valor actual de «Spread corporativo Baa − Tesoro 10 años (estrés crediticio)» (FRED, serie BAA10Y). Más alto = más estrés crediticio (inversionistas exigen más premio por riesgo) |
| `f_BAA10Y_chg30` | Spread corporativo Baa − Tesoro 10 años (estrés crediticio): cambio 30 d | Cambio en 30 días de «Spread corporativo Baa − Tesoro 10 años (estrés crediticio)» (FRED, serie BAA10Y). |
| `f_DFII10_chg30` | Tasa real 10 años (TIPS): cambio 30 d | Cambio en 30 días de «Tasa real 10 años (TIPS)» (FRED, serie DFII10). |
| `f_DGS2_chg30` | Tasa del Tesoro a 2 años: cambio 30 d | Cambio en 30 días de «Tasa del Tesoro a 2 años» (FRED, serie DGS2). |
| `f_T10Y2Y_chg30` | Pendiente curva Tesoro 10a-2a: cambio 30 d | Cambio en 30 días de «Pendiente curva Tesoro 10a-2a» (FRED, serie T10Y2Y). |
| `f_USEPUINDXD` | Índice de incertidumbre de política económica (EE. UU.): nivel | Valor actual de «Índice de incertidumbre de política económica (EE. UU.)» (FRED, serie USEPUINDXD). Promedio de 30 días; más alto = más incertidumbre de política económica |
| `f_USEPUINDXD_chg30` | Índice de incertidumbre de política económica (EE. UU.): cambio 30 d | Cambio en 30 días de «Índice de incertidumbre de política económica (EE. UU.)» (FRED, serie USEPUINDXD). |

## Mercado y macro

| Variable | Nombre | Qué mide |
|---|---|---|
| `m_BVSP_ret21` | Bovespa (Brasil): retorno 1 mes | Retorno logarítmico de Bovespa (Brasil) en los últimos 30 días naturales. |
| `m_BVSP_ret5` | Bovespa (Brasil): retorno 1 semana | Retorno logarítmico de Bovespa (Brasil) en los últimos 7 días naturales. |
| `m_CLF_ret21` | Petróleo WTI: retorno 1 mes | Retorno logarítmico de Petróleo WTI en los últimos 30 días naturales. |
| `m_CLF_ret5` | Petróleo WTI: retorno 1 semana | Retorno logarítmico de Petróleo WTI en los últimos 7 días naturales. |
| `m_DXYNYB_ret21` | Índice del dólar (DXY): retorno 1 mes | Retorno logarítmico de Índice del dólar (DXY) en los últimos 30 días naturales. |
| `m_DXYNYB_ret5` | Índice del dólar (DXY): retorno 1 semana | Retorno logarítmico de Índice del dólar (DXY) en los últimos 7 días naturales. |
| `m_EURUSDX_ret21` | Tipo de cambio EUR/USD: retorno 1 mes | Retorno logarítmico de Tipo de cambio EUR/USD en los últimos 30 días naturales. |
| `m_EURUSDX_ret5` | Tipo de cambio EUR/USD: retorno 1 semana | Retorno logarítmico de Tipo de cambio EUR/USD en los últimos 7 días naturales. |
| `m_EWW_ret21` | ETF de México (EWW): retorno 1 mes | Retorno logarítmico de ETF de México (EWW) en los últimos 30 días naturales. |
| `m_EWW_ret5` | ETF de México (EWW): retorno 1 semana | Retorno logarítmico de ETF de México (EWW) en los últimos 7 días naturales. |
| `m_GCF_ret21` | Oro: retorno 1 mes | Retorno logarítmico de Oro en los últimos 30 días naturales. |
| `m_GCF_ret5` | Oro: retorno 1 semana | Retorno logarítmico de Oro en los últimos 7 días naturales. |
| `m_GDAXI_ret21` | DAX (Alemania): retorno 1 mes | Retorno logarítmico de DAX (Alemania) en los últimos 30 días naturales. |
| `m_GDAXI_ret5` | DAX (Alemania): retorno 1 semana | Retorno logarítmico de DAX (Alemania) en los últimos 7 días naturales. |
| `m_GSPC_ret21` | S&P 500: retorno 1 mes | Retorno logarítmico de S&P 500 en los últimos 30 días naturales. |
| `m_GSPC_ret5` | S&P 500: retorno 1 semana | Retorno logarítmico de S&P 500 en los últimos 7 días naturales. |
| `m_MXNX_ret21` | Tipo de cambio USD/MXN: retorno 1 mes | Retorno logarítmico de Tipo de cambio USD/MXN en los últimos 30 días naturales. |
| `m_MXNX_ret5` | Tipo de cambio USD/MXN: retorno 1 semana | Retorno logarítmico de Tipo de cambio USD/MXN en los últimos 7 días naturales. |
| `m_MXX_ret21` | IPC (Bolsa Mexicana): retorno 1 mes | Retorno logarítmico de IPC (Bolsa Mexicana) en los últimos 30 días naturales. |
| `m_MXX_ret5` | IPC (Bolsa Mexicana): retorno 1 semana | Retorno logarítmico de IPC (Bolsa Mexicana) en los últimos 7 días naturales. |
| `m_STOXX50E_ret21` | Euro Stoxx 50: retorno 1 mes | Retorno logarítmico de Euro Stoxx 50 en los últimos 30 días naturales. |
| `m_STOXX50E_ret5` | Euro Stoxx 50: retorno 1 semana | Retorno logarítmico de Euro Stoxx 50 en los últimos 7 días naturales. |
| `m_TNX` | Tasa del Tesoro de EE. UU. a 10 años: nivel | Valor actual de Tasa del Tesoro de EE. UU. a 10 años. |
| `m_TNX_chg21` | Tasa del Tesoro de EE. UU. a 10 años: cambio 1 mes | Cambio en puntos de Tasa del Tesoro de EE. UU. a 10 años en los últimos 30 días naturales. |
| `m_VIX` | VIX (volatilidad esperada del S&P 500): nivel | Valor actual de VIX (volatilidad esperada del S&P 500). |
| `m_VIX_chg21` | VIX (volatilidad esperada del S&P 500): cambio 1 mes | Cambio en puntos de VIX (volatilidad esperada del S&P 500) en los últimos 30 días naturales. |

## Tasas de bancos centrales

| Variable | Nombre | Qué mide |
|---|---|---|
| `cb_AR_chg90` | Banco Central de la República Argentina: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de la República Argentina en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_AU_chg90` | Banco de la Reserva de Australia: cambio de tasa 90 d | Cambio de la tasa de política de Banco de la Reserva de Australia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_BR_chg90` | Banco Central do Brasil: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central do Brasil en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CA_chg90` | Banco de Canadá: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Canadá en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CH_chg90` | Banco Nacional Suizo: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional Suizo en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CL_chg90` | Banco Central de Chile: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Chile en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CN_chg90` | Banco Popular de China: cambio de tasa 90 d | Cambio de la tasa de política de Banco Popular de China en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CO_chg90` | Banco de la República (Colombia): cambio de tasa 90 d | Cambio de la tasa de política de Banco de la República (Colombia) en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_CZ_chg90` | Banco Nacional Checo: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional Checo en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_DK_chg90` | Danmarks Nationalbank: cambio de tasa 90 d | Cambio de la tasa de política de Danmarks Nationalbank en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_GB_chg90` | Banco de Inglaterra: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Inglaterra en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_GR_chg90` | GR: cambio de tasa 90 d | Cambio de la tasa de política de GR en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_HK_chg90` | Autoridad Monetaria de Hong Kong: cambio de tasa 90 d | Cambio de la tasa de política de Autoridad Monetaria de Hong Kong en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_HR_chg90` | Banco Nacional de Croacia: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Croacia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_HU_chg90` | Banco Nacional de Hungría: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Hungría en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_ID_chg90` | Banco de Indonesia: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Indonesia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_IL_chg90` | Banco de Israel: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Israel en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_IN_chg90` | Banco de la Reserva de India: cambio de tasa 90 d | Cambio de la tasa de política de Banco de la Reserva de India en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_IS_chg90` | Banco Central de Islandia: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Islandia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_JP_chg90` | Banco de Japón: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Japón en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_KR_chg90` | Banco de Corea: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Corea en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_KW_chg90` | Banco Central de Kuwait: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Kuwait en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_MA_chg90` | MA: cambio de tasa 90 d | Cambio de la tasa de política de MA en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_MK_chg90` | Banco Nacional de Macedonia del Norte: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Macedonia del Norte en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_MX_chg90` | Banco de México: cambio de tasa 90 d | Cambio de la tasa de política de Banco de México en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_MY_chg90` | Banco Negara Malasia: cambio de tasa 90 d | Cambio de la tasa de política de Banco Negara Malasia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_NO_chg90` | Norges Bank: cambio de tasa 90 d | Cambio de la tasa de política de Norges Bank en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_NZ_chg90` | Banco de la Reserva de Nueva Zelanda: cambio de tasa 90 d | Cambio de la tasa de política de Banco de la Reserva de Nueva Zelanda en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_PE_chg90` | Banco Central de Reserva del Perú: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Reserva del Perú en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_PH_chg90` | Bangko Sentral ng Pilipinas: cambio de tasa 90 d | Cambio de la tasa de política de Bangko Sentral ng Pilipinas en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_PL_chg90` | Banco Nacional de Polonia: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Polonia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_RO_chg90` | Banco Nacional de Rumanía: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Rumanía en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_RS_chg90` | Banco Nacional de Serbia: cambio de tasa 90 d | Cambio de la tasa de política de Banco Nacional de Serbia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_RU_chg90` | Banco de Rusia: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Rusia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_SA_chg90` | Banco Central de Arabia Saudita: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Arabia Saudita en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_SE_chg90` | Riksbank (Suecia): cambio de tasa 90 d | Cambio de la tasa de política de Riksbank (Suecia) en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_TH_chg90` | Banco de Tailandia: cambio de tasa 90 d | Cambio de la tasa de política de Banco de Tailandia en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_TR_chg90` | Banco Central de Turquía: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central de Turquía en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_US_chg30` | Reserva Federal (EE. UU.): cambio de tasa 30 d | Cambio de la tasa de política de Reserva Federal (EE. UU.) en 30 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_US_chg90` | Reserva Federal (EE. UU.): cambio de tasa 90 d | Cambio de la tasa de política de Reserva Federal (EE. UU.) en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_XM_chg90` | Banco Central Europeo: cambio de tasa 90 d | Cambio de la tasa de política de Banco Central Europeo en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_ZA_chg90` | Banco de la Reserva de Sudáfrica: cambio de tasa 90 d | Cambio de la tasa de política de Banco de la Reserva de Sudáfrica en 90 días, en puntos base. Negativo = recortes (política más laxa), positivo = alzas. |
| `cb_breadth` | Ciclo monetario global | (Bancos centrales que recortaron − los que subieron) / total, en 90 días. +1 = todos recortando, -1 = todos subiendo. |
| `cb_spread_GB_US` | Diferencial de tasas: Banco de Inglaterra − Fed | Tasa de Banco de Inglaterra menos la de la Fed (puntos porcentuales). Afecta tipo de cambio y flujos de capital. |
| `cb_spread_JP_US` | Diferencial de tasas: Banco de Japón − Fed | Tasa de Banco de Japón menos la de la Fed (puntos porcentuales). Afecta tipo de cambio y flujos de capital. |
| `cb_spread_MX_US` | Diferencial de tasas: Banco de México − Fed | Tasa de Banco de México menos la de la Fed (puntos porcentuales). Afecta tipo de cambio y flujos de capital. |
| `cb_spread_XM_US` | Diferencial de tasas: Banco Central Europeo − Fed | Tasa de Banco Central Europeo menos la de la Fed (puntos porcentuales). Afecta tipo de cambio y flujos de capital. |

## Técnicos de la acción

| Variable | Nombre | Qué mide |
|---|---|---|
| `atr_pct` | Rango verdadero promedio (%) | ATR de 14 días dividido entre el precio: tamaño típico del movimiento diario. |
| `bb_pos` | Posición en Bandas de Bollinger | Dónde está el precio dentro de las bandas de 20 días: -1 = banda inferior, 0 = media, +1 = superior. |
| `dist_52w_high` | Distancia al máximo de 52 semanas | Precio / máximo del último año − 1. 0 = en máximos. |
| `macd_hist` | Histograma MACD | MACD (EMA 12 − EMA 26) menos su señal de 9 días, dividido entre el precio. Mide si el impulso de corto plazo se acelera o se frena. |
| `rel_ret_21` | Momentum relativo 21 días | Retorno de 21 días de la acción menos el promedio de todas las acciones del mercado ese día. Positivo: le va mejor que al resto. |
| `rel_ret_63` | Momentum relativo 63 días | Retorno de 63 días de la acción menos el promedio de todas las acciones del mercado ese día. Positivo: le va mejor que al resto. |
| `ret_1` | Retorno de 1 día | Cambio porcentual (logarítmico) del precio respecto al cierre anterior. |
| `ret_126` | Momentum 126 días | Retorno logarítmico de la acción en los últimos 126 días hábiles (~6.0 meses). |
| `ret_21` | Momentum 21 días | Retorno logarítmico de la acción en los últimos 21 días hábiles (~1.0 meses). |
| `ret_5` | Momentum 5 días | Retorno logarítmico de la acción en los últimos 5 días hábiles (~0.2 meses). |
| `ret_63` | Momentum 63 días | Retorno logarítmico de la acción en los últimos 63 días hábiles (~3.0 meses). |
| `rsi_14` | RSI 14 días | Índice de fuerza relativa (0–100). > 70 sobrecompra, < 30 sobreventa. |
| `sma200_ratio` | Distancia a la media de 200 días | Precio / media móvil de 200 días − 1. Tendencia de largo plazo (~10 meses). |
| `sma50_ratio` | Distancia a la media de 50 días | Precio / media móvil de 50 días − 1. Positivo: cotiza por encima de su tendencia de ~2 meses. |
| `stoch_k` | Estocástico %K | Dónde cerró el precio dentro del rango máximo-mínimo de 14 días (0–100). |
| `vol_21` | Volatilidad 21 días | Desviación estándar anualizada de los retornos diarios de los últimos 21 días. |
| `vol_5` | Volatilidad 5 días | Desviación estándar anualizada de los retornos diarios de los últimos 5 días. |
| `vol_63` | Volatilidad 63 días | Desviación estándar anualizada de los retornos diarios de los últimos 63 días. |
| `vol_ratio` | Volatilidad reciente vs. trimestral | Volatilidad de 5 días dividida entre la de 63 días. > 1: la acción está más agitada de lo normal. |
| `volume_z` | Volumen anormal | Volumen de hoy en desviaciones estándar respecto a su promedio de 20 días. |
