"""Textos explicativos del dashboard (separados del código de la interfaz para que sea fácil editarlos)."""

INTRO = """
**stockcast** estima retornos esperados de corte transversal: en cada fecha ordena las emisoras de un mismo
mercado según su retorno esperado relativo en el horizonte seleccionado. El objeto de estimación no es el nivel
del precio ni la dirección del mercado, sino la **prima relativa** entre activos, explotable mediante una cartera
*long-short* (largo en el quintil superior, corto en el inferior) neutral al mercado.
"""

HOW_TO_READ_SIGNALS = """
- **COMPRA**: la emisora pertenece al quintil superior de retorno esperado del mercado en ese horizonte.
- **VENTA EN CORTO**: pertenece al quintil inferior. La señal es **relativa**: indica un desempeño esperado
  inferior al del resto del universo, no necesariamente un rendimiento negativo. Su interpretación correcta es
  como la pata corta de una cartera long-short; aislada, en un mercado alcista, puede generar pérdidas.
- **NEUTRAL**: quintiles intermedios; sin ventaja esperada significativa.
- **SIN SEÑAL**: en ese horizonte el modelo no superó, fuera de muestra, a la hipótesis nula de ordenamiento
  aleatorio. Los pronósticos se presentan sólo con fines descriptivos.
- **Nivel de evidencia del horizonte**: ✅ *fuerte* = la rentabilidad ajustada por riesgo y costos se mantiene tras
  la corrección por pruebas múltiples (Deflated Sharpe ≥ 95%). 🟡 *moderada* = el ordenamiento es superior al
  azar y la cartera fue rentable fuera de muestra, pero no se descarta que la rentabilidad obedezca al número de
  especificaciones evaluadas (Deflated Sharpe < 95%). Con evidencia moderada se recomienda exposición reducida o
  simulación (*paper trading*).
- **Retorno esperado**: mediana condicional del pronóstico (cuantil 0.5).
- **Intervalo p10 – p90**: intervalo de predicción al 80%, calibrado de forma conformal; en la validación, el 80%
  de los retornos observados cayó dentro de él. Su amplitud refleja la incertidumbre del pronóstico.
"""

HOW_TO_READ_RANGE = """
Cada punto representa la **mediana condicional** del retorno y el segmento el **intervalo de predicción p10 – p90**.
- Un punto a la derecha de 0 implica un escenario central de rendimiento positivo.
- La amplitud del segmento aproxima la volatilidad condicional de la emisora.
- La información relevante es la **posición relativa** en el ordenamiento, no el nivel absoluto del pronóstico.
"""

HOW_TO_READ_STOCK_CHART = """
- **Velas**: rango diario de apertura, máximo, mínimo y cierre. Verde: cierre superior a la apertura; rojo: inferior.
- **Media móvil de 50 días** (naranja) y **de 200 días** (morada): tendencia de mediano y largo plazo. Un precio
  por encima de ambas se asocia con una tendencia alcista.
- **Abanico azul**: pronóstico del modelo. La línea discontinua es la mediana condicional y la zona sombreada el
  intervalo p10 – p90, que se amplía con el horizonte porque la varianza del retorno acumulado crece con él.
- **RSI (14)**: razón entre la magnitud de las alzas y de las bajas recientes (escala 0 – 100). Valores superiores a
  70 indican sobrecompra; inferiores a 30, sobreventa.
"""

HOW_TO_READ_RATES = """
Cada serie es la **tasa de política monetaria** fijada por el banco central (costo del dinero a un día).
- **Alzas** (postura restrictiva): elevan la tasa de descuento y endurecen las condiciones financieras; tienden a
  presionar las valuaciones, en particular de emisoras de crecimiento (mayor duración de sus flujos) y apalancadas.
- **Recortes** (postura expansiva): abaratan el financiamiento y tienden a favorecer a los activos de riesgo.
- Los precios de los activos reaccionan principalmente al **cambio** de la tasa y a la **dirección del ciclo**,
  más que a su nivel; por ello el modelo emplea la variación a 90 días.
"""

HOW_TO_READ_CYCLE = """
Indicador de difusión del ciclo monetario global:
**(bancos que recortaron − bancos que subieron) / total**, en los últimos 90 días.
- **Positivo**: predominan los recortes; expansión de la liquidez global, entorno favorable a activos de riesgo.
- **Negativo**: predominan las alzas (p. ej., 2022); endurecimiento monetario sincronizado.
- **Cercano a 0**: posturas divergentes o pausa generalizada.
"""

FRED_SERIES = {
    "USEPUINDXD": {
        "titulo": "Incertidumbre de política económica (EE. UU.)",
        "que_es": "Índice de Baker, Bloom y Davis (2016), construido a partir de la frecuencia de términos de "
                  "incertidumbre económica y de política (aranceles, impuestos, regulación) en la prensa. Promedio "
                  "móvil de 30 días.",
        "alto": "Incertidumbre elevada: posposición de decisiones de inversión y mayor prima de riesgo exigida; "
                "suele coincidir con caídas y aumentos de volatilidad.",
        "bajo": "Incertidumbre reducida: entorno más predecible, normalmente favorable para la renta variable.",
    },
    "BAA10Y": {
        "titulo": "Diferencial corporativo Baa − Tesoro a 10 años",
        "que_es": "Prima que pagan los bonos corporativos calificados Baa (el escalón inferior del grado de "
                  "inversión) sobre el bono del Tesoro a 10 años, en puntos porcentuales. Indicador estándar de "
                  "estrés crediticio y del ciclo financiero (Gilchrist y Zakrajšek, 2012).",
        "alto": "Diferencial amplio: mayor probabilidad percibida de incumplimiento; históricamente asociado con "
                "recesiones y mercados bajistas (2008, 2020).",
        "bajo": "Diferencial estrecho: condiciones crediticias holgadas y apetito por riesgo; niveles muy bajos "
                "pueden reflejar una valuación complaciente del riesgo.",
    },
    "T10Y2Y": {
        "titulo": "Pendiente de la curva de rendimientos (10 años − 2 años)",
        "que_es": "Diferencia entre los rendimientos del Tesoro a 10 y a 2 años. En condiciones normales es positiva, "
                  "por la prima por plazo.",
        "alto": "Pendiente positiva y pronunciada: expectativas de crecimiento; típica de fases de recuperación.",
        "bajo": "Curva **invertida** (negativa): el mercado descuenta recortes ante una desaceleración. Ha antecedido "
                "a casi todas las recesiones de EE. UU. desde 1970, con 6 a 24 meses de anticipación "
                "(Estrella y Mishkin, 1998).",
    },
    "DGS2": {
        "titulo": "Rendimiento del Tesoro a 2 años",
        "que_es": "Resume la trayectoria esperada de la tasa de fondos federales en los dos años siguientes "
                  "(Gürkaynak, Sack y Swanson, 2005). El modelo emplea su variación.",
        "alto": "Alza acelerada: el mercado anticipa una política más restrictiva; presión sobre emisoras de "
                "crecimiento.",
        "bajo": "Caída acelerada: el mercado anticipa recortes, con frecuencia por expectativas de desaceleración.",
    },
    "DFII10": {
        "titulo": "Tasa real a 10 años (TIPS)",
        "que_es": "Rendimiento del bono indexado a la inflación a 10 años; aproxima el costo real del capital a "
                  "largo plazo.",
        "alto": "Tasa real elevada: mayor tasa de descuento, lo que reduce el valor presente de los flujos, en "
                "particular de emisoras de larga duración (tecnología, crecimiento).",
        "bajo": "Tasa real baja o negativa: sostiene valuaciones elevadas de renta variable y del oro.",
    },
}

METHOD = """
### 1. Objeto de estimación

El sistema estima **retornos esperados de corte transversal**: en cada fecha *t* ordena las emisoras de un mismo
mercado según su rendimiento esperado en los *h* días hábiles siguientes (*h* = 5, 10 o 21). No se intenta
pronosticar la dirección del mercado agregado (*market timing*), cuya predictibilidad fuera de muestra es
notoriamente débil (Welch y Goyal, 2008), sino la **prima relativa** entre activos, fundamento de las carteras
*long-short*.

### 2. Variable dependiente

Para cada emisora *i* y fecha *t*, la variable dependiente es el **retorno logarítmico futuro**
`y(i,t) = ln(P(i,t+h) / P(i,t))`. Se modelan retornos y no niveles de precio porque estos últimos son no
estacionarios: un pronóstico que reproduce el último precio observado minimiza el error de nivel sin contener
información alguna sobre el retorno.

### 3. Variables explicativas

Todas se construyen **exclusivamente con información disponible al cierre de *t*** (sin sesgo de anticipación):

| Familia | Ejemplos | Fundamento económico |
|---|---|---|
| **Momentum y reversión** | retornos a 1, 5, 21, 63 y 126 días; momentum relativo al mercado | Persistencia de rendimientos a 3–12 meses (Jegadeesh y Titman, 1993) y reversión de corto plazo (Lehmann, 1990). |
| **Riesgo** | volatilidad a 5/21/63 días, ATR, distancia al máximo anual | Anomalía de baja volatilidad (Ang et al., 2006; Frazzini y Pedersen, 2014). |
| **Indicadores técnicos** | RSI, MACD, bandas de Bollinger, estocástico, volumen anormal | Presión de órdenes de corto plazo y sobrerreacción. |
| **Mercado y macro** | índice de referencia, VIX, tasa a 10 años, dólar, petróleo, tipo de cambio | Exposiciones heterogéneas (betas) a choques comunes; el modelo aprende qué emisoras se benefician en cada régimen. |
| **Política monetaria** | variación de la tasa de ~40 bancos centrales (BIS), ciclo global, diferenciales contra la Fed | Canal de tasas y de liquidez: el costo del capital modifica el valor presente de los flujos y la tolerancia al riesgo. |
| **Estrés financiero** | diferencial Baa, pendiente 10a–2a, tasa real, incertidumbre de política | Indicadores adelantados de régimen: en episodios de estrés las emisoras defensivas superan a las cíclicas. |

Cada mercado **excluye automáticamente** las familias cuya contribución marginal fuera de muestra es nula o
negativa (análisis de ablación), con el fin de reducir la dimensionalidad y el riesgo de sobreajuste.

### 4. Especificación

- **Panel agrupado** (*pooled panel*): un solo modelo por mercado y horizonte, estimado con todas las emisoras,
  como en Gu, Kelly y Xiu (2020). Se gana eficiencia estadística respecto a modelos individuales por emisora.
- **Regresión cuantil** (p10, p50, p90) mediante *gradient boosting* (LightGBM), que captura no linealidades e
  interacciones (p. ej., momentum condicionado al régimen de volatilidad), y **regresión Ridge** como
  especificación lineal regularizada. Se evalúan también su **ensamble** (promedio) y dos referencias: caminata
  aleatoria y momentum simple.
- La especificación de cada mercado y horizonte se elige mediante **selección anidada** (sección 5).

### 5. Validación

- **Walk-forward con purga**: estimación con información pasada y evaluación en un periodo posterior no observado,
  con una separación de *h* días que impide el traslape entre la variable dependiente de entrenamiento y la de
  prueba (López de Prado, 2018). Cuatro bloques sobre el 40% final de una muestra 2000–2026.
- **Prueba de permutación**: los pronósticos se permutan aleatoriamente entre emisoras dentro de cada fecha (300
  réplicas). Se exige que el coeficiente de información (IC) y el Sharpe de la cartera superen a la distribución
  bajo la hipótesis nula de ordenamiento aleatorio (p < 0.05).
- **Cartera simulada con costos**: largo en el quintil superior y corto en el inferior, con carteras traslapadas
  formadas semanalmente (Jegadeesh y Titman, 1993). Se descuentan los costos de transacción de cada mercado
  (10 pb; BMV 25 pb) sobre la **rotación neta** (Novy-Marx y Velikov, 2016) y 50 pb anuales por préstamo de
  valores.
- **Selección anidada** (*nested walk-forward*): la especificación evaluada en cada bloque se elige sólo con los
  bloques anteriores; el desempeño reportado corresponde al **procedimiento de selección**, no a la mejor
  especificación identificada *ex post*.
- **Corrección por pruebas múltiples**: dado el número de especificaciones evaluadas (modelos × horizontes), se
  calcula el **Deflated Sharpe Ratio** (Bailey y López de Prado, 2014), en línea con Harvey, Liu y Zhu (2016).
- **Criterio de activación**: un horizonte emite señales sólo si el IC es significativo (p < 0.05, IC ≥ 0.02) y la
  cartera long-short obtiene un Sharpe ≥ 0.3 significativamente superior al azar.
- **Intervalos conformales** (*conformalized quantile regression*; Romano, Patterson y Candès, 2019): se reescala
  el intervalo p10 – p90 para que su cobertura empírica sea del 80%.

### 6. Limitaciones

- **La dirección del mercado no es predecible con este sistema**: ninguna especificación supera a la regla
  "siempre sube" en tasa de acierto direccional. El valor se concentra en el **ordenamiento relativo**.
- **Magnitud del efecto**: un IC de 0.03 – 0.07 es económicamente relevante, pero implica que el ordenamiento
  acierta sólo marginalmente más que el azar en cada fecha; la rentabilidad proviene de la repetición y la
  diversificación.
- **Sesgo de supervivencia**: los universos se definen con listas actuales; las emisoras deslistadas no se
  incluyen, lo que puede sobrestimar el desempeño histórico. El universo dinámico por liquidez lo atenúa, pero no
  lo elimina.
- **Inestabilidad de régimen**: las relaciones estimadas pueden dejar de operar. La validación se repite
  semanalmente y los pronósticos emitidos se contrastan con los retornos realizados (seguimiento en vivo).
- **Universos reducidos** (12 y 26 emisoras): pocas posiciones por pata, lo que incrementa la varianza de los
  resultados.
- Uso académico y educativo; **no constituye recomendación de inversión**.
"""

CONCEPTS = {
    "IC (coeficiente de información)": "Correlación de rangos de Spearman entre el retorno pronosticado y el "
        "observado entre las emisoras de una misma fecha, promediada en el tiempo. 0 = ordenamiento sin "
        "información; 1 = ordenamiento perfecto. Un IC sostenido de 0.03 – 0.10 se considera económicamente "
        "relevante (Grinold y Kahn, 2000).",
    "Razón de Sharpe": "Retorno medio de la cartera dividido entre su desviación estándar, anualizado. Mide el "
        "rendimiento por unidad de riesgo total.",
    "Cartera long-short": "Posición larga en las emisoras con mayor retorno esperado y corta, por el mismo monto, en "
        "las de menor. Su rendimiento depende del diferencial entre ambas patas y no de la dirección del mercado.",
    "Venta en corto": "Venta de un valor obtenido en préstamo, con la obligación de recomprarlo posteriormente. "
        "Genera ganancia si el precio disminuye (o, dentro de una cartera long-short, si aumenta menos que la pata "
        "larga). Implica un costo de préstamo.",
    "p-valor": "Probabilidad de observar un resultado al menos tan favorable bajo la hipótesis nula (aquí, "
        "ordenamiento aleatorio). Valores inferiores a 0.05 se consideran evidencia contra la hipótesis nula.",
    "Mediana / p10 / p90": "Cuantiles condicionales del retorno: escenario central, inferior y superior. El "
        "intervalo p10 – p90 tiene una cobertura nominal del 80%.",
    "Walk-forward": "Esquema de validación que reproduce el uso en tiempo real: estimar con el pasado y evaluar en "
        "el periodo inmediato siguiente, avanzando secuencialmente en el tiempo.",
    "Punto base (pb)": "Una centésima de punto porcentual (0.01%). Un alza de 25 pb equivale a +0.25 puntos "
        "porcentuales.",
    "Drawdown máximo": "Mayor pérdida acumulada desde un máximo previo de la cartera; aproxima el riesgo de cola "
        "experimentado por el inversionista.",
    "Ablación": "Ejercicio que reestima el modelo excluyendo una familia de variables para medir su contribución "
        "marginal fuera de muestra.",
    "Deflated Sharpe Ratio": "Probabilidad de que el Sharpe verdadero sea positivo, corrigiendo por el número de "
        "especificaciones evaluadas y por la no normalidad de los rendimientos (Bailey y López de Prado, 2014). Al "
        "evaluar muchas estrategias, la mejor exhibe un Sharpe elevado aun cuando ninguna tenga valor; el DSR "
        "corrige ese sesgo de selección.",
    "Selección anidada": "Elección de la especificación utilizando únicamente información anterior al periodo de "
        "evaluación, repetida en cada bloque. Elimina el sesgo de seleccionar *ex post* la especificación ganadora.",
    "Universo punto en el tiempo": "Conjunto de emisoras elegible en cada fecha con la información disponible en "
        "ese momento (aquí, las 50 de mayor volumen operado en los seis meses previos), en lugar de la composición "
        "actual del índice.",
    "Rotación neta": "Fracción de la cartera que efectivamente se compra o vende en cada rebalanceo, Σ|w(t) − "
        "w(t−1)|. Los costos de transacción se aplican sólo sobre ella.",
}
