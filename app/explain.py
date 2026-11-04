"""Textos explicativos del dashboard (separados del código de la interfaz para que sea fácil editarlos)."""

INTRO = """
**stockcast** estima qué acciones tienen más probabilidad de **ganarle al resto de su mercado** en los próximos
días, y cuáles de quedarse atrás. No predice el precio exacto: **ordena** las acciones de mejor a peor
perspectiva. Comprar las de arriba y vender en corto las de abajo es lo que se comprobó que funciona.
"""

HOW_TO_READ_SIGNALS = """
- **COMPRA**: está en el 20% de acciones con mejor retorno esperado del mercado para ese plazo.
- **VENTA EN CORTO**: está en el 20% con peor retorno esperado. Es una señal **relativa**: se espera que le vaya
  peor que al resto, no necesariamente que baje. Úsala **en pareja con las compras** (estrategia long-short,
  neutral al mercado). Sola, en un mercado alcista, puede perder.
- **NEUTRAL**: en medio del ranking; sin ventaja clara.
- **SIN SEÑAL**: en este plazo el modelo **no demostró** ganar más que el azar en el pasado, así que no se
  recomienda operar con él (se muestra sólo como referencia).
- **Nivel de evidencia del plazo**: ✅ *fuerte* = la ganancia sobrevive incluso a la corrección por haber probado
  muchas configuraciones. 🟡 *moderada* = ordena mejor que el azar y ganó en la validación, pero esa ganancia
  todavía podría ser suerte (Deflated Sharpe < 95%). Con evidencia moderada conviene operar poco o en *paper*.
- **Retorno esperado**: la *mediana* del pronóstico (escenario central).
- **Rango p10 – p90**: en 8 de cada 10 casos el retorno real cayó dentro de este rango durante la validación.
  Un rango ancho significa más incertidumbre.
"""

HOW_TO_READ_RANGE = """
Cada punto es el **retorno esperado** (mediana) y la línea es el **rango probable** (p10 – p90).
- Si el punto está a la derecha del 0, el escenario central es de ganancia.
- Una línea larga indica una acción más volátil o más incierta.
- Lo importante es la **posición relativa**: las de arriba tienen mejores perspectivas que las de abajo.
"""

HOW_TO_READ_STOCK_CHART = """
- **Velas**: cada vela es un día. Verde = cerró arriba de su apertura; roja = abajo. Las mechas muestran el
  máximo y el mínimo del día.
- **Media de 50 días (naranja)**: tendencia de corto/mediano plazo. **Media de 200 días (morada)**: tendencia de
  largo plazo. Precio arriba de ambas suele indicar tendencia alcista sana.
- **Abanico azul**: pronóstico del modelo. La línea punteada es el escenario central y la zona sombreada el
  rango p10 – p90. Se abre con el tiempo porque la incertidumbre crece con el plazo.
- **RSI (abajo)**: fuerza de las subidas recientes contra las bajadas (0 – 100). Arriba de 70 = sobrecompra
  (subió muy rápido, podría corregir). Abajo de 30 = sobreventa (cayó muy rápido, podría rebotar).
"""

HOW_TO_READ_RATES = """
Cada línea es la **tasa de interés de referencia** que fija un banco central (lo que cuesta el dinero a un día).
- **Subidas** (política *restrictiva*): el banco frena la economía para bajar la inflación. Suelen presionar a
  la baja a las acciones, sobre todo a las de crecimiento y a las más endeudadas.
- **Bajadas** (política *expansiva*): abaratan el crédito y suelen favorecer a las acciones.
- Lo que más mueve a los mercados es el **cambio** de la tasa y la **dirección del ciclo**, no el nivel. Por eso
  el modelo usa el cambio en 90 días, no la tasa en sí.
"""

HOW_TO_READ_CYCLE = """
Resume a todos los bancos centrales en una sola línea:
**(bancos que recortaron − bancos que subieron) / total**, en los últimos 90 días.
- **Positivo** (zona arriba de 0): la mayoría está bajando tasas → liquidez global en aumento, entorno
  favorable para activos de riesgo.
- **Negativo**: la mayoría está subiendo tasas (como en 2022) → endurecimiento global.
- Cerca de 0: bancos divididos o en pausa.
"""

FRED_SERIES = {
    "USEPUINDXD": {
        "titulo": "Incertidumbre de política económica (EE. UU.)",
        "que_es": "Índice de Baker, Bloom y Davis basado en cuántas notas de prensa hablan de incertidumbre "
                  "económica y de política (aranceles, impuestos, regulación). Promedio de 30 días.",
        "alto": "Alta incertidumbre: las empresas posponen inversión y los mercados exigen más premio por riesgo; "
                "suele acompañar caídas y más volatilidad.",
        "bajo": "Baja incertidumbre: entorno más predecible, normalmente favorable para las acciones.",
    },
    "BAA10Y": {
        "titulo": "Spread corporativo Baa − Tesoro (estrés crediticio)",
        "que_es": "Cuánto más interés pagan las empresas con calificación Baa (el escalón más bajo de grado de "
                  "inversión) que el bono del Tesoro a 10 años, en puntos porcentuales. Es un termómetro clásico "
                  "del miedo en el mercado de crédito (Gilchrist y Zakrajšek, 2012).",
        "alto": "Spread alto: los inversionistas temen impagos; históricamente coincide con recesiones y "
                "mercados bajistas (2008, 2020).",
        "bajo": "Spread bajo: crédito barato y apetito por riesgo; entorno favorable, aunque niveles muy bajos "
                "pueden indicar complacencia.",
    },
    "T10Y2Y": {
        "titulo": "Pendiente de la curva de rendimientos (10 años − 2 años)",
        "que_es": "Diferencia entre la tasa del bono del Tesoro a 10 años y a 2 años. Normalmente es positiva "
                  "(prestar a más plazo paga más).",
        "alto": "Curva positiva y empinada: el mercado espera crecimiento; típico de recuperaciones.",
        "bajo": "Curva **invertida** (negativa): el mercado anticipa recortes de tasas por una desaceleración. "
                "Ha precedido a casi todas las recesiones de EE. UU. desde 1970, con 6 a 24 meses de anticipación.",
    },
    "DGS2": {
        "titulo": "Tasa del Tesoro a 2 años",
        "que_es": "Rendimiento del bono del gobierno de EE. UU. a 2 años. Resume lo que el mercado espera que haga "
                  "la Fed en los próximos dos años (Gürkaynak, Sack y Swanson, 2005). El modelo usa su cambio.",
        "alto": "Subiendo rápido: el mercado anticipa más alzas de la Fed; suele presionar a las acciones de "
                "crecimiento.",
        "bajo": "Bajando rápido: el mercado anticipa recortes, a menudo por miedo a una desaceleración.",
    },
    "DFII10": {
        "titulo": "Tasa real a 10 años (bonos TIPS)",
        "que_es": "Rendimiento del bono a 10 años **descontando la inflación esperada**. Es el costo real del dinero "
                  "a largo plazo.",
        "alto": "Tasa real alta: el dinero 'sin riesgo' rinde mucho, así que las acciones (sobre todo las de "
                "crecimiento y tecnología) valen menos en valor presente.",
        "bajo": "Tasa real baja o negativa: impulsa las valuaciones de las acciones y del oro.",
    },
}

METHOD = """
### 1. Qué problema resuelve

En términos de economía financiera, el modelo estima **retornos esperados de corte transversal**. En cada fecha
pregunta qué acciones de un mismo mercado rendirán más que las otras en los próximos *h* días hábiles
(*h* = 1, 5, 10 o 21). No intenta adivinar si el mercado completo sube o baja (el *timing* de mercado es mucho
más difícil). Busca la **prima relativa** entre acciones, que es la base de los fondos *long-short*.

### 2. Variable objetivo

Para cada acción *i* y fecha *t*, el objetivo es el **retorno logarítmico futuro**
`y = ln(P(t+h) / P(t))`. Se usan retornos y no precios porque los precios son no estacionarios. Un modelo que
"predice" el precio de mañana igual al de hoy parece muy preciso y no aporta nada; ese era el problema del LSTM
original.

### 3. Variables explicativas (*factores*)

Todas se calculan **sólo con información disponible al cierre de t**, sin mirar el futuro:

| Familia | Ejemplos | Intuición económica |
|---|---|---|
| **Momentum y reversión** | retorno a 1, 5, 21, 63 y 126 días; momentum relativo al mercado | Las ganadoras de 3 a 12 meses tienden a seguir ganando (Jegadeesh y Titman, 1993); a 1 semana hay reversión de corto plazo. |
| **Riesgo** | volatilidad 5/21/63 d, ATR, distancia al máximo anual | Anomalía de baja volatilidad: las acciones menos riesgosas rinden más de lo que el CAPM predice. |
| **Técnicos** | RSI, MACD, Bollinger, estocástico, volumen anormal | Presión de compra o venta de corto plazo y sobrerreacción. |
| **Macro y mercado** | S&P 500, IPC, VIX, tasa a 10 años, dólar, petróleo, tipo de cambio | Cada acción tiene distinta sensibilidad (beta) a estos choques; el modelo aprende quién gana en cada entorno. |
| **Política monetaria** | cambio de tasa de 40 bancos centrales, ciclo global, diferenciales con la Fed | Canal de tasas y de liquidez: el costo del dinero cambia el valor presente de los flujos y el apetito por riesgo. |
| **Estrés financiero** | spread *high yield*, curva 10a-2a, tasa real, incertidumbre de política | Indicadores adelantados de régimen: en estrés, las acciones defensivas superan a las cíclicas. |

Cada mercado **descarta automáticamente** las familias que no mejoran la predicción (ablación). Por ejemplo, en
México las tasas de bancos centrales no aportaron y se quitaron.

### 4. Modelo

Se usa **regresión cuantil con *gradient boosting*** (LightGBM) y, según el horizonte, un **ensamble** con una
regresión Ridge (lineal regularizada).
- **Panel agrupado**: un solo modelo para todas las acciones del mercado, igual que en Gu, Kelly y Xiu (2020).
  Hay muchas más observaciones que con un modelo por acción y se reduce el sobreajuste.
- **Árboles de decisión** que capturan interacciones no lineales. Por ejemplo, "el momentum sólo funciona
  cuando el VIX está bajo".
- **Cuantiles** p10, p50 y p90: en vez de un solo número se estima la distribución del retorno.
- Para cada mercado y horizonte se elige el modelo **barato** que mejor resultó en la validación. El LSTM se
  evaluó y se retiró: costaba unas 10 veces más y no fue mejor.

### 5. Validación (lo que hace confiable al sistema)

- **Walk-forward con purga**: se entrena con el pasado y se prueba con un periodo posterior que el modelo nunca
  vio, dejando *h* días de separación para que no se filtre información.
- **Prueba contra el azar (permutación)**: se barajan los pronósticos entre acciones 300 veces. El modelo debe
  superar a esos "modelos aleatorios" (p < 0.05).
- **Backtest con costos**: cada *h* días se compra el 20% superior y se vende en corto el 20% inferior,
  descontando 10 pb por operación y 50 pb anuales de préstamo de títulos.
- **Selección anidada** (*nested walk-forward*): cuando se elige entre varios modelos, la elección de cada
  periodo de prueba se hace sólo con los periodos anteriores. Lo que se reporta es el desempeño del
  *procedimiento de elección*, no del "mejor visto a posteriori".
- **Corrección por pruebas múltiples**: al probar muchas combinaciones (modelos × plazos), alguna sale bien por
  suerte (Harvey, Liu y Zhu, 2016). Se calcula el **Deflated Sharpe** (Bailey y López de Prado, 2014), que
  descuenta el número de intentos, y se exige que la probabilidad de ganancia real sea ≥ 95%.
- **Filtro de utilidad**: un plazo sólo genera señales si cumple todo lo anterior: ordena mejor que el azar, la
  estrategia gana (Sharpe ≥ 0.3) de forma significativa y sobrevive a la corrección por pruebas múltiples.
- **Bandas calibradas** (*conformal prediction*): se ajusta el ancho del rango para que el 80% de los retornos
  reales caiga dentro.

### 6. Limitaciones (léelas)

- **No predice la dirección del mercado.** En la validación ningún modelo superó a "siempre sube" para decir
  si una acción sube o baja. Lo que sí funciona es el **orden relativo**.
- **Ventaja pequeña**: un IC de 0.05 – 0.10 es bueno en finanzas, pero significa que el ranking acierta sólo un
  poco más de la mitad de las veces. La ganancia viene de repetirlo muchas veces y diversificar.
- **Sesgo de supervivencia**: las listas de acciones son las de hoy. Las empresas que quebraron o salieron del
  índice no están, lo que puede inflar un poco los resultados históricos. El universo dinámico por liquidez lo
  reduce, pero no lo elimina: el grupo de candidatas sigue siendo el S&P 500 y el Nasdaq-100 actuales.
- **Cambio de régimen**: lo que funcionó entre 2018 y 2026 puede dejar de funcionar. Por eso conviene volver a
  validar cada semana (lo hace la corrida de los lunes).
- **Universos pequeños** (12 y 26 acciones): pocas acciones por lado, así que los resultados son ruidosos.
- Uso educativo; **no es recomendación de inversión**.
"""

CONCEPTS = {
    "IC (coeficiente de información)": "Correlación (de rangos) entre el retorno pronosticado y el real entre "
        "las acciones de un mismo día, promediada en el tiempo. 0 = no ordena; 1 = ordena perfecto. En la "
        "industria, un IC sostenido de 0.03 – 0.10 se considera valioso.",
    "Sharpe": "Retorno promedio de la estrategia dividido entre su volatilidad, anualizado. Mide la ganancia por "
        "unidad de riesgo. Mayor que 0.5 es aceptable y mayor que 1 es muy bueno.",
    "Long-short": "Comprar las acciones con mejor pronóstico y vender en corto las de peor, con el mismo monto. Se "
        "gana si las compradas le ganan a las vendidas, suba o baje el mercado.",
    "Venta en corto": "Pedir prestada una acción, venderla hoy y recomprarla después. Se gana si el precio baja (o "
        "si sube menos que las acciones compradas en la estrategia long-short). Tiene costo de préstamo.",
    "p-valor": "Probabilidad de obtener un resultado igual de bueno por pura suerte. Menor que 0.05 se considera "
        "evidencia de que no es suerte.",
    "Mediana / p10 / p90": "Escenario central / pesimista / optimista. En 8 de cada 10 casos el resultado real "
        "cae entre p10 y p90.",
    "Walk-forward": "Validación que simula el uso real: entrenar con el pasado y probar con el futuro inmediato, "
        "avanzando en el tiempo.",
    "Punto base (pb)": "0.01%. Una subida de 25 pb en la tasa es +0.25 puntos porcentuales.",
    "Drawdown": "Caída desde el máximo anterior de la inversión. Mide el peor momento que habría que aguantar.",
    "Ablación": "Experimento que quita un grupo de variables y reentrena para medir cuánto aportaba.",
    "Deflated Sharpe": "Probabilidad de que el Sharpe verdadero sea positivo después de descontar cuántas "
        "configuraciones se probaron. Si pruebas 50 estrategias, la mejor tendrá un Sharpe alto aunque ninguna "
        "sirva; el Deflated Sharpe corrige eso.",
    "Selección anidada": "Elegir el modelo sólo con datos anteriores al periodo en que se evalúa, repitiendo el "
        "proceso en cada periodo. Evita el sesgo de quedarse con 'el que salió mejor'.",
    "Universo punto en el tiempo": "Lista de acciones que se habría podido elegir en cada fecha con la información "
        "de ese momento (aquí, las 50 más líquidas de los 6 meses previos), en vez de usar la lista de hoy.",
}
