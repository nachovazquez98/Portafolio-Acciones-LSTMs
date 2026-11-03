# Próximos pasos: llevarlo a la nube pagando lo mínimo (cuando haga falta)

Hoy todo corre en local: los datos en `data/` (parquet), los modelos en `artifacts/`, el timer de systemd y el dashboard en `localhost:8501`. No hace falta AWS. Esta guía es para cuando quieras tenerlo disponible 24/7 o compartirlo.

> Verifica precios y límites antes de usarlos. Los free tiers cambian con frecuencia.

## Opción 0: seguir en local, pero accesible desde el celular ($0)
- **Tailscale** (gratis para uso personal): `tailscale up` en la laptop y abres `http://<laptop>:8501` desde el celular. No hay que abrir puertos.

## Opción 1: todo gratis con GitHub ($0)
1. **GitHub Actions** con `schedule: cron` ejecuta `stockcast run --market all --no-eval` cada día hábil. En repos públicos los minutos son ilimitados, y en privados hay una cuota mensual gratis. El pipeline LightGBM tarda 1–10 min en CPU.
2. Guarda `artifacts/*.parquet` como artifact del workflow o haz commit a una rama `data`.
3. Dashboard en **Streamlit Community Cloud** o **Hugging Face Spaces** (CPU gratis), leyendo los parquet de esa rama.
- Limitaciones: sin GPU (el LSTM queda fuera), los contenedores se duermen sin uso y Yahoo a veces bloquea IPs de nube.

## Opción 2: una VM gratis que replica tu laptop ($0)
- **Oracle Cloud Always Free (ARM Ampere)**: VM con varios núcleos y bastante RAM sin costo. Ahí se instala exactamente lo mismo: `make install && make schedule`, y el dashboard queda detrás de Tailscale o Cloudflare Tunnel (gratis).

## Opción 3: VPS barato (~US$4–6/mes)
- Hetzner, Contabo o DigitalOcean: 2 vCPU y 4 GB alcanzan para LightGBM con los mercados US + MX. El universo global (~1000 acciones) necesita unos 8 GB.

## Cuándo sí convendría AWS u otra nube grande
- Si necesitas GPU por horas (LSTM o modelos fundacionales sobre miles de acciones): instancias spot o servicios por segundo como Modal o RunPod, apagadas el resto del tiempo.
- Si operas dinero real con órdenes automáticas y necesitas alta disponibilidad.

## Para que la migración no duela (ya está preparado)
- Configuración en un solo YAML (`config/settings.yaml`) y rutas sobreescribibles con `STOCKCAST_ROOT` / `STOCKCAST_CONFIG`.
- Datos en parquet: se suben tal cual a S3, R2 o B2 si algún día hace falta. Para consultas SQL locales sin servidor, DuckDB lee los parquet directamente.
- Un solo comando idempotente (`stockcast run`) que cualquier scheduler puede llamar (systemd, cron, GitHub Actions, Prefect).

## Universo global (probado y desactivado)
`data/universe.py` arma automáticamente unas 1600 acciones de 20 bolsas (S&P 500, Nikkei 225, KOSPI 200, FTSE 100, DAX, CAC 40, NIFTY 50 y otras) a partir de Wikipedia. Se descargaron y funcionan, pero se desactivó a propósito por tres razones:
- multiplica el cómputo (unos 3 millones de filas);
- muchas bolsas no tienen todas las variables (banco central propio, índice y divisa);
- agrega complejidad sin ventaja demostrada.

Para reactivarlo, agrega un mercado con `universe: [sp500, nikkei225, ...]` en `config/settings.yaml`. El gate decide si vale la pena.

## Otras mejoras de modelo pendientes
- Medir la precisión real: `artifacts/<mercado>/forecast_history.parquet` guarda cada pronóstico. En unas semanas se puede comparar contra lo que pasó.
- Sentimiento como feature: se activa solo cuando `data/news.parquet` cubra ≥ 250 días.
- Seguimiento de experimentos con MLflow local (`mlflow ui`), sin servidor, si se empiezan a probar muchas configuraciones.
