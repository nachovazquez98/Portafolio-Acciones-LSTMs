# Marco de despliegue (preparado, **no activo**)

Hoy todo corre en local. Estos archivos dejan listo el camino a la nube para cuando se decida, sin
cambiar código. Nada aquí se ejecuta solo: el workflow está fuera de `.github/workflows/` a propósito.

| Archivo | Para qué | Costo |
|---|---|---|
| `Dockerfile` | Imagen única (pipeline + dashboard) para cualquier servidor | – |
| `docker-compose.yml` | Dashboard 24/7 + corrida diaria en un VPS / Oracle Always Free | $0–6/mes |
| `github-actions/daily.yml` | Corre el pipeline gratis en GitHub y publica artefactos en la rama `data` | $0 |
| `streamlit-cloud/requirements.txt` | Dashboard en línea en Streamlit Community Cloud | $0 |

## Ruta recomendada (cuando toque)

1. **Probar la imagen en local** (no requiere nube):
   ```bash
   docker build -f deploy/Dockerfile -t stockcast .
   docker run -p 8501:8501 -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts stockcast
   ```
2. **Pipeline en la nube gratis**: copiar `github-actions/daily.yml` a `.github/workflows/`. Los
   pronósticos quedan en la rama `data` del repo.
3. **Dashboard en línea gratis**: en share.streamlit.io conectar el repo, archivo `app/streamlit_app.py`.
   Falta un paso de código (pendiente, ~20 líneas): que la app descargue `artifacts/` de la rama `data`
   al arrancar cuando no exista localmente (p. ej. con la variable `STOCKCAST_ARTIFACTS_BRANCH=data`).
4. **Todo en un servidor** (alternativa a 2 + 3): `docker compose -f deploy/docker-compose.yml up -d`
   en Oracle Cloud Always Free o un VPS de ~US$5, con Tailscale o Cloudflare Tunnel para acceso privado.

## Consideraciones antes de activarlo

- **Yahoo Finance** a veces bloquea IPs de nube (GitHub Actions incluido). Si pasa, correr el pipeline
  en local y subir sólo `artifacts/`, o usar el VPS.
- **Repo público + Actions** = minutos ilimitados; repo privado tiene una cuota mensual gratuita.
- **Streamlit Community Cloud** publica la app; si el repo es privado se puede restringir el acceso.
- **Secretos** (p. ej. `ALPACA_API_KEY` si algún día se envían órdenes): en *Settings → Secrets* de
  GitHub / Streamlit, nunca en el repo.
- El LSTM no va en la nube (necesita GPU y no superó a los modelos baratos).
