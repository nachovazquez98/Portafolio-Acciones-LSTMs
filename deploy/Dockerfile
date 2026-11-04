# Imagen única para el pipeline y el dashboard (CPU; el LSTM no se incluye).
# Construir:  docker build -f deploy/Dockerfile -t stockcast .
# Dashboard:  docker run -p 8501:8501 -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts stockcast
# Pipeline:   docker run -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts stockcast stockcast run --market all
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STOCKCAST_ROOT=/app
WORKDIR /app

# libgomp: requerido por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY app ./app
COPY .streamlit ./.streamlit

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
