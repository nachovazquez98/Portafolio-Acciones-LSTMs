#!/usr/bin/env bash
# Ejecución diaria local: actualiza datos, reentrena y pronostica ambos mercados.
# Lo llama el timer de systemd o cron. Los logs quedan en logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/run_$(date +%F).log"
# Diario: actualiza datos y reentrena SÓLO los mercados activos y SÓLO sus plazos con evidencia.
# Lunes: además, evaluación por etapas (prueba rápida → completa) que descarta lo que deja de funcionar.
.venv/bin/stockcast data >>"$LOG" 2>&1
if [[ "$(date +%u)" == "1" ]]; then
  .venv/bin/stockcast evaluate --market all >>"$LOG" 2>&1
fi
.venv/bin/stockcast news --market all >>"$LOG" 2>&1   # acumula histórico de noticias (contexto / futuro feature)
.venv/bin/stockcast run --market all --no-eval --offline --no-news >>"$LOG" 2>&1
find logs -name 'run_*.log' -mtime +30 -delete
