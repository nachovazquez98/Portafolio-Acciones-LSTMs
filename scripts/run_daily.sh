#!/usr/bin/env bash
# Ejecución diaria local: actualiza datos, reentrena y pronostica ambos mercados.
# Lo llama el timer de systemd o cron. Los logs quedan en logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/run_$(date +%F).log"
# Validación walk-forward completa sólo los lunes (es lo más costoso); el resto de días sólo reentrena.
EVAL_FLAG="--no-eval"
[[ "$(date +%u)" == "1" ]] && EVAL_FLAG=""
.venv/bin/stockcast run --market all $EVAL_FLAG >>"$LOG" 2>&1
find logs -name 'run_*.log' -mtime +30 -delete
