"""Registro de experimentos: qué mercados siguen vivos, cuáles se descartaron y por qué.

Evita recalcular todo cada semana. La evaluación es por etapas (idea de *successive halving* / Hyperband:
Jamieson & Talwalkar 2016; Li et al. 2018 — cortar pronto lo que no promete):

  1. prueba rápida  : sólo Ridge + 100 permutaciones. ¿Ordena mejor que el azar en algún plazo?
                      Si no → "descartado" sin entrenar LightGBM ni hacer ablación.
  2. evaluación completa (compare + ablación) sólo para los que pasan la etapa 1.
  3. el filtro (gate) decide el nivel de evidencia por plazo. Sin ningún plazo con evidencia → "en pausa"
     (no se entrena a diario; se vuelve a evaluar cada semana si es núcleo).

Los descartados se re-prueban cada `gate.recheck_days` días (los mercados cambian), no cada semana.
El estado vive en artifacts/experiments.json.
"""
from __future__ import annotations

import json
from datetime import date, datetime

from stockcast import config

STATUS_LABEL = {"activo": "✅ activo", "pausa": "⏸️ suspendido", "descartado": "🗑️ descartado",
                "pendiente": "🕒 pendiente"}


def _path():
    return config.path("artifacts") / "experiments.json"


def load() -> dict:
    p = _path()
    return json.loads(p.read_text()) if p.exists() else {}


def save(reg: dict) -> None:
    _path().write_text(json.dumps(reg, indent=2, ensure_ascii=False, default=str))


def record(market_key: str, status: str, reason: str, stage: str, **extra) -> dict:
    reg = load()
    reg[market_key] = {"status": status, "reason": reason, "stage": stage,
                       "updated": datetime.now().isoformat(timespec="seconds"), **extra}
    save(reg)
    return reg[market_key]


def get(market_key: str) -> dict:
    return load().get(market_key, {"status": "pendiente", "reason": "aún no evaluado", "stage": "-"})


def due_for_evaluation(market_key: str) -> bool:
    """¿Toca evaluar este mercado? Núcleo y activos/pausados: sí. Descartados: sólo tras recheck_days."""
    e = load().get(market_key)
    if not e or e["status"] != "descartado":
        return True
    days = (date.today() - datetime.fromisoformat(e["updated"]).date()).days
    return days >= config.load()["gate"].get("recheck_days", 180)


def summary_table():
    import pandas as pd
    reg = load()
    rows = []
    for mk in config.markets():
        e = reg.get(mk, {"status": "pendiente", "reason": "aún no evaluado", "stage": "-", "updated": None})
        rows.append({"Mercado": config.load()["markets"][mk]["name"], "Estado": STATUS_LABEL.get(e["status"], e["status"]),
                     "Etapa": e.get("stage", "-"), "Motivo": e.get("reason", ""),
                     "Actualizado": (e.get("updated") or "")[:10]})
    return pd.DataFrame(rows)
