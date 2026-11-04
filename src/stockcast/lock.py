"""Candado entre procesos para corridas pesadas (CLI, timer diario y botón del dashboard)."""
from __future__ import annotations

import logging

from stockcast import config


class RunLock:
    """Candado entre procesos: evita que dos corridas pesadas (p. ej. el timer diario y una manual) compitan
    por los núcleos, lo que hace que LightGBM se vuelva decenas de veces más lento."""

    def __init__(self):
        self.path = config.path("data") / ".stockcast.lock"
        self.fh = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:  # Windows: sin candado
            return self
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.getLogger("stockcast").info("Otra corrida de stockcast está en curso; esperando a que termine…")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()
