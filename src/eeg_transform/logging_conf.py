"""Configuración de logging del proyecto (consola + archivo)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LEVEL = logging.getLevelName("DEBUG")

_loggers_duplicates: set[str] = set()


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    format: str = _DEFAULT_FORMAT,
) -> None:
    """Configura el logging global.

    Parameters
    ----------
    level:
        Nivel de severidad ('DEBUG', 'INFO', ...).
    log_file:
        Ruta opcional a un archivo donde se volcarán los registros.
    format:
        Plantilla de mensaje.
    """
    root = logging.getLogger("eeg_transform")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root in _loggers_duplicates:
        return
    _loggers_duplicates.add(str(id(root)))

    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(logging.Formatter(format))
    root.addHandler(handler_console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler_file = logging.FileHandler(log_file, encoding="utf-8")
        handler_file.setFormatter(logging.Formatter(format))
        root.addHandler(handler_file)

    root.propagate = False


def get_logger(name: str = __name__) -> logging.Logger:
    """Devuelve un logger del paquete ``eeg_transform``."""
    return logging.getLogger(f"eeg_transform.{name}")