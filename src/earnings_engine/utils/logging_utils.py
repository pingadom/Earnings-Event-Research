"""Minimal, dependency-free logging setup shared by the CLI and library code."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(level: int | str | None = None, *, quiet: bool = False) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        if level is not None:
            logging.getLogger().setLevel(_as_level(level))
        return
    if level is None:
        level = os.environ.get("EEE_LOG_LEVEL", "WARNING" if quiet else "INFO")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(_as_level(level))
    _CONFIGURED = True


def _as_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
