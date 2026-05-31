"""Centralised logging configuration.

Provides a single :func:`get_logger` entry point so every module emits
structured, consistently formatted logs. The framework never uses ``print``;
all diagnostic output flows through the standard :mod:`logging` module, which
keeps the library import-safe and lets the host application control handlers,
levels, and sinks.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether the root configuration has run so repeated calls are idempotent.
_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    stream=sys.stdout,
) -> None:
    """Configure the root logger for the framework once.

    Parameters
    ----------
    level : int, optional
        Minimum severity to emit, by default :data:`logging.INFO`.
    fmt : str, optional
        Format string passed to :class:`logging.Formatter`.
    stream : IO, optional
        Output stream for the :class:`logging.StreamHandler`, by default
        :data:`sys.stdout`.

    Notes
    -----
    Calling this function more than once is a no-op after the first call to
    avoid attaching duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=_DATE_FORMAT))

    root = logging.getLogger("quantframework")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use.

    Parameters
    ----------
    name : str, optional
        Dotted module name. The framework namespace ``quantframework`` is
        prepended automatically if not already present.

    Returns
    -------
    logging.Logger
        A logger that inherits the framework's handler and level.
    """
    if not _CONFIGURED:
        configure_logging()

    if name is None:
        name = "quantframework"
    elif not name.startswith("quantframework"):
        name = f"quantframework.{name}"

    return logging.getLogger(name)
