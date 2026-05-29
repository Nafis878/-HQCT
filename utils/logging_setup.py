"""
utils/logging_setup.py -- Structured logging for the HQCT pipeline.

Replaces print() statements with a proper logging hierarchy:
  - File handler  → logs/experiment_{timestamp}.log
  - Stream handler → console (INFO level)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


_loggers: dict = {}


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with file + stream handlers.
    Subsequent calls with the same name return the cached logger.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        _loggers[name] = logger
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_path / f"experiment_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _loggers[name] = logger
    return logger
