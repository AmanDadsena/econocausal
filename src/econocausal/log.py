"""Logging setup shared by the pipeline and the API.

The level comes from the LOG_LEVEL environment variable (default INFO), e.g.
    LOG_LEVEL=DEBUG python -m econocausal.pipeline
"""

import logging
import os

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once. An unknown level name falls back to INFO."""
    name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric = logging.getLevelName(name)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    root = logging.getLogger()
    if not any(getattr(h, "_econocausal", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        handler._econocausal = True
        root.addHandler(handler)
    root.setLevel(numeric)
