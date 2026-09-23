"""Standard-library logging setup shared across the package.

Deliberately stdlib-only (no `structlog`/`loguru` dependency) to keep the core install
minimal, per the research spec's requirement (§15) that installing the library not
require the whole research environment. Downstream consumers who want structured JSON
logs can attach their own handler to the `corefinity_adaptive` logger.
"""

from __future__ import annotations

import logging

_LOGGER_NAME = "corefinity_adaptive"


def get_logger(component: str | None = None) -> logging.Logger:
    name = _LOGGER_NAME if component is None else f"{_LOGGER_NAME}.{component}"
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler with a consistent format, idempotently."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
