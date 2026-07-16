from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with structured console format: level, time, message."""
    global _CONFIGURED

    if not _CONFIGURED:
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(levelname)s %(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(handler)
            root.setLevel(logging.INFO)
        _CONFIGURED = True

    return logging.getLogger(name)
