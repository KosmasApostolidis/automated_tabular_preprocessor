"""Module-level logger factory. Library never configures the root logger."""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
