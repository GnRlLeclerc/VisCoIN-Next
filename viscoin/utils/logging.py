"""Logging utilities to follow the training of a model."""

import logging


def get_logger():
    """Returns the current scope logger"""
    return logging.getLogger(__name__)


def configure_score_logging(log_path: str):
    """Configure logging to a file by appending lines.
    The file will be overwritten if it already exists.

    Args:
        log_path (str): Path to the log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
