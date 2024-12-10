"""Logging utilities to follow the training of a model."""

import logging

from viscoin.utils.types import TestingResults


def get_logger():
    """Returns the current scope logger"""
    return logging.getLogger(__name__)


def configure_score_logging(log_path: str):
    """Configure logging to a file by appending lines.
    The file will be overwritten if it already exists.

    Args:
        log_path (str): Path to the log file.
    """
    logging.basicConfig(level=logging.INFO, filemode="w", format="%(message)s", filename=log_path)


def parse_viscoin_training_logs(filename: str) -> list[TestingResults]:
    # Import this for the eval function
    import numpy as np  # type: ignore

    results: list[TestingResults] = []

    with open(filename, "r") as f:

        for line in f:
            # Skip lines that are not about testing results
            if not line.startswith("TestingResults"):
                continue

            results.append(eval(line))

    return results
