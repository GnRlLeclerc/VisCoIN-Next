"""Utility math functions"""

import numpy as np


def normalize(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalize an array between 0 and 1.
    If dim=None, normalize the whole array."""

    return (arr - arr.min(axis=axis)) / (arr.max(axis=axis) - arr.min(axis=axis))
