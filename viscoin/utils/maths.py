"""Utility math functions"""

import numpy as np


def normalize(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalize an array between 0 and 1.
    If dim=None, normalize the whole array."""

    min = arr.min(axis=axis, keepdims=True)
    max = arr.max(axis=axis, keepdims=True)

    return (arr - min) / (max - min)
