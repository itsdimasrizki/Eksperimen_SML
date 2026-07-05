"""
Inference helper utilities.

Author: Dimas Rizki
"""

from __future__ import annotations

import time

import numpy as np


# ==========================================================
# Timer
# ==========================================================


class Timer:
    """
    Simple execution timer.
    """

    def __init__(self) -> None:

        self._start = time.perf_counter()

    def elapsed(self) -> float:

        return (
            time.perf_counter()
            - self._start
        )


# ==========================================================
# Prediction Formatter
# ==========================================================


def format_prediction(
    prediction: np.ndarray,
) -> list[int]:
    """
    Convert prediction array to Python list.
    """

    return [
        int(value)
        for value in prediction
    ]


def format_probability(
    probability: np.ndarray,
    precision: int = 6,
) -> list[float]:
    """
    Convert probability array to rounded list.
    """

    return [
        round(float(value), precision)
        for value in probability
    ]


# ==========================================================
# Batch Helper
# ==========================================================


def is_batch_prediction(
    payload: dict | list[dict],
) -> bool:
    """
    Check whether payload is batch input.
    """

    return isinstance(
        payload,
        list,
    )