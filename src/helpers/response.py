"""
Response helper for inference API.

Author: Dimas Rizki
"""

from __future__ import annotations

from typing import Any


def success_response(
    *,
    prediction: list[int],
    probability: list[float],
    model: str,
    version: str,
    processing_time: float,
    batch: bool,
) -> dict[str, Any]:
    """
    Build success response.
    """

    return {
        "success": True,
        "batch_prediction": batch,
        "model": model,
        "version": version,
        "prediction": prediction,
        "probability": [
            round(value, 6)
            for value in probability
        ],
        "processing_time_seconds": round(
            processing_time,
            4,
        ),
    }


def error_response(
    message: str,
    detail: str | None = None,
) -> dict[str, Any]:
    """
    Build error response.
    """

    response = {
        "success": False,
        "message": message,
    }

    if detail:

        response["detail"] = detail

    return response