from __future__ import annotations

from monitoring.metrics import (
    REQUEST_COUNTER,
    PREDICTION_COUNTER,
    ERROR_COUNTER,
    PREDICTION_LATENCY,
    MODEL_INFO,
)


def record_request() -> None:
    REQUEST_COUNTER.inc()


def record_prediction() -> None:
    PREDICTION_COUNTER.inc()


def record_error() -> None:
    ERROR_COUNTER.inc()


def record_latency(seconds: float) -> None:
    PREDICTION_LATENCY.observe(seconds)


def register_model(metadata: dict) -> None:
    MODEL_INFO.info(
        {
            "name": metadata.get(
                "model_name",
                "Unknown",
            ),
            "version": metadata.get(
                "version",
                "Unknown",
            ),
        }
    )