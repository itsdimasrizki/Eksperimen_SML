"""
Production-ready FastAPI application for Heart Disease Prediction System.

This module exposes REST endpoints for model inference.

Author: Dimas Rizki
"""

from __future__ import annotations

import logging
import sys
import time

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse

from inference import InferenceService

# from prometheus_exporter import (
#     metrics_endpoint,
# )

# from monitoring.monitoring_helper import (
#     record_request,
#     register_model,
#     record_prediction,
#     record_error,
#     record_latency,
# )

# from monitoring.collectors import (
#     collect_system_metrics,
# )

from prometheus_exporter import (
    metrics_endpoint,
    record_request,
    register_model,
    record_prediction,
    record_error,
    record_latency,
    collect_system_metrics,
)

# ==========================================================
# Logging
# ==========================================================


def setup_logging() -> logging.Logger:
    """
    Configure application logger.
    """

    logger = logging.getLogger("serving")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()

# ==========================================================
# Global Service
# ==========================================================

service: InferenceService | None = None


# ==========================================================
# Application Lifecycle
# ==========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model once during startup.
    """

    global service

    logger.info("=" * 60)
    logger.info("STARTING PREDICTION SERVICE")
    logger.info("=" * 60)

    service = InferenceService()

    register_model(service.metadata)

    logger.info("Inference service initialized.")

    yield

    logger.info("=" * 60)
    logger.info("SHUTTING DOWN SERVICE")
    logger.info("=" * 60)


# ==========================================================
# FastAPI
# ==========================================================


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-ready inference API",
    version="1.0.0",
    lifespan=lifespan,
)
@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next,
):
    record_request()

    collect_system_metrics()

    response = await call_next(request)

    return response


# ==========================================================
# Health Check
# ==========================================================


@app.get("/")
def root():
    """
    Root endpoint.
    """

    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
    }


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    """

    return metrics_endpoint()


# ==========================================================
# Prediction Endpoint
# ==========================================================


@app.post("/predict")
def predict(payload: dict):
    """
    Predict a single sample or batch.
    """

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service not initialized.",
        )

    start_time = time.perf_counter()

    try:

        result = service.predict(payload)

        record_prediction()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        record_latency(elapsed)

        return JSONResponse(
            status_code=200,
            content=result,
        )

    except ValueError as error:

        record_error()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        record_latency(elapsed)

        logger.warning(str(error))

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:

        record_error()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        record_latency(elapsed)

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )