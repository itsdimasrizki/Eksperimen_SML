from __future__ import annotations

from fastapi.responses import PlainTextResponse

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests received.",
)

PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total prediction requests.",
)

ERROR_COUNTER = Counter(
    "prediction_errors_total",
    "Total prediction errors.",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency.",
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage.",
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory usage percentage.",
)

DISK_USAGE = Gauge(
    "disk_usage_percent",
    "Disk usage percentage.",
)

import psutil

MODEL_INFO = Gauge(
    "model_info",
    "Loaded model metadata.",
    ["name", "version"],
)


def record_request():
    REQUEST_COUNTER.inc()


def record_prediction():
    PREDICTION_COUNTER.inc()


def record_error():
    ERROR_COUNTER.inc()


def record_latency(seconds: float):
    PREDICTION_LATENCY.observe(seconds)


def register_model(metadata: dict):
    MODEL_INFO.labels(
        name=metadata.get("model_name", "unknown"),
        version=metadata.get("model_version", "unknown"),
    ).set(1)

def collect_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent())

    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.percent)

    disk = psutil.disk_usage("/")
    DISK_USAGE.set(disk.percent)

def metrics_endpoint():
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )