from __future__ import annotations

from prometheus_client import Counter, Histogram, Info, Gauge

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

MODEL_INFO = Info(
    "model",
    "Loaded model metadata.",
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