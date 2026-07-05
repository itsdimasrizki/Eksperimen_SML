from __future__ import annotations

import psutil

from monitoring.metrics import (
    CPU_USAGE,
    MEMORY_USAGE,
    DISK_USAGE,
)


def collect_system_metrics() -> None:
    CPU_USAGE.set(psutil.cpu_percent())

    MEMORY_USAGE.set(
        psutil.virtual_memory().percent
    )

    DISK_USAGE.set(
        psutil.disk_usage("/").percent
    )