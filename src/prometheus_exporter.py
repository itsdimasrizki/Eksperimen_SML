from __future__ import annotations

from fastapi.responses import PlainTextResponse

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)


def metrics_endpoint():
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )