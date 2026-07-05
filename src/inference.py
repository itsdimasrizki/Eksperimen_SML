"""
Production-ready inference pipeline for Heart Disease Prediction System.

Author: Dimas Rizki
"""

from __future__ import annotations

import logging
import sys
import time

import pandas as pd

from helpers.loader import (
    load_model,
    load_preprocessor,
    load_metadata,
)

from helpers.validation import (
    validate_payload,
    reorder_columns,
)

from helpers.response import (
    success_response,
)

from helpers.inference_helper import (
    Timer,
    format_prediction,
    format_probability,
    is_batch_prediction,
)

# ==========================================================
# Logging
# ==========================================================


def setup_logging() -> logging.Logger:

    logger = logging.getLogger("inference")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger = setup_logging()

# ==========================================================
# Inference Service
# ==========================================================


class InferenceService:

    """
    Production inference service.
    """

    def __init__(self):
        logger.info("Loading model...")

        self.model = load_model()

        logger.info("Loading preprocessor...")

        self.preprocessor = (
            load_preprocessor()
        )

        logger.info("Loading metadata...")

        self.metadata = (
            load_metadata()
        )

        logger.info(
        "Inference service ready."
    )

    def predict(
        self,
        payload: dict | list[dict],
    ) -> dict:
        """
        Execute inference pipeline.
        """

        timer = Timer()

        try:

            dataframe = validate_payload(
                payload
            )

            dataframe = reorder_columns(
                dataframe
            )

            transformed = (
                self.preprocessor.transform(
                    dataframe
                )
            )

            feature_names = self.metadata.get(
                "feature_names"
            )

            if feature_names is not None:

                transformed = pd.DataFrame(
                    transformed,
                    columns=feature_names,
                )

            prediction = self.model.predict(
                transformed
            )

            probability = (
                self.model.predict_proba(
                    transformed
                )[:, 1]
            )

            elapsed = timer.elapsed()

            logger.info(
                "Prediction completed in %.4f seconds.",
                elapsed,
            )

            batch = is_batch_prediction(
                payload
            )

            return success_response(
                prediction=format_prediction(
                    prediction
                ),
                probability=format_probability(
                    probability
                ),
                model=self.metadata.get(
                    "model_name",
                    "Unknown",
                ),
                version=self.metadata.get(
                    "version",
                    "Unknown",
                ),
                processing_time=elapsed,
                batch=batch,
            )

        except Exception:

            logger.exception(
                "Inference failed."
            )

            raise