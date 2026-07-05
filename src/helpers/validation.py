"""
Input validation helper for inference.

Author: Dimas Rizki
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ==========================================================
# Expected Features
# ==========================================================

EXPECTED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

INTEGER_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


# ==========================================================
# Validation
# ==========================================================


def validate_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Validate request payload and return DataFrame.
    """

    if isinstance(payload, dict):

        dataframe = pd.DataFrame([payload])

    elif isinstance(payload, list):

        if not payload:
            raise ValueError(
                "Payload list cannot be empty."
            )

        dataframe = pd.DataFrame(payload)

    else:

        raise ValueError(
            "Payload must be a dictionary or list of dictionaries."
        )

    validate_columns(dataframe)

    validate_missing_values(dataframe)

    validate_numeric_types(dataframe)

    return dataframe


# ==========================================================
# Column Validation
# ==========================================================


def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate input schema.
    """

    actual = set(dataframe.columns)

    expected = set(EXPECTED_FEATURES)

    missing = sorted(
        expected - actual
    )

    extra = sorted(
        actual - expected
    )

    if missing:

        raise ValueError(
            f"Missing feature(s): {missing}"
        )

    if extra:

        raise ValueError(
            f"Unknown feature(s): {extra}"
        )


# ==========================================================
# Missing Values
# ==========================================================


def validate_missing_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Reject missing values.
    """

    if dataframe.isnull().values.any():

        raise ValueError(
            "Input contains missing values."
        )


# ==========================================================
# Datatype Validation
# ==========================================================


def validate_numeric_types(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate numeric datatype.
    """

    for column in NUMERIC_COLUMNS:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    for column in INTEGER_COLUMNS:

        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="raise",
            )
            .astype(int)
        )


# ==========================================================
# Column Order
# ==========================================================


def reorder_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure feature order matches training.
    """

    return dataframe[
        EXPECTED_FEATURES
    ]