"""
Artifact loader helper.

Author: Dimas Rizki
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

PREPROCESSOR_PATH = (
    ARTIFACT_DIR / "preprocessor.pkl"
)

METADATA_PATH = (
    MODEL_DIR / "best_model_metadata.json"
)


# ==========================================================
# Generic Loader
# ==========================================================


def load_artifact(
    path: Path,
) -> Any:
    """
    Load serialized artifact.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    return joblib.load(path)


def load_json(
    path: Path,
) -> dict:
    """
    Load JSON metadata.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Public API
# ==========================================================


def load_model():
    """
    Load production model.
    """

    return load_artifact(
        BEST_MODEL_PATH
    )


def load_preprocessor():
    """
    Load fitted preprocessor.
    """

    return load_artifact(
        PREPROCESSOR_PATH
    )


def load_metadata():
    """
    Load model metadata.
    """

    return load_json(
        METADATA_PATH
    )