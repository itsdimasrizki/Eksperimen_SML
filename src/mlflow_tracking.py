"""
Production-ready MLflow tracking pipeline for Heart Disease Prediction System.

This module logs baseline and tuned model experiments into MLflow
without retraining existing models.

Outputs
-------
- MLflow Experiments
- MLflow Model Registry
- logs/mlflow.log

Author: Dimas Rizki
"""

from __future__ import annotations

import json
import logging
import sys
import time

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from mlflow.models.signature import infer_signature

from mlflow.tracking import MlflowClient

import joblib
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "mlflow.log"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

TRACKING_URI = f"sqlite:///{MLFLOW_DB.resolve()}"

TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.resolve()}"
)

EXPERIMENT_NAME = "heart_disease_prediction"

REGISTERED_MODEL_NAME = "heart_disease_model"

BASELINE_RUN_NAME = "baseline_model"

TUNED_RUN_NAME = "tuned_model"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

BASELINE_METADATA_PATH = (
    MODEL_DIR / "model_metadata.json"
)

BEST_METADATA_PATH = (
    MODEL_DIR / "best_model_metadata.json"
)

BASELINE_METRICS_PATH = (
    ARTIFACT_DIR / "metrics.json"
)

TUNED_METRICS_PATH = (
    ARTIFACT_DIR / "tuning_metrics.json"
)

RANDOM_STATE = 42

# ==========================================================
# Logging
# ==========================================================


def setup_logging() -> logging.Logger:
    """
    Configure MLflow logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "mlflow_tracking"
    )

    if logger.handlers:
        return logger

    logger.setLevel(
        logging.INFO,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
    )

    file_handler.setFormatter(
        formatter,
    )

    stream_handler = logging.StreamHandler(
        sys.stdout,
    )

    stream_handler.setFormatter(
        formatter,
    )

    logger.addHandler(
        file_handler,
    )

    logger.addHandler(
        stream_handler,
    )

    return logger


logger = setup_logging()

# ==========================================================
# Directory Utilities
# ==========================================================


def initialize_directories() -> None:
    """
    Create required project directories.
    """

    directories = (
        LOG_DIR,
    )

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info(
        "Project directories initialized."
    )

# ==========================================================
# Generic Helper
# ==========================================================


def validate_file_exists(
    path: Path,
) -> None:
    """
    Validate file existence.

    Parameters
    ----------
    path:
        File path to validate.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load JSON file.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict
    """

    validate_file_exists(path)

    logger.info(
        "Loading JSON: %s",
        path.name,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save dictionary into JSON file.

    Parameters
    ----------
    data:
        Dictionary to save.

    output_path:
        Destination path.
    """

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved JSON: %s",
        output_path.name,
    )


def load_dataframe(
    path: Path,
) -> pd.DataFrame:
    """
    Load CSV file.

    Parameters
    ----------
    path:
        CSV file path.

    Returns
    -------
    pd.DataFrame
    """

    validate_file_exists(path)

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"{path.name} is empty."
        )

    logger.info(
        "Loaded DataFrame: %s %s",
        path.name,
        dataframe.shape,
    )

    return dataframe


def load_artifact(
    path: Path,
) -> Any:
    """
    Load serialized artifact.

    Parameters
    ----------
    path:
        Artifact path.

    Returns
    -------
    Any
    """

    validate_file_exists(path)

    logger.info(
        "Loading artifact: %s",
        path.name,
    )

    return joblib.load(path)


def get_file_size(
    path: Path,
) -> float:
    """
    Get file size in MB.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    float
    """

    validate_file_exists(path)

    size = (
        path.stat().st_size
        / (1024 * 1024)
    )

    return round(
        size,
        2,
    )


def log_file_information(
    path: Path,
) -> None:
    """
    Log file information.

    Parameters
    ----------
    path:
        File path.
    """

    logger.info(
        "File : %s | %.2f MB",
        path.name,
        get_file_size(path),
    )

# ==========================================================
# MLflow Helper
# ==========================================================


def initialize_mlflow() -> None:
    """
    Initialize MLflow tracking configuration.
    """

    logger.info(
        "Initializing MLflow..."
    )

    mlflow.set_tracking_uri(
        TRACKING_URI,
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME,
    )

    logger.info(
        "Tracking URI : %s",
        TRACKING_URI,
    )

    logger.info(
        "Experiment : %s",
        EXPERIMENT_NAME,
    )


def set_common_tags(
    model_version: str,
) -> None:
    """
    Log common MLflow tags.

    Parameters
    ----------
    model_version:
        baseline or tuned.
    """

    mlflow.set_tags(
        {
            "project": "Heart Disease Prediction",

            "author": "Dimas Rizki",

            "framework": "Scikit-Learn",

            "problem_type": "Binary Classification",

            "dataset": "Heart Disease Dataset",

            "model_version": model_version,

            "tracking": "MLflow",
        }
    )


def log_parameters(
    parameters: dict[str, Any],
) -> None:
    """
    Log model parameters.

    Parameters
    ----------
    parameters:
        Dictionary of parameters.
    """

    if not parameters:
        return

    mlflow.log_params(
        parameters,
    )

    logger.info(
        "Logged %d parameters.",
        len(parameters),
    )


def log_metrics(
    metrics: dict[str, float],
) -> None:
    """
    Log evaluation metrics.

    Parameters
    ----------
    metrics:
        Metric dictionary.
    """

    if not metrics:
        return

    mlflow.log_metrics(
        metrics,
    )

    logger.info(
        "Logged %d metrics.",
        len(metrics),
    )


def log_artifact(
    path: Path,
    artifact_path: str,
) -> None:
    """
    Log a single artifact.

    Parameters
    ----------
    path:
        Local artifact path.

    artifact_path:
        MLflow artifact folder.
    """

    validate_file_exists(path)

    mlflow.log_artifact(
        local_path=str(path),
        artifact_path=artifact_path,
    )

    logger.info(
        "Logged artifact: %s",
        path.name,
    )


def log_artifact_directory(
    paths: list[Path],
    artifact_path: str,
) -> None:
    """
    Log multiple artifacts.

    Parameters
    ----------
    paths:
        List of artifact paths.

    artifact_path:
        MLflow artifact directory.
    """

    for path in paths:

        log_artifact(
            path,
            artifact_path,
        )


def log_model(
    model: Any,
    X_sample: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Log sklearn model.

    Parameters
    ----------
    model:
        Trained model.

    X_sample:
        Sample dataframe.

    model_name:
        Artifact name.
    """

    signature = infer_signature(
        X_sample,
        model.predict(X_sample),
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        signature=signature,
        input_example=X_sample.head(5),
    )

    logger.info(
        "Logged model: %s",
        model_name,
    )

# ==========================================================
# Dataset & Model Loader
# ==========================================================


def load_processed_dataset() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Load processed datasets.

    Returns
    -------
    tuple
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    """

    logger.info(
        "Loading processed datasets..."
    )

    X_train = load_dataframe(
        PROCESSED_DIR / "X_train.csv"
    )

    X_val = load_dataframe(
        PROCESSED_DIR / "X_val.csv"
    )

    X_test = load_dataframe(
        PROCESSED_DIR / "X_test.csv"
    )

    y_train = load_dataframe(
        PROCESSED_DIR / "y_train.csv"
    ).iloc[:, 0]

    y_val = load_dataframe(
        PROCESSED_DIR / "y_val.csv"
    ).iloc[:, 0]

    y_test = load_dataframe(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    logger.info(
        "Processed datasets loaded successfully."
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ==========================================================
# Baseline Loader
# ==========================================================


def load_baseline_model() -> Any:
    """
    Load baseline model.
    """

    return load_artifact(
        BASELINE_MODEL_PATH,
    )


def load_baseline_metrics() -> dict[str, Any]:
    """
    Load baseline metrics.
    """

    return load_json(
        BASELINE_METRICS_PATH,
    )


def load_baseline_metadata() -> dict[str, Any]:
    """
    Load baseline metadata.
    """

    return load_json(
        BASELINE_METADATA_PATH,
    )


# ==========================================================
# Tuned Loader
# ==========================================================


def load_best_model() -> Any:
    """
    Load tuned model.
    """

    return load_artifact(
        BEST_MODEL_PATH,
    )


def load_tuned_metrics() -> dict[str, Any]:
    """
    Load tuned metrics.
    """

    return load_json(
        TUNED_METRICS_PATH,
    )


def load_best_metadata() -> dict[str, Any]:
    """
    Load tuned model metadata.
    """

    return load_json(
        BEST_METADATA_PATH,
    )


# ==========================================================
# Parameter Helper
# ==========================================================


def extract_baseline_parameters(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract baseline parameters.
    """

    parameters = {
        "model_name": metadata.get(
            "model_name",
        ),
        "version": metadata.get(
            "version",
        ),
        "random_state": metadata.get(
            "random_state",
        ),
        "training_features": metadata.get(
            "training_features",
        ),
        "training_samples": metadata.get(
            "training_samples",
        ),
        "validation_samples": metadata.get(
            "validation_samples",
        ),
        "test_samples": metadata.get(
            "test_samples",
        ),
    }

    return {
        key: value
        for key, value in parameters.items()
        if value is not None
    }


def extract_tuned_parameters(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract tuned parameters.
    """

    parameters = {
        "model_name": metadata.get(
            "model_name",
        ),
        "version": metadata.get(
            "version",
        ),
        "random_state": metadata.get(
            "random_state",
        ),
        "cv_best_score": metadata.get(
            "cv_best_score",
        ),
    }

    best_params = metadata.get(
        "best_parameters",
        {},
    )

    parameters.update(
        best_params,
    )

    return {
        key: value
        for key, value in parameters.items()
        if value is not None
    }

def collect_tuned_artifacts() -> list[Path]:
    """
    Collect tuned artifacts.

    Returns
    -------
    list[Path]
    """

    artifacts = [
        TUNED_METRICS_PATH,
        BEST_METADATA_PATH,
        ARTIFACT_DIR /
        "tuning_classification_report.txt",
        ARTIFACT_DIR /
        "tuning_confusion_matrix.png",
        ARTIFACT_DIR /
        "tuning_roc_curve.png",
    ]

    return [
        artifact
        for artifact in artifacts
        if artifact.exists()
    ]

# ==========================================================
# Metadata Helper
# ==========================================================


def log_metadata_as_tags(
    metadata: dict[str, Any],
) -> None:
    """
    Log selected metadata as MLflow tags.

    Parameters
    ----------
    metadata:
        Model metadata dictionary.
    """

    ignored = {
        "feature_names",
        "train_shape",
        "validation_shape",
        "test_shape",
        "best_parameters",
    }

    for key, value in metadata.items():

        if key in ignored:
            continue

        if isinstance(
            value,
            (list, dict),
        ):
            continue

        mlflow.set_tag(
            key,
            value,
        )

    logger.info(
        "Metadata tags logged."
    )

# ==========================================================
# Artifact Helper
# ==========================================================


def collect_baseline_artifacts() -> list[Path]:
    """
    Collect baseline artifacts.

    Returns
    -------
    list[Path]
    """

    artifacts = [
        BASELINE_METRICS_PATH,
        BASELINE_METADATA_PATH,
        ARTIFACT_DIR / "classification_report.txt",
        ARTIFACT_DIR / "confusion_matrix.png",
        ARTIFACT_DIR / "roc_curve.png",
    ]

    return [
        artifact
        for artifact in artifacts
        if artifact.exists()
    ]

def log_baseline_artifacts() -> None:
    """
    Log all baseline artifacts.
    """

    logger.info(
        "Logging baseline artifacts..."
    )

    artifacts = collect_baseline_artifacts()

    log_artifact_directory(
        paths=artifacts,
        artifact_path="baseline",
    )

    logger.info(
        "Baseline artifacts logged."
    )


def log_tuned_artifacts() -> None:
    """
    Log all tuned artifacts.
    """

    logger.info(
        "Logging tuned artifacts..."
    )

    artifacts = collect_tuned_artifacts()

    log_artifact_directory(
        paths=artifacts,
        artifact_path="tuned",
    )

    logger.info(
        "Tuned artifacts logged."
    )

def log_model_summary(
    model_path: Path,
) -> None:
    """
    Log model file information.

    Parameters
    ----------
    model_path:
        Serialized model path.
    """

    validate_file_exists(
        model_path,
    )

    mlflow.log_metric(
        "model_size_mb",
        get_file_size(model_path),
    )

    logger.info(
        "Model size : %.2f MB",
        get_file_size(model_path),
    )

# ==========================================================
# Baseline Tracking
# ==========================================================


def log_baseline_run(
    X_sample: pd.DataFrame,
) -> None:
    """
    Log baseline model into MLflow.
    """

    logger.info("=" * 60)
    logger.info("LOGGING BASELINE MODEL")
    logger.info("=" * 60)

    model = load_baseline_model()

    metrics = load_baseline_metrics()

    metadata = load_baseline_metadata()

    parameters = extract_baseline_parameters(
        metadata,
    )

    with mlflow.start_run(
        run_name=BASELINE_RUN_NAME,
    ):

        set_common_tags(
            "baseline",
        )

        log_metadata_as_tags(
            metadata,
        )

        log_parameters(
            parameters,
        )

        log_metrics(
            metrics,
        )

        log_baseline_artifacts()

        log_model_summary(
            BASELINE_MODEL_PATH,
        )

        log_model(
            model=model,
            X_sample=X_sample,
            model_name="baseline_model",
        )

    logger.info(
        "Baseline tracking completed."
    )

# ==========================================================
# Tuned Tracking
# ==========================================================


def log_tuned_run(
    X_sample: pd.DataFrame,
) -> None:
    """
    Log tuned model into MLflow.

    Parameters
    ----------
    X_sample:
        Sample dataframe used for model signature.
    """

    logger.info("=" * 60)
    logger.info("LOGGING TUNED MODEL")
    logger.info("=" * 60)

    model = load_best_model()

    metrics = load_tuned_metrics()

    metadata = load_best_metadata()

    parameters = extract_tuned_parameters(
        metadata,
    )

    with mlflow.start_run(
        run_name=TUNED_RUN_NAME,
    ):

        set_common_tags(
            "tuned",
        )

        log_metadata_as_tags(
            metadata,
        )

        log_parameters(
            parameters,
        )

        log_metrics(
            metrics,
        )

        log_tuned_artifacts()

        log_model_summary(
            BEST_MODEL_PATH,
        )

        log_model(
            model=model,
            X_sample=X_sample,
            model_name="best_model",
        )

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

    logger.info(
        "Tuned tracking completed."
    )

    register_best_model(
        run_id,
    )


# ==========================================================
# Model Registry
# ==========================================================


def register_best_model(
    run_id: str,
) -> None:
    """
    Register best model into MLflow Model Registry.

    Parameters
    ----------
    run_id:
        MLflow run identifier.
    """

    logger.info(
        "Registering best model..."
    )

    model_uri = (
        f"runs:/{run_id}/best_model"
    )

    try:

        registered_model = (
            mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME,
            )
        )

        client = MlflowClient()

        latest_version = (
            registered_model.version
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "production",
            latest_version,
        )

        logger.info(
            "Registered model version %s",
            latest_version,
        )

    except Exception as error:

        logger.warning(
            "Model registration skipped: %s",
            error,
        )

# ==========================================================
# Pipeline
# ==========================================================


def run_pipeline() -> None:
    """
    Execute complete MLflow tracking pipeline.
    """

    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("STARTING MLFLOW TRACKING")
    logger.info("=" * 60)

    initialize_directories()

    initialize_mlflow()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_processed_dataset()

    logger.info(
        "Train shape : %s",
        X_train.shape,
    )

    logger.info(
        "Validation shape : %s",
        X_val.shape,
    )

    logger.info(
        "Test shape : %s",
        X_test.shape,
    )

    logger.info(
        "Tracking baseline experiment..."
    )

    log_baseline_run(
        X_sample=X_train,
    )

    logger.info(
        "Tracking tuned experiment..."
    )

    log_tuned_run(
        X_sample=X_train,
    )

    logger.info("=" * 60)
    logger.info("MLFLOW TRACKING FINISHED")
    logger.info("=" * 60)

    elapsed = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Pipeline finished in %.2f seconds.",
        elapsed,
    )


# ==========================================================
# CLI
# ==========================================================


def main() -> None:
    """
    Application entry point.
    """

    try:

        run_pipeline()

    except KeyboardInterrupt:

        logger.warning(
            "MLflow tracking interrupted by user."
        )

    except Exception:

        logger.exception(
            "Unexpected error occurred."
        )

        raise


if __name__ == "__main__":

    main()