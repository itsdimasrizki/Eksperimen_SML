"""
Production-ready hyperparameter tuning pipeline for Heart Disease Prediction
System.

This module performs hyperparameter tuning using RandomizedSearchCV and
selects the best estimator for production deployment.

Outputs
-------
- best_model.pkl
- best_model_metadata.json
- tuning_metrics.json
- tuning_classification_report.txt
- tuning_confusion_matrix.png
- tuning_roc_curve.png
- model_comparison.csv
- tuning.log

Author: Dimas Rizki
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

REPORT_DIR = PROJECT_ROOT / "reports"

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "tuning.log"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

BEST_METADATA_PATH = MODEL_DIR / "best_model_metadata.json"

MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"

METRICS_PATH = ARTIFACT_DIR / "tuning_metrics.json"

CLASSIFICATION_REPORT_PATH = ARTIFACT_DIR / "tuning_classification_report.txt"

CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "tuning_confusion_matrix.png"

ROC_CURVE_PATH = ARTIFACT_DIR / "tuning_roc_curve.png"

RANDOM_STATE = 42

N_ITER = 30

CV_SPLITS = 5

MODEL_NAME = "RandomForestClassifier"

# ==========================================================
# Logging
# ==========================================================


def setup_logging() -> logging.Logger:

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("tuning")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE)

    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)

    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()

# ==========================================================
# Directory
# ==========================================================


def initialize_directories() -> None:

    for directory in (
        MODEL_DIR,
        ARTIFACT_DIR,
        REPORT_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ==========================================================
# Generic Helper
# ==========================================================


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:

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


def load_dataframe(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(path.name)

    logger.info(
        "Loaded %s %s",
        path.name,
        df.shape,
    )

    return df


def load_artifact(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(path)

    logger.info(
        "Loading %s",
        path.name,
    )

    return joblib.load(path)


def save_artifact(
    artifact: Any,
    output_path: Path,
) -> None:

    joblib.dump(
        artifact,
        output_path,
    )

    logger.info(
        "Saved %s",
        output_path.name,
    )


# ==========================================================
# Dataset
# ==========================================================


def load_processed_datasets():

    X_train = load_dataframe(PROCESSED_DIR / "X_train.csv")

    X_val = load_dataframe(PROCESSED_DIR / "X_val.csv")

    X_test = load_dataframe(PROCESSED_DIR / "X_test.csv")

    y_train = load_dataframe(PROCESSED_DIR / "y_train.csv").iloc[:, 0]

    y_val = load_dataframe(PROCESSED_DIR / "y_val.csv").iloc[:, 0]

    y_test = load_dataframe(PROCESSED_DIR / "y_test.csv").iloc[:, 0]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ==========================================================
# Model
# ==========================================================


def load_baseline_model() -> ClassifierMixin:

    return load_artifact(
        BASELINE_MODEL_PATH,
    )


def build_candidate_model() -> RandomForestClassifier:

    return RandomForestClassifier(
        random_state=RANDOM_STATE,
    )


def get_search_space() -> dict[str, list[Any]]:

    return {
        "n_estimators": [
            100,
            200,
            300,
            500,
        ],
        "max_depth": [
            None,
            5,
            10,
            20,
        ],
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
        ],
        "max_features": [
            "sqrt",
            "log2",
        ],
        "bootstrap": [
            True,
            False,
        ],
    }


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    RandomForestClassifier,
    float,
]:

    logger.info("Starting hyperparameter tuning...")

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=build_candidate_model(),
        param_distributions=get_search_space(),
        n_iter=N_ITER,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )

    start = time.perf_counter()

    search.fit(
        X_train,
        y_train,
    )

    elapsed = time.perf_counter() - start

    logger.info(
        "Best CV Score : %.4f",
        search.best_score_,
    )

    logger.info(
        "Best Parameters : %s",
        search.best_params_,
    )

    return (
        search,
        elapsed,
    )


# ==========================================================
# Prediction
# ==========================================================


def predict_model(
    model: ClassifierMixin,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return (
        predictions,
        probabilities,
    )


# ==========================================================
# Evaluation
# ==========================================================


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
    }


def evaluate_model(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
):

    prediction, probability = predict_model(
        model,
        X,
    )

    metrics = calculate_metrics(
        y,
        prediction,
        probability,
    )

    return (
        metrics,
        prediction,
        probability,
    )


# ==========================================================
# Report
# ==========================================================


def save_metrics(
    metrics: dict[str, float],
) -> None:

    save_json(
        metrics,
        METRICS_PATH,
    )


def save_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )

    with CLASSIFICATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    logger.info(
        "Saved %s",
        CLASSIFICATION_REPORT_PATH.name,
    )


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6),
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
    ).plot(
        ax=ax,
        colorbar=False,
    )

    ax.set_title("Tuned Model Confusion Matrix")

    fig.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved %s",
        CONFUSION_MATRIX_PATH.name,
    )


def save_roc_curve(
    y_true: pd.Series,
    probability: np.ndarray,
) -> None:

    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )

    auc = roc_auc_score(
        y_true,
        probability,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=auc,
    ).plot(
        ax=ax,
    )

    ax.set_title("ROC Curve (Tuned Model)")

    fig.savefig(
        ROC_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved %s",
        ROC_CURVE_PATH.name,
    )


# ==========================================================
# Metadata
# ==========================================================


def save_metadata(
    search: RandomizedSearchCV,
    metrics: dict[str, float],
    training_time: float,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:

    metadata = {
        "model_name": MODEL_NAME,
        "version": "tuned",
        "random_state": RANDOM_STATE,
        "training_timestamp": (datetime.now().isoformat()),
        "training_time_seconds": round(
            training_time,
            4,
        ),
        "cv_best_score": float(
            search.best_score_,
        ),
        "best_parameters": search.best_params_,
        "train_shape": list(
            X_train.shape,
        ),
        "validation_shape": list(
            X_val.shape,
        ),
        "test_shape": list(
            X_test.shape,
        ),
        "feature_names": (X_train.columns.tolist()),
        **metrics,
    }

    save_json(
        metadata,
        BEST_METADATA_PATH,
    )

    logger.info(
        "Saved %s",
        BEST_METADATA_PATH.name,
    )


# ==========================================================
# Comparison
# ==========================================================


def append_model_comparison(
    metrics: dict[str, float],
    training_time: float,
) -> None:

    row = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "version": "tuned",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "training_time": training_time,
            }
        ]
    )

    if MODEL_COMPARISON_PATH.exists():

        previous = pd.read_csv(
            MODEL_COMPARISON_PATH,
        )

        row = pd.concat(
            [
                previous,
                row,
            ],
            ignore_index=True,
        )

    row.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    logger.info(
        "Updated %s",
        MODEL_COMPARISON_PATH.name,
    )


# ==========================================================
# Save Model
# ==========================================================


def save_best_model(
    model: ClassifierMixin,
) -> None:

    save_artifact(
        model,
        BEST_MODEL_PATH,
    )


# ==========================================================
# Pipeline
# ==========================================================


def run_pipeline() -> None:

    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("STARTING MODEL TUNING")
    logger.info("=" * 60)

    initialize_directories()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_processed_datasets()

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

    # ------------------------------------------------------
    # Load Baseline
    # ------------------------------------------------------

    baseline_model = load_baseline_model()

    baseline_metrics, _, _ = evaluate_model(
        baseline_model,
        X_val,
        y_val,
    )

    logger.info(
        "Baseline Validation F1 : %.4f",
        baseline_metrics["f1_score"],
    )

    # ------------------------------------------------------
    # Hyperparameter Tuning
    # ------------------------------------------------------

    search, tuning_time = tune_model(
        X_train,
        y_train,
    )

    tuned_model = search.best_estimator_

    tuned_metrics, prediction, probability = evaluate_model(
        tuned_model,
        X_val,
        y_val,
    )

    logger.info(
        "Tuned Validation F1 : %.4f",
        tuned_metrics["f1_score"],
    )

    # ------------------------------------------------------
    # Model Selection
    # ------------------------------------------------------

    if tuned_metrics["f1_score"] >= baseline_metrics["f1_score"]:

        logger.info("Selected tuned model.")

        best_model = tuned_model

    else:

        logger.warning("Baseline model performs better.")

        best_model = baseline_model

    # ------------------------------------------------------
    # Final Test Evaluation (Only Once)
    # ------------------------------------------------------

    logger.info("Evaluating best model on test dataset...")

    test_metrics, test_prediction, test_probability = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    for key, value in test_metrics.items():

        logger.info(
            "%s : %.4f",
            key,
            value,
        )

    # ------------------------------------------------------
    # Save Outputs
    # ------------------------------------------------------

    save_metrics(
        test_metrics,
    )

    save_classification_report(
        y_test,
        test_prediction,
    )

    save_confusion_matrix(
        y_test,
        test_prediction,
    )

    save_roc_curve(
        y_test,
        test_probability,
    )

    save_best_model(
        best_model,
    )

    save_metadata(
        search,
        test_metrics,
        tuning_time,
        X_train,
        X_val,
        X_test,
    )

    append_model_comparison(
        test_metrics,
        tuning_time,
    )

    elapsed = time.perf_counter() - start

    logger.info(
        "Pipeline finished in %.2f seconds.",
        elapsed,
    )

    logger.info("=" * 60)
    logger.info("MODEL TUNING FINISHED SUCCESSFULLY")
    logger.info("=" * 60)


# ==========================================================
# CLI
# ==========================================================


def main() -> None:

    try:

        run_pipeline()

    except KeyboardInterrupt:

        logger.warning("Model tuning interrupted by user.")

    except Exception:

        logger.exception("Unexpected error occurred.")

        raise


if __name__ == "__main__":

    main()
