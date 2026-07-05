"""
Production-ready preprocessing automation for Heart Disease Prediction System.

This module automates the complete preprocessing pipeline based on the
validated experiment notebook.

Author: Dimas Rizki
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import time

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "preprocessing.log"

TARGET_COLUMN = "target"

EXPECTED_COLUMNS = [
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
    TARGET_COLUMN,
]

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
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
# Logging
# ==========================================================


def setup_logging() -> logging.Logger:
    """
    Configure application logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("preprocessing")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logging()

# ==========================================================
# Directory Utilities
# ==========================================================


def initialize_directories() -> None:
    """
    Create project output directories if they do not exist.
    """

    directories = [
        INTERIM_DIR,
        PROCESSED_DIR,
        REPORT_DIR,
        ARTIFACT_DIR,
        LOG_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# JSON Utilities
# ==========================================================


def save_json_report(data: dict[str, Any], output_path: Path) -> None:
    """
    Save report dictionary into JSON.

    Parameters
    ----------
    data:
        Dictionary to save.

    output_path:
        Destination path.
    """

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )

# ==========================================================
# CSV Utilities
# ==========================================================


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save dataframe into CSV.

    Parameters
    ----------
    df:
        DataFrame to save.

    output_path:
        Output csv path.
    """

    df.to_csv(
        output_path,
        index=False,
    )
    logger.info(
        "Saved %s",
        output_path.name,
    )

# ==========================================================
# Artifact Utilities
# ==========================================================


def save_artifact(
    artifact: Any,
    output_path: Path,
) -> None:
    """
    Serialize object using joblib.
    """

    joblib.dump(
        artifact,
        output_path,
    )


def load_artifact(
    artifact_path: Path,
) -> Any:
    """
    Load serialized object.
    """

    return joblib.load(artifact_path)

# ==========================================================
# Dataset Loader
# ==========================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset.

    Parameters
    ----------
    path:
        CSV dataset path.

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Loading dataset...")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    logger.info(
        "Dataset path : %s",
        path,
    )

    if df.empty:
        raise ValueError("Dataset is empty.")

    logger.info("Dataset loaded successfully.")
    logger.info("Shape : %s", df.shape)

    return df

# ==========================================================
# Validation
# ==========================================================


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate dataset schema.
    """

    logger.info("Validating dataset schema...")

    actual_columns = list(df.columns)

    missing_columns = sorted(
        set(EXPECTED_COLUMNS) - set(actual_columns)
    )

    unexpected_columns = sorted(
        set(actual_columns) - set(EXPECTED_COLUMNS)
    )

    duplicated_columns = (
        df.columns[df.columns.duplicated()]
        .tolist()
    )

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicated_columns": duplicated_columns,
        "is_valid": (
            len(missing_columns) == 0
            and len(duplicated_columns) == 0
        ),
    }

    save_json_report(
        report,
        REPORT_DIR / "data_validation.json",
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )
    
    if duplicated_columns:
        raise ValueError(
            f"Duplicated columns: {duplicated_columns}"
    )

    logger.info("Schema validation completed.")

    return report

# ==========================================================
# Missing Value Analysis
# ==========================================================


def analyze_missing_values(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze missing values.
    """

    logger.info("Analyzing missing values...")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    report = {
        "total_missing": int(missing.sum()),
        "missing_per_column": {
            column: int(value)
            for column, value in missing.items()
        },
    }

# ==========================================================
# Duplicate Analysis
# ==========================================================


def analyze_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Analyze and remove duplicate rows.
    """

    logger.info("Checking duplicate rows...")

    duplicate_count = int(df.duplicated().sum())

    cleaned_df = df.drop_duplicates()

    report = {
        "duplicate_rows": duplicate_count,
        "rows_before": int(df.shape[0]),
        "rows_after": int(cleaned_df.shape[0]),
    }

    save_json_report(
        report,
        REPORT_DIR / "duplicate_report.json",
    )

    logger.info(
        "Removed %d duplicate rows.",
        duplicate_count,
    )

    return cleaned_df, report

# ==========================================================
# Outlier Analysis
# ==========================================================


def analyze_outliers(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze outliers using IQR.

    This function DOES NOT remove outliers.
    """

    logger.info("Analyzing outliers...")

    report = {}

    for column in NUMERICAL_FEATURES:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total = int(
            (
                (df[column] < lower)
                | (df[column] > upper)
            ).sum()
        )

        report[column] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outliers": total,
        }

    save_json_report(
        report,
        REPORT_DIR / "outlier_report.json",
    )

    return report

# ==========================================================
# Preprocessing Pipeline
# ==========================================================


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    logger.info("Building preprocessing pipeline...")

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logger.info("Preprocessor created successfully.")

    logger.info(
        "Numerical features : %s",
        NUMERICAL_FEATURES,
    )

    logger.info(
        "Categorical features : %s",
        CATEGORICAL_FEATURES,
    )

    return preprocessor

# ==========================================================
# Dataset Split
# ==========================================================


def split_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split dataset into train,
    validation and test.
    """

    logger.info("Splitting dataset...")

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(
        "Train : %s",
        X_train.shape,
    )

    logger.info(
        "Validation : %s",
        X_val.shape,
    )

    logger.info(
        "Test : %s",
        X_test.shape,
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
# Preprocessor Execution
# ==========================================================


def preprocess_dataset(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fit preprocessing pipeline on training
    data only and transform all datasets.
    """

    logger.info(
        "Fitting preprocessing pipeline..."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_val_processed = pd.DataFrame(
        X_val_processed,
        columns=feature_names,
        index=X_val.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    logger.info(
        "Preprocessing completed."
    )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
    )

# ==========================================================
# Save Processed Dataset
# ==========================================================

def save_series(
    series: pd.Series,
    output_path: Path,
) -> None:
    """
    Save pandas Series to CSV.
    """

    series.to_frame().to_csv(
        output_path,
        index=False,
    )

def save_processed_dataset(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Save processed datasets.
    """

    logger.info(
        "Saving processed dataset..."
    )

    save_dataframe(
        X_train,
        PROCESSED_DIR / "X_train.csv",
    )

    save_dataframe(
        X_val,
        PROCESSED_DIR / "X_val.csv",
    )

    save_dataframe(
        X_test,
        PROCESSED_DIR / "X_test.csv",
    )

    save_series(
        y_train,
        PROCESSED_DIR / "y_train.csv",
    )

    save_series(
        y_val,
        PROCESSED_DIR / "y_val.csv",
    )

    save_series(
        y_test,
        PROCESSED_DIR / "y_test.csv",
    )

    logger.info(
        "Processed datasets saved."
    )

# ==========================================================
# Save Clean Dataset
# ==========================================================


def save_clean_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Save cleaned dataset.
    """

    save_dataframe(
        df,
        INTERIM_DIR / "heart_cleaned.csv",
    )

    logger.info(
        "Clean dataset saved."
    )

# ==========================================================
# Save Preprocessor
# ==========================================================


def save_preprocessor(
    preprocessor: ColumnTransformer,
) -> None:
    """
    Save preprocessing artifact.
    """

    save_artifact(
        preprocessor,
        ARTIFACT_DIR / "preprocessor.pkl",
    )

    logger.info(
        "Preprocessor artifact saved."
    )

# ==========================================================
# Preprocessing Report
# ==========================================================


def generate_preprocessing_report(
    validation_report: dict[str, Any],
    missing_report: dict,
    duplicate_report: dict,
    outlier_report: dict,
) -> None:
    """
    Generate preprocessing summary report.
    """

    report = {
        "schema_validation": validation_report,
        "missing_values": missing_report,
        "duplicates": duplicate_report,
        "outliers": outlier_report,
        "random_state": RANDOM_STATE,
        "split": {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE,
        },
    }

    save_json_report(
        report,
        REPORT_DIR / "preprocessing_report.json",
    )

    logger.info(
        "Preprocessing report generated."
    )

# ==========================================================
# Main Pipeline
# ==========================================================

def save_pipeline_metadata() -> None:
    """
    Save preprocessing configuration metadata.
    """

    metadata = {
        "random_state": RANDOM_STATE,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    save_json_report(
        metadata,
        ARTIFACT_DIR / "preprocessing_metadata.json",
    )

def run_pipeline() -> None:
    """
    Execute the complete preprocessing pipeline.
    """
    start_time = time.perf_counter()

    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    initialize_directories()

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    df = load_dataset(RAW_DATA_PATH)

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    validation_report = validate_schema(df)

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    missing_report = analyze_missing_values(df)

    # ------------------------------------------------------
    # Duplicate Analysis
    # ------------------------------------------------------

    df, duplicate_report = analyze_duplicates(df)

    # ------------------------------------------------------
    # Save Clean Dataset
    # ------------------------------------------------------

    save_clean_dataset(df)

    # ------------------------------------------------------
    # Outlier Analysis
    # ------------------------------------------------------

    outlier_report = analyze_outliers(df)

    # ------------------------------------------------------
    # Dataset Split
    # ------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(df)

    logger.info(
        "Train target distribution:\n%s",
        y_train.value_counts(normalize=True),
    )

    logger.info(
        "Validation target distribution:\n%s",
        y_val.value_counts(normalize=True),
    )

    logger.info(
        "Test target distribution:\n%s",
        y_test.value_counts(normalize=True),
    )

    # ------------------------------------------------------
    # Build Preprocessor
    # ------------------------------------------------------

    preprocessor = build_preprocessor()

    # ------------------------------------------------------
    # Transform Dataset
    # ------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
    ) = preprocess_dataset(
        preprocessor,
        X_train,
        X_val,
        X_test,
    )

    # ------------------------------------------------------
    # Save Dataset
    # ------------------------------------------------------

    save_processed_dataset(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # ------------------------------------------------------
    # Save Artifact
    # ------------------------------------------------------

    save_preprocessor(preprocessor)
    save_pipeline_metadata()

    # ------------------------------------------------------
    # Reports
    # ------------------------------------------------------

    generate_preprocessing_report(
        validation_report,
        missing_report,
        duplicate_report,
        outlier_report,
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSING FINISHED SUCCESSFULLY")
    logger.info("=" * 60)

    elapsed = time.perf_counter() - start_time

    logger.info(
    "Execution time : %.2f seconds",
    elapsed,
)

# ==========================================================
# Exception Handler
# ==========================================================


def main() -> None:
    """
    Application entry point.
    """

    try:

        run_pipeline()

    except KeyboardInterrupt:

        logger.warning(
            "Pipeline interrupted by user."
        )

    except Exception:

        logger.exception(
            "Unexpected error occurred."
        )

        raise

# ==========================================================
# CLI
# ==========================================================


if __name__ == "__main__":

    main()
