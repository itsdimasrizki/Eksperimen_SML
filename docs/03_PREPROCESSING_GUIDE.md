# PREPROCESSING GUIDE

## Heart Disease Prediction System using End-to-End Machine Learning Pipeline

---

# 1. Introduction

This document describes the complete preprocessing design for the Heart Disease Prediction System project.

The preprocessing stage is responsible for transforming raw medical data into clean, validated, and machine-learning-ready datasets that can be consumed by the training pipeline.

The preprocessing workflow is designed to support:

* Reproducible experiments
* Automated machine learning pipelines
* Continuous Integration and Continuous Deployment (CI/CD)
* Model deployment and serving
* Monitoring and future maintenance

This document serves as the implementation blueprint for:

* Exploratory notebook
* automate.py
* GitHub Actions preprocessing workflow
* Model training pipeline

---

# 2. Preprocessing Objectives

The objectives of preprocessing are:

1. Validate dataset integrity.
2. Detect and handle data quality issues.
3. Clean and standardize raw data.
4. Prepare features for machine learning algorithms.
5. Produce reproducible preprocessing pipelines.
6. Generate training-ready datasets.
7. Automate preprocessing execution.

Expected workflow:

```text
Raw Dataset
      ↓
Validation
      ↓
Cleaning
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Scaling
      ↓
Train Validation Test Split
      ↓
Processed Dataset
```

---

# 3. Input and Output Data

## Input Dataset

```text
data/raw/heart.csv
```

Dataset characteristics:

* 1025 observations
* 13 input features
* 1 target variable
* Binary classification problem

---

## Output Dataset

```text
data/interim/heart_cleaned.csv

data/processed/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv

artifacts/
└── preprocessor.pkl

reports/
├── data_validation.json
├── preprocessing_report.json
├── duplicate_report.json
├── outlier_report.json
└── missing_value_report.json
```

---

# 4. Dataset Directory Structure

```text
data/
│
├── raw/
│     └── heart.csv
│
├── interim/
│     └── heart_cleaned.csv
│
└── processed/
      ├── X_train.csv
      ├── X_val.csv
      ├── X_test.csv
      ├── y_train.csv
      ├── y_val.csv
      └── y_test.csv
```

Reports:

```text
reports/
├── data_validation.json
├── preprocessing_report.json
├── duplicate_report.json
├── outlier_report.json
└── missing_value_report.json
```

Artifacts:

```text
artifacts/
└── preprocessor.pkl
```

Logs:

```text
logs/
└── preprocessing.log
```

---

# 5. Data Loading Strategy

Library:

```python
import pandas as pd
```

Dataset loading:

```python
df = pd.read_csv("data/raw/heart.csv")
```

Validation after loading:

```python
df.head()
df.info()
df.describe()
```

The following checks must be performed:

* File existence
* File format validation
* Empty dataset validation
* Column validation
* Data type validation
* Null value inspection

---

# 6. Data Validation

Expected columns:

```python
[
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
    "target"
]
```

Validation rules:

| Validation        | Rule                     |
| ----------------- | ------------------------ |
| Missing column    | Error                    |
| Additional column | Warning                  |
| Empty dataframe   | Error                    |
| Duplicate column  | Error                    |
| Null target       | Error                    |
| Wrong datatype    | Auto-convert if possible |

Validation output:

```text
reports/data_validation.json
```

---

# 7. Missing Value Handling

Initial dataset inspection indicates that the dataset does not contain significant missing values.

However, the preprocessing pipeline must remain robust for future datasets.

## Numerical Features

Strategy:

```python
SimpleImputer(strategy="median")
```

Reason:

* Robust to outliers
* Suitable for medical data
* Reproducible

---

## Categorical Features

Strategy:

```python
SimpleImputer(strategy="most_frequent")
```

Reason:

* Preserves categorical distribution.
* Handles unexpected missing values in future data.

---

# 8. Duplicate Data Handling

Duplicate records may introduce training bias and data leakage.

Duplicate detection:

```python
df.duplicated().sum()
```

Treatment:

```python
df.drop_duplicates()
```

Output:

```text
reports/duplicate_report.json
```

---

# 9. Outlier Detection and Treatment

Potential numerical features containing outliers:

* age
* trestbps
* chol
* thalach
* oldpeak

---

## Detection Method

### Boxplot Analysis

### Interquartile Range (IQR)

```python
Q1 = feature.quantile(0.25)
Q3 = feature.quantile(0.75)
IQR = Q3 - Q1
```

Lower boundary:

```python
Q1 - 1.5 * IQR
```

Upper boundary:

```python
Q3 + 1.5 * IQR
```

---

## Treatment Strategy

The final treatment strategy will be determined after exploratory analysis.

Possible approaches:

* Keep outliers if medically meaningful.
* Winsorization (capping).
* Log transformation.
* Remove observations if proven to be data entry errors.

Reason:

Medical datasets may naturally contain extreme values that are clinically relevant.

Output:

```text
reports/outlier_report.json
```

---

# 10. Feature Engineering

Feature engineering is optional and will only be applied if experimental results indicate a significant performance improvement.

Potential feature engineering candidates:

* Age group categorization
* Cholesterol categorization
* Blood pressure categorization
* Interaction features

Examples:

```text
Age Category:
young
middle
old
```

```text
Cholesterol Category:
normal
borderline
high
```

```text
Blood Pressure Category:
normal
elevated
high
```

All engineered features must be:

* Experimentally validated
* Logged in MLflow
* Properly documented

---

# 11. Encoding Strategy

Categorical features:

```python
[
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]
```

Encoding strategy:

```python
OneHotEncoder(handle_unknown="ignore")
```

Reason:

* Avoids ordinal assumptions.
* Suitable for multiple algorithms.
* Compatible with production deployment.

Implementation:

```python
ColumnTransformer()
```

---

# 12. Feature Scaling Strategy

Numerical features:

```python
[
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]
```

Candidate scalers:

* StandardScaler
* MinMaxScaler
* RobustScaler

The final scaler will be selected based on experimental evaluation.

Reason:

Different machine learning algorithms may perform better with different scaling methods.

Preprocessor artifact:

```text
artifacts/preprocessor.pkl
```

---

# 13. Data Splitting Strategy

Dataset split:

| Dataset    | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

Method:

```python
train_test_split(
    stratify=y,
    random_state=42
)
```

Splitting procedure:

```text
Dataset
│
├── Train (70%)
└── Temporary (30%)
       │
       ├── Validation (15%)
       └── Test (15%)
```

Reason:

Stratified sampling preserves class distribution.

---

# 14. Reproducibility Configuration

```python
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
```

```python
np.random.seed(RANDOM_STATE)
```

Reason:

Ensures experiment reproducibility.

---

# 15. Preprocessing Pipeline Flow

```text
Load Dataset
      ↓
Validate Dataset
      ↓
Handle Missing Values
      ↓
Remove Duplicates
      ↓
Detect Outliers
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Scaling
      ↓
Train Validation Test Split
      ↓
Save Dataset
      ↓
Save Artifacts
      ↓
Generate Reports
      ↓
Generate Logs
```

---

# 16. Automation Strategy (automate.py)

Responsibilities:

1. Load raw dataset.
2. Validate schema.
3. Perform data cleaning.
4. Execute preprocessing.
5. Split dataset.
6. Save processed datasets.
7. Save preprocessing artifacts.
8. Generate reports.
9. Generate logs.

CLI execution:

```bash
python automate.py
```

Expected output:

```text
✓ data validation completed
✓ preprocessing completed
✓ artifacts saved
✓ reports generated
✓ processed datasets saved
✓ preprocessing metadata logged
```

---

# 17. Logging Strategy

Log file:

```text
logs/preprocessing.log
```

Information recorded:

* Dataset shape
* Missing values
* Duplicate count
* Outlier count
* Execution time
* Pipeline status
* Error messages

---

# 18. GitHub Actions Preprocessing Workflow

File:

```text
.github/workflows/preprocessing.yml
```

Trigger:

```yaml
on:
  push:
    branches:
      - main

  workflow_dispatch:
```

Pipeline:

```text
Checkout Repository
       ↓
Setup Python
       ↓
Install Dependencies
       ↓
Execute automate.py
       ↓
Generate Reports
       ↓
Upload Artifacts
       ↓
Upload Processed Dataset
```

---

# 19. Risks and Mitigation

| Risk                      | Mitigation                |
| ------------------------- | ------------------------- |
| Missing columns           | Schema validation         |
| Corrupted dataset         | Exception handling        |
| Data drift                | Monitoring and retraining |
| Class imbalance           | Stratified split          |
| Unexpected missing values | Imputation                |
| Pipeline failure          | GitHub Actions logging    |
| Reproducibility issue     | Random seed               |

---

# 20. Expected Deliverables

```text
data/interim/heart_cleaned.csv

data/processed/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv

artifacts/
└── preprocessor.pkl

reports/
├── preprocessing_report.json
├── data_validation.json
├── duplicate_report.json
├── outlier_report.json
└── missing_value_report.json

logs/
└── preprocessing.log
```

---

# 21. Pseudocode Preprocessing Pipeline

```python
LOAD dataset

VALIDATE schema

CHECK missing values
IF missing:
    IMPUTE values

CHECK duplicates
REMOVE duplicates

DETECT outliers
APPLY treatment strategy

OPTIONAL feature engineering

ENCODE categorical variables

SCALE numerical variables

SPLIT dataset:
    train
    validation
    test

SAVE processed datasets

SAVE preprocessing artifact

GENERATE reports

GENERATE logs
```

---

# 22. Justification of Preprocessing Choices for MLOps Implementation

## Reproducibility

Entire preprocessing can be executed automatically and consistently.

---

## Automation

Compatible with:

* automate.py
* GitHub Actions
* MLflow pipeline

---

## Portability

Preprocessor can be serialized:

```python
joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
```

---

## Deployment Compatibility

Inference service can reuse the same preprocessing artifact.

```python
joblib.load("artifacts/preprocessor.pkl")
```

---

## Monitoring Readiness

Reports and logs support:

* Data quality monitoring
* Data drift analysis
* CI/CD validation

---

## Experiment Tracking

Preprocessing parameters and artifacts can be logged to MLflow.

---

## Dicoding Five-Star Readiness

This preprocessing design supports:

* Automated preprocessing pipeline
* Reproducible experiments
* CI/CD implementation
* Artifact management
* Model deployment
* Monitoring and maintenance
* End-to-end MLOps implementation

---

# Conclusion

The preprocessing pipeline is designed to provide a robust, reproducible, and production-ready foundation for the Heart Disease Prediction System.

This design enables smooth integration with:

* Experiment notebook
* Automated training pipeline
* MLflow tracking
* FastAPI deployment
* Docker containerization
* Monitoring and alerting systems
* Continuous Integration and Continuous Deployment workflows
