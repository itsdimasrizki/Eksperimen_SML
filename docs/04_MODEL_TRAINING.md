# MODEL TRAINING

## Heart Disease Prediction System using End-to-End Machine Learning Pipeline

---

# 1. Introduction

This document describes the complete model training design for the Heart Disease Prediction System project.

The training phase is responsible for:

- Building machine learning models.
- Comparing multiple algorithms.
- Performing hyperparameter tuning.
- Tracking experiments using MLflow.
- Generating reproducible model artifacts.
- Registering production-ready models.
- Preparing future CI/CD integration.

This document serves as the implementation blueprint for:

- `src/modelling.py`
- `src/modelling_tuning.py`
- MLflow Tracking
- Model Artifact Management
- Future Continuous Training Workflow

---

# 2. Training Objectives

The objectives of this phase are:

1. Train multiple classification algorithms.
2. Evaluate model performance.
3. Select the best performing model.
4. Track all experiments using MLflow.
5. Generate reproducible artifacts.
6. Register production-ready models.
7. Support automated retraining pipelines.

Expected workflow:

```text
Processed Dataset
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
MLflow Tracking
        ↓
Model Registration
        ↓
Artifact Generation
```

---

# 3. Input and Output Artifacts

## Input

```text
data/processed/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv

artifacts/
└── preprocessor.pkl
```

## Output

```text
models/
├── baseline_model.pkl
├── best_model.pkl
└── model_metadata.json

artifacts/
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── classification_report.txt
└── metrics.json

logs/
└── training.log

reports/
└── model_comparison.csv

mlruns/
```

---

# 4. Dataset Usage Strategy

Dataset split:

| Dataset | Usage |
|----------|--------|
| Train | Model training |
| Validation | Hyperparameter tuning |
| Test | Final evaluation |

Split ratio:

```text
Train      : 70%
Validation : 15%
Test       : 15%
```

Stratified sampling:

```python
random_state = 42
stratify = y
```

---

# 5. Candidate Algorithms

## Logistic Regression

Advantages:

- Simple and interpretable.
- Fast training and inference.
- Strong baseline for binary classification.

---

## Random Forest Classifier

Advantages:

- Handles nonlinear relationships.
- Robust against overfitting.
- Provides feature importance.

---

## XGBoost Classifier

Advantages:

- Excellent performance on tabular datasets.
- Handles complex feature interactions.
- Frequently achieves state-of-the-art performance.

---

## Gradient Boosting Classifier

Advantages:

- Strong ensemble learning algorithm.
- Good generalization ability.

---

## Support Vector Machine (SVM)

Advantages:

- Effective on small datasets.
- Robust decision boundaries.

---

# 6. Baseline Model Selection

Baseline model:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    random_state=42
)
```

Reason:

- Excellent performance on structured tabular datasets.
- Minimal preprocessing requirements.
- Robust against noisy data.
- Provides explainability through feature importance.
- Easy integration with MLflow.

---

# 7. Hyperparameter Tuning Strategy

Technique:

- Grid Search
- Random Search

Primary approach:

```python
RandomizedSearchCV
```

Reason:

- Faster than exhaustive search.
- Lower computational cost.
- Suitable for CI/CD retraining.

---

## Random Forest Search Space

```python
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
```

---

## XGBoost Search Space

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}
```

---

# 8. Cross Validation Strategy

Method:

```python
StratifiedKFold
```

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Reason:

- Preserves class distribution.
- Produces more reliable evaluation.
- Reduces metric variance.

---

# 9. Evaluation Metrics

## Accuracy

Measures overall prediction correctness.

## Precision

Measures prediction quality of positive class.

## Recall

Measures ability to detect positive cases.

## F1 Score

Balances Precision and Recall.

## ROC-AUC

Measures model discrimination capability.

---

## Target Metrics

| Metric | Target |
|---------|---------|
| Accuracy | ≥ 85% |
| Precision | ≥ 85% |
| Recall | ≥ 85% |
| F1 Score | ≥ 85% |
| ROC-AUC | ≥ 90% |

---

# 10. Model Comparison Strategy

Comparison criteria:

1. Validation Accuracy
2. Validation F1 Score
3. ROC-AUC
4. Generalization Gap
5. Training Time
6. Inference Time
7. Model Size

Comparison output:

```text
reports/model_comparison.csv
```

Example:

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|-----------|-----------|----------|
| Logistic Regression | 0.85 | 0.84 | 0.90 |
| Random Forest | 0.89 | 0.88 | 0.94 |
| XGBoost | 0.91 | 0.90 | 0.96 |

---

# 11. Experiment Tracking Strategy using MLflow

Experiment name:

```python
heart_disease_prediction
```

Tracking URI:

```python
mlruns/
```

Parameters logged:

```text
algorithm
hyperparameters
dataset_version
random_state
cross_validation
```

Metrics logged:

```text
accuracy
precision
recall
f1
roc_auc
training_time
inference_time
```

Artifacts logged:

```text
confusion_matrix.png
roc_curve.png
classification_report.txt
feature_importance.png
best_model.pkl
```

---

# 12. Model Registration Strategy

Registered model:

```text
heart_disease_model
```

Model stages:

```text
None
Staging
Production
Archived
```

Promotion criteria:

```text
Accuracy >= 0.85
F1 Score >= 0.85
ROC-AUC >= 0.90
```

---

# 13. Artifact Management Strategy

Directory:

```text
models/
artifacts/
mlruns/
reports/
```

Versioning example:

```text
best_model_v1.pkl
best_model_v2.pkl
```

Metadata example:

```json
{
  "algorithm": "RandomForest",
  "version": "v1",
  "accuracy": 0.89,
  "f1": 0.88
}
```

---

# 14. Reproducibility Configuration

```python
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 50
```

```python
np.random.seed(RANDOM_STATE)
```

```python
PYTHONHASHSEED=42
```

---

# 15. Directory Structure

```text
src/
├── modelling.py
└── modelling_tuning.py

models/
├── baseline_model.pkl
├── best_model.pkl
└── model_metadata.json

artifacts/
├── confusion_matrix.png
├── feature_importance.png
├── roc_curve.png
└── metrics.json

logs/
└── training.log

reports/
└── model_comparison.csv

mlruns/
```

---

# 16. Workflow of modelling.py

```text
Load Processed Dataset
        ↓
Train Baseline Models
        ↓
Evaluate Models
        ↓
Compare Metrics
        ↓
Save Best Model
        ↓
Generate Artifacts
        ↓
Log to MLflow
```

Responsibilities:

1. Train baseline algorithms.
2. Evaluate metrics.
3. Compare model performance.
4. Save artifacts.
5. Log experiments.

---

# 17. Workflow of modelling_tuning.py

```text
Load Dataset
        ↓
Load Candidate Model
        ↓
Hyperparameter Search
        ↓
Cross Validation
        ↓
Evaluate Tuned Model
        ↓
Register Model
        ↓
Save Artifacts
        ↓
Log to MLflow
```

Responsibilities:

1. Hyperparameter tuning.
2. Evaluate tuned models.
3. Register best model.
4. Save artifacts.

---

# 18. Logging Strategy

Log file:

```text
logs/training.log
```

Information recorded:

- Dataset shape.
- Selected algorithm.
- Hyperparameters.
- Metrics.
- Training duration.
- Validation duration.
- Artifact generation status.
- Model registration status.
- Error messages.

---

# 19. Risks and Mitigation

| Risk | Mitigation |
|------|-------------|
| Overfitting | Cross Validation |
| Underfitting | Hyperparameter Tuning |
| Data Leakage | Strict Dataset Split |
| Experiment Loss | MLflow Tracking |
| Non-Reproducibility | Random Seed |
| Pipeline Failure | Logging and Exception Handling |
| Model Drift | Future Monitoring |

---

# 20. Expected Deliverables

```text
models/
├── baseline_model.pkl
├── best_model.pkl
└── model_metadata.json

artifacts/
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── metrics.json
└── classification_report.txt

reports/
└── model_comparison.csv

logs/
└── training.log

mlruns/
```

---

# 21. Pseudocode for Training Pipeline

```python
LOAD datasets

FOR each algorithm:

    TRAIN model

    PREDICT validation set

    COMPUTE metrics

    LOG parameters to MLflow

    LOG metrics to MLflow

    SAVE artifacts

SELECT best model

SAVE model

REGISTER model

GENERATE reports
```

---

# 22. Justification of Algorithm and Training Choices for End-to-End MLOps Implementation

## Why Random Forest as Baseline

- Strong performance on tabular medical datasets.
- Easy deployment and maintenance.
- Low inference cost.
- Provides feature importance.
- Requires minimal preprocessing.

---

## Why Multiple Algorithms

- Prevents algorithm bias.
- Produces objective comparisons.
- Demonstrates experiment tracking capabilities.

---

## Why MLflow

- Experiment reproducibility.
- Artifact versioning.
- Model registry.
- Deployment compatibility.
- CI/CD integration.

---

## Why RandomizedSearchCV

- Faster than exhaustive grid search.
- Suitable for automated retraining pipelines.

---

## Why StratifiedKFold

- Stable evaluation.
- Preserves class balance.
- Reduces overfitting risk.

---

# Conclusion

The model training phase provides a reproducible and production-ready workflow that integrates:

```text
Processed Dataset
        ↓
Training
        ↓
Evaluation
        ↓
MLflow
        ↓
Model Registry
        ↓
Artifact Management
        ↓
CI/CD
        ↓
Deployment
        ↓
Monitoring
```

This design enables:

- Automated retraining
- Experiment tracking
- Artifact management
- CI/CD integration
- Docker deployment
- Monitoring and alerting
- Production model versioning
- Dicoding Five-Star submission readiness