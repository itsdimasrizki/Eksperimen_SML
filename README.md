# Membangun Sistem Machine Learning

Proyek end-to-end machine learning untuk submission Dicoding.

## Tech Stack

- Python 3.12
- Scikit-Learn
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Prometheus
- Grafana

## Struktur Repository

- Eksperimen
- Training
- CI/CD
- Monitoring

## Struktur Folder
❯ tree
.
├── artifacts
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── preprocessing_metadata.json
│   ├── preprocessor.pkl
│   ├── roc_curve.png
│   ├── tuning_classification_report.txt
│   ├── tuning_confusion_matrix.png
│   ├── tuning_metrics.json
│   └── tuning_roc_curve.png
├── conda.yaml
├── data
│   ├── interim
│   │   └── heart_cleaned.csv
│   ├── processed
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── X_val.csv
│   │   ├── y_test.csv
│   │   ├── y_train.csv
│   │   └── y_val.csv
│   └── raw
│       └── heart.csv
├── Dockerfile
├── docs
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_DATASET_PLAN.md
│   ├── 03_PREPROCESSING_GUIDE.md
│   ├── 04_MODEL_TRAINING.md
│   ├── 05_MLFLOW_GUIDE.md
│   ├── 06_CI_CD_WORKFLOW.md
│   ├── 07_DEPLOYMENT_GUIDE.md
│   ├── 08_MONITORING_GUIDE.md
│   ├── 09_ALERTING_GUIDE.md
│   ├── 10_SUBMISSION_CHECKLIST.md
│   └── PHASE_DEVELOPMING.md
├── image
│   ├── Baseline_model.png
│   ├── Training_runs.png
│   └── Tuned_model.png
├── logs
│   ├── mlflow.log
│   ├── preprocessing.log
│   ├── training.log
│   └── tuning.log
├── mlflow.db
├── mlruns
│   └── 1
│       ├── 16a9f00bb6474aa09b5e00ee4559383b
│       │   └── artifacts
│       │       └── tuned
│       │           ├── best_model_metadata.json
│       │           ├── tuning_classification_report.txt
│       │           ├── tuning_confusion_matrix.png
│       │           ├── tuning_metrics.json
│       │           └── tuning_roc_curve.png
│       ├── 8387e94e542c4fcb9ef35e90f933bcb2
│       │   └── artifacts
│       │       └── baseline
│       │           ├── classification_report.txt
│       │           ├── confusion_matrix.png
│       │           ├── metrics.json
│       │           ├── model_metadata.json
│       │           └── roc_curve.png
│       ├── 8df55f8e4e6a4e84aff23fc8961a382e
│       │   └── artifacts
│       │       └── baseline
│       │           ├── classification_report.txt
│       │           ├── confusion_matrix.png
│       │           ├── metrics.json
│       │           ├── model_metadata.json
│       │           └── roc_curve.png
│       ├── cbb95a4ec5bf430aa68292b8c2e98b77
│       │   └── artifacts
│       │       └── tuned
│       │           ├── best_model_metadata.json
│       │           ├── tuning_classification_report.txt
│       │           ├── tuning_confusion_matrix.png
│       │           ├── tuning_metrics.json
│       │           └── tuning_roc_curve.png
│       └── models
│           ├── m-11257313273b4435948520567bbc2bff
│           │   └── artifacts
│           │       ├── conda.yaml
│           │       ├── input_example.json
│           │       ├── MLmodel
│           │       ├── model.skops
│           │       ├── python_env.yaml
│           │       ├── requirements.txt
│           │       └── serving_input_example.json
│           ├── m-1839bce1176245feb65dba4f7100e22b
│           │   └── artifacts
│           │       ├── conda.yaml
│           │       ├── input_example.json
│           │       ├── MLmodel
│           │       ├── model.skops
│           │       ├── python_env.yaml
│           │       ├── requirements.txt
│           │       └── serving_input_example.json
│           ├── m-1aeacbe6982f4b9985696c8943e85df5
│           │   └── artifacts
│           │       ├── conda.yaml
│           │       ├── input_example.json
│           │       ├── MLmodel
│           │       ├── model.skops
│           │       ├── python_env.yaml
│           │       ├── requirements.txt
│           │       └── serving_input_example.json
│           └── m-a81e7403cc934dfe82cf771bc7fe0621
│               └── artifacts
│                   ├── conda.yaml
│                   ├── input_example.json
│                   ├── MLmodel
│                   ├── model.skops
│                   ├── python_env.yaml
│                   ├── requirements.txt
│                   └── serving_input_example.json
├── models
│   ├── baseline_model.pkl
│   ├── best_model_metadata.json
│   ├── best_model.pkl
│   └── model_metadata.json
├── monitoring
│   └── prometheus.yml
├── notebook
│   └── eksperimen.ipynb
├── README.md
├── reports
│   ├── data_validation.json
│   ├── duplicate_report.json
│   ├── missing_value_report.json
│   ├── model_comparison.csv
│   ├── outlier_report.json
│   └── preprocessing_report.json
├── requirements.txt
└── src
    ├── app.py
    ├── automate.py
    ├── inference.py
    ├── mlflow_tracking.py
    ├── modelling.py
    ├── modelling_tuning.py
    └── prometheus_exporter.py

37 directories, 108 files