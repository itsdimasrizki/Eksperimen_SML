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
❯ tree -L 3
.
├── artifacts
│   └── preprocessor.pkl
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
├── monitoring
│   └── prometheus.yml
├── notebook
│   └── eksperimen.ipynb
├── README.md
├── reports
│   ├── data_validation.json
│   ├── duplicate_report.json
│   ├── missing_value_report.json
│   └── preprocessing_report.json
├── requirements.txt
└── src
    ├── app.py
    ├── automate.py
    ├── inference.py
    ├── modelling.py
    ├── modelling_tuning.py
    └── prometheus_exporter.py

11 directories, 36 files