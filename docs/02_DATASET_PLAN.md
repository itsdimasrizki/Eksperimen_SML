# DATASET PLAN

## Project

Heart Disease Prediction System using End-to-End Machine Learning Pipeline

---

# 1. Dataset Overview

The dataset used in this project is the **Heart Disease Dataset** obtained from Kaggle. The dataset contains medical information from patients that can be utilized to predict the presence of heart disease.

This dataset is widely used in machine learning research and educational projects because it represents a real-world binary classification problem with numerical and categorical medical attributes.

The objective of using this dataset is to build an end-to-end machine learning system capable of:

* Data preprocessing automation
* Model training and evaluation
* Experiment tracking
* Model deployment
* Monitoring and alerting

---

# 2. Dataset Source

**Dataset Name:**
Heart Disease Dataset

**Source Platform:**
Kaggle

**Dataset URL:**
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

**Original Source:**
UCI Machine Learning Repository - Heart Disease Dataset

**License:**
Open for educational and research purposes.

---

# 3. Dataset Characteristics

| Characteristic     | Description                        |
| ------------------ | ---------------------------------- |
| Problem Type       | Binary Classification              |
| Number of Records  | 1025 observations                  |
| Number of Features | 13 features                        |
| Target Variable    | Heart Disease Presence             |
| Data Type          | Structured Tabular Data            |
| Missing Values     | None (based on initial inspection) |
| Duplicate Records  | Possible                           |
| Class Type         | Binary                             |
| Dataset Size       | Small to Medium                    |

---

# 4. Feature Description

| Feature  | Description                                    | Data Type |
| -------- | ---------------------------------------------- | --------- |
| age      | Age of patient                                 | Integer   |
| sex      | Gender of patient (1 = Male, 0 = Female)       | Integer   |
| cp       | Chest pain type                                | Integer   |
| trestbps | Resting blood pressure                         | Integer   |
| chol     | Serum cholesterol level                        | Integer   |
| fbs      | Fasting blood sugar > 120 mg/dl                | Integer   |
| restecg  | Resting electrocardiographic results           | Integer   |
| thalach  | Maximum heart rate achieved                    | Integer   |
| exang    | Exercise-induced angina                        | Integer   |
| oldpeak  | ST depression induced by exercise              | Float     |
| slope    | Slope of peak exercise ST segment              | Integer   |
| ca       | Number of major vessels colored by fluoroscopy | Integer   |
| thal     | Thalassemia type                               | Integer   |

# 4.1 Dataset Schema

| Column | Type | Role |
|--------|------|------|
| age | Numerical | Feature |
| sex | Categorical | Feature |
| cp | Categorical | Feature |
...
| target | Binary | Label |

---

# 5. Target Variable

| Value | Meaning                   |
| ----- | ------------------------- |
| 0     | No Heart Disease          |
| 1     | Presence of Heart Disease |

Prediction output:

```text
Input Patient Data
        ↓
Machine Learning Model
        ↓
Prediction:
0 → No Heart Disease
1 → Heart Disease
```

---

# 6. Data Quality Assessment

## Missing Values

Initial inspection indicates no significant missing values. Validation will still be performed during preprocessing.

## Duplicate Records

Duplicate observations may exist and should be removed to avoid bias during model training.

## Data Types

Most variables are numerical or encoded categorical features.

## Outliers

Potential outliers may exist in:

* cholesterol
* resting blood pressure
* maximum heart rate

Outlier analysis will be conducted using:

* Boxplot
* Interquartile Range (IQR)
* Statistical distribution analysis

## Class Balance

Class distribution will be evaluated to determine whether resampling techniques are required.

---

# 7. Problem Formulation

## Business Problem

Heart disease is one of the leading causes of mortality worldwide. Early prediction can help healthcare providers perform preventive interventions.

## Machine Learning Problem

Develop a binary classification model capable of predicting whether a patient has heart disease based on medical attributes.

## Research Questions

1. Can machine learning accurately predict heart disease?
2. Which features contribute most to prediction?
3. Can the model be deployed and monitored automatically using MLOps practices?

---

# 8. Evaluation Metrics

Because this project is a binary classification problem, several evaluation metrics will be used.

## Accuracy

Measures overall prediction correctness.

## Precision

Measures how many predicted positive cases are actually positive.

## Recall

Measures the model's ability to detect actual positive cases.

## F1 Score

Balances Precision and Recall.

## ROC-AUC

Measures model discrimination ability across thresholds.

---

## Target Metrics

| Metric    | Target |
| --------- | ------ |
| Accuracy  | ≥ 80%  |
| Precision | ≥ 80%  |
| Recall    | ≥ 80%  |
| F1 Score  | ≥ 80%  |
| ROC-AUC   | ≥ 85%  |

---

# 9. Data Splitting Strategy

The dataset will be divided into:

| Dataset        | Percentage |
| -------------- | ---------- |
| Training Set   | 70%        |
| Validation Set | 15%        |
| Test Set       | 15%        |

Stratified sampling will be applied to preserve class distribution.

```text
Dataset
│
├── Train (70%)
├── Validation (15%)
└── Test (15%)
```

Random State:

```python
random_state = 42
```

---

# 10. Risks and Assumptions

## Risks

### Small Dataset

Limited observations may cause overfitting.

### Dataset Bias

Data may not represent all populations.

### Class Imbalance

Unequal distribution may reduce model performance.

### Data Drift

Future production data may differ from training data.

### Overfitting

The model may memorize training data.

---

## Assumptions

* Dataset represents general heart disease characteristics.
* Features have predictive relationships with the target variable.
* Production data follows a similar distribution.
* Labels are correctly assigned.

---

# 11. Expected Output

The expected outputs of this phase are:

1. Cleaned dataset.
2. Preprocessed dataset.
3. Training-ready dataset.
4. Trained classification model.
5. Model artifacts.
6. Experiment tracking records.
7. REST API prediction service.
8. Monitoring dashboard.
9. Automated CI/CD pipeline.

---

# 12. Justification for Dataset Selection in End-to-End MLOps Implementation

The Heart Disease Dataset is selected because it is highly suitable for demonstrating an end-to-end MLOps pipeline.

## Reasons

### Structured Tabular Dataset

Allows implementation of complete preprocessing pipelines.

### Binary Classification Problem

Suitable for various evaluation metrics and model comparison.

### Moderate Complexity

Complex enough to demonstrate machine learning workflows while remaining manageable.

### Reproducible Experiments

Widely used by the machine learning community.

### Suitable for Monitoring

Prediction metrics and system metrics can be monitored continuously.

### Suitable for Deployment

The model has low computational requirements and can be served efficiently through FastAPI and Docker.

### Suitable for CI/CD Demonstration

The small dataset size allows fast retraining and automated deployment.

### Suitable for Experiment Tracking

Multiple algorithms and hyperparameter tuning can be demonstrated using MLflow.

---

# Conclusion

The Heart Disease Dataset provides an excellent foundation for implementing a complete machine learning lifecycle, including:

* Data preprocessing
* Model training
* Experiment tracking
* Model serving
* Monitoring
* Alerting
* Continuous Integration and Continuous Deployment (CI/CD)

Therefore, this dataset is highly appropriate for achieving a **Dicoding Membangun Sistem Machine Learning Submission with a Five-Star Target**.
