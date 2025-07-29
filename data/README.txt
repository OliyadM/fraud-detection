# Fraud Detection in E-commerce and Bank Transactions

This repository contains a machine learning pipeline designed to detect fraudulent financial transactions. The project includes data preprocessing, model training, evaluation, and comparison across two real-world datasets.

## Project Overview

Fraudulent activities in financial systems pose significant threats. This project aims to develop and compare models that accurately identify fraudulent transactions. We use two datasets:
- **E-commerce Transactions** (`Fraud_Data.csv`)
- **Credit Card Transactions** (`creditcard.csv`)

Both datasets exhibit strong class imbalance, with fraudulent cases being rare. Appropriate resampling techniques and evaluation metrics have been used to address this.

## Objectives

- Explore and preprocess fraud-related financial datasets.
- Handle class imbalance using techniques such as SMOTE.
- Train baseline and advanced machine learning models.
- Evaluate and compare models using precision-recall-focused metrics.
- Generate insights for model deployment and risk mitigation.

## Directory Structure

FRAU.../
├── data/
│ ├── processed/
│ ├── raw/
│ └── README.txt
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_modeling.ipynb
├── outputs/
├── src/
│ └── preprocess.py
├── requirements.txt
└── README.md

## How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OliyadM/fraud-detection.git
   cd fraud-detection
pip install -r requirements.txt
Run Notebooks
Open and run the Jupyter notebooks in order:

notebooks/01_eda.ipynb

notebooks/02_modeling.ipynb

Dependencies
Key libraries used:

pandas

numpy

scikit-learn

imbalanced-learn

xgboost

matplotlib, seaborn

Install all dependencies using:

pip install -r requirements.txt

Current Progress
Task	Status	Notes
Task 1: EDA & Preprocessing	✅ Completed	Includes SMOTE sampling and normalization
Task 2: Modeling	✅ Completed	Trained Logistic Regression, Random Forest, and XGBoost
Task 3: Tuning & Deployment	⏳ Upcoming	Will include hyperparameter tuning and reporting