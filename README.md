# COSC2669 Individual Task 1 - Fraud Detection Analysis

Code for Individual Task 1: Part 1, Case Studies in Data Science, RMIT University.

## Overview
This repository contains the machine learning analysis referenced in my Individual Task 1 executive summary, targeting the Deloitte Data & Analytics Vacationer Program.

Two fraud classification datasets are analysed using two machine learning models:
- **Random Forest**
- **Neural Network (MLPClassifier)**

## Datasets
- `auto_insurance_fraud.xlsx` — Auto Insurance Claims Data (Shah, 2019), 1,000 records
- `healthcare_fraud_detection.csv` — Healthcare Fraud Detection Dataset (Abbas, 2023), 10,000 records

## Method
- Data preprocessing: missing value handling, categorical encoding
- Class imbalance addressed for the Neural Network using SMOTE oversampling
- Models evaluated using precision, recall, F1-score, and AUC (chosen over plain accuracy due to class imbalance)

## Results Summary
| Dataset | Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Auto Insurance | Random Forest | 0.61 | 0.71 | 0.66 | 0.82 |
| Auto Insurance | Neural Network | 0.49 | 0.55 | 0.52 | 0.74 |
| Healthcare | Random Forest | 0.64 | 0.94 | 0.76 | 0.99 |
| Healthcare | Neural Network | 0.92 | 0.87 | 0.90 | 1.00 |

## Files
- `Task1_Analysis.ipynb` — full analysis notebook
- `auto_insurance_fraud.xlsx`, `healthcare_fraud_detection.csv` — datasets

## Requirements
See `requirements.txt`
