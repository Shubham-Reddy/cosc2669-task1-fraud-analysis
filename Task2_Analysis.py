"""
COSC2669 Individual Task 2, Part 2 - Deliberation on Task 1
Learning curve and fairness analysis, building on the Task 1 fraud detection models.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def preprocess(df, target_col, drop_cols=None, keep_raw_col=None):
    df = df.copy()
    raw_series = df[keep_raw_col].copy() if keep_raw_col else None
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object).fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    y = df[target_col]
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        y = y.map(lambda v: 1 if str(v).strip().upper() in ['Y','YES','1','TRUE','FRAUD'] else 0)
    X = df.drop(columns=[target_col])
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        elif np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = pd.to_datetime(X[col]).astype('int64') // 10**9
    return X, y, raw_series

def run_learning_curve(X, y, name):
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=RANDOM_STATE)
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, scoring='f1', random_state=RANDOM_STATE,
        train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True
    )
    print(f"\n=== Learning Curve: {name} (Random Forest, F1, 5-fold CV) ===")
    for i, size in enumerate(train_sizes):
        print(f"Train size {size:5d} | Train F1: {train_scores[i].mean():.3f} | Val F1: {test_scores[i].mean():.3f}")

def run_fairness(X, y, sensitive_raw, name):
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_raw, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mf = MetricFrame(
        metrics={
            'selection_rate': selection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'recall': recall_score,
        },
        y_true=y_test, y_pred=y_pred, sensitive_features=sens_test
    )
    print(f"\n=== Fairlearn MetricFrame: {name} (by sensitive attribute) ===")
    print(mf.by_group)
    print("\nDisparity (max - min):")
    print(mf.difference())

# ===== AUTO INSURANCE =====
auto = pd.read_excel('auto_insurance_fraud.xlsx', sheet_name='Fraud_Detection_decsion tree')
X_auto, y_auto, sex_auto = preprocess(auto, target_col='fraud_reported',
                              drop_cols=['policy_number', 'incident_location', 'policy_bind_date', 'incident_date'],
                              keep_raw_col='insured_sex')
run_learning_curve(X_auto, y_auto, "AUTO INSURANCE FRAUD")
run_fairness(X_auto, y_auto, sex_auto, "AUTO INSURANCE FRAUD (by insured_sex)")

# ===== HEALTHCARE =====
health = pd.read_csv('healthcare_fraud_detection.csv')
X_health, y_health, gender_health = preprocess(health, target_col='Is_Fraud',
                                  drop_cols=['Claim_ID', 'Claim_Submission_Date'],
                                  keep_raw_col='Patient_Gender')
run_learning_curve(X_health, y_health, "HEALTHCARE FRAUD")
run_fairness(X_health, y_health, gender_health, "HEALTHCARE FRAUD (by Patient_Gender)")
