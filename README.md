# COSC2669 Individual Task 1 - Fraud Detection Analysis

This is the code behind my Individual Task 1 report for Case Studies in Data Science, targeting the Deloitte Data & Analytics Vacationer Program.

I looked at two fraud datasets and ran two ML models on each - Random Forest and a Neural Network - to see how well they could catch fraudulent claims.

**Datasets:**
- Auto insurance claims (1,000 records)
- Healthcare claims (10,000 records, synthetic)

Both are pretty imbalanced (fraud is rare), so I used precision, recall, F1, and AUC instead of just accuracy - plain accuracy would be misleading here.

**Results:**

| Dataset | Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Auto Insurance | Random Forest | 0.61 | 0.71 | 0.66 | 0.82 |
| Auto Insurance | Neural Network | 0.49 | 0.55 | 0.52 | 0.74 |
| Healthcare | Random Forest | 0.64 | 0.94 | 0.76 | 0.99 |
| Healthcare | Neural Network | 0.92 | 0.87 | 0.90 | 1.00 |

Random Forest ended up being the more consistent performer across both datasets.

**Files:**
- `Task1_Analysis.ipynb` - main notebook
- `auto_insurance_fraud.xlsx`, `healthcare_fraud_detection.csv` - the two datasets
