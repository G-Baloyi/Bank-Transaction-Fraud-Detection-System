# Bank Transaction Fraud Detection System

A full end-to-end machine learning pipeline for detecting fraudulent bank transactions — from raw data exploration through model deployment, threshold tuning, explainability analysis, and a production-style risk scoring dashboard.

---

## Project Overview

Financial fraud causes significant losses to banks and customers alike. This project builds a complete fraud detection system using supervised machine learning on a real-world-style banking transactions dataset. The pipeline covers every stage of a production data science workflow: data preparation, feature engineering, multi-model training, threshold optimisation, SHAP-based interpretability, and a risk-tiered scoring dashboard.

---

## Project Structure

```
Bank-Transaction-Fraud-Detection-System/
│
├── 01_Data_Preparation_EDA.ipynb         # Exploratory data analysis & cleaning
├── 02_Feature_Engineering.ipynb          # Feature creation & encoding
├── 03_Model_Training_Evaluation.ipynb    # Training & comparing 3 classifiers
├── 04_Threshold_Optimization_CV.ipynb    # Threshold tuning & cross-validation
├── 05_SHAP_Analysis.ipynb                # Model explainability with SHAP
├── 06_Risk_Scoring_Dashboard.ipynb       # Production risk scoring & business rules
│
├── data/
│   ├── Bank_Transaction_Fraud_Detection.csv   # Raw dataset
│   ├── data_after_eda.csv                     # Cleaned data (post-EDA)
│   ├── data_feature_engineered.csv            # Final feature set for modelling
│   └── features_list.npy                      # Saved feature list
│
└── models/
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    └── scaler.pkl
```

---

## Pipeline Walkthrough

### Notebook 1 — Data Preparation & EDA
- Loaded and profiled the raw dataset (shape, dtypes, nulls, duplicates)
- Analysed class imbalance between legitimate and fraudulent transactions
- Explored fraud rates across categorical features: device type, account type, transaction type, merchant category, and gender
- Investigated temporal patterns by hour, day of week, and month
- Examined transaction amount distributions and account balance vs. amount scatter plots

### Notebook 2 — Feature Engineering
- Extracted time-based features: `Hour`, `Minute`, `DayOfWeek`, `Month`, `DayOfMonth`
- Created behavioural flags: `IsNight`, `IsWeekend`, `IsRushHour`
- Engineered ratio and log-transform features: `Amt_to_Balance`, `Amt_to_Bal_Clipped`, `Log_Amount`, `Log_Balance`
- Label-encoded categorical columns: `Gender`, `Account_Type`, `Transaction_Type`, `Merchant_Category`, `Device_Type`, `State`
- Conducted correlation analysis to assess feature-target relationships

### Notebook 3 — Model Training & Evaluation
- Applied **SMOTE** to address class imbalance in the training set
- Trained three classifiers side-by-side:

  | Model | Notes |
  |---|---|
  | Logistic Regression | Fast, interpretable baseline |
  | Random Forest | Non-linear, robust to outliers |
  | XGBoost | Gradient boosting; best overall performance |

- Evaluated each model using AUC-ROC, Average Precision, F1-score, Precision, and Recall
- Visualised ROC curves, Precision-Recall curves, and confusion matrices for all three models
- Saved all trained models and the feature scaler to the `models/` directory

### Notebook 4 — Threshold Optimisation & Cross-Validation
- Swept thresholds from 0.05–0.95 to find the optimal decision boundary for the XGBoost model (maximising F1)
- Ran **5-fold stratified cross-validation** to verify model stability
- Visualised CV results with boxplots and fold-by-fold AUC breakdown
- Plotted XGBoost feature importances

### Notebook 5 — SHAP Analysis
- Used **SHAP TreeExplainer** to compute global and local feature attributions
- Generated beeswarm and bar plots to rank features by mean absolute impact
- Explained individual high-probability fraud transactions using waterfall/force plots
- Produced dependence plots for top features: `Transaction_Amount`, `Amt_to_Balance`, `Hour`, `Account_Balance`

### Notebook 6 — Risk Scoring Dashboard
- Scored the full dataset with XGBoost fraud probabilities
- Assigned transactions to four risk tiers with corresponding business rules:

  | Risk Tier | Score Range | Recommended Action |
  |---|---|---|
  | Low | 0.00 – 0.20 | Auto-approve |
  | Medium | 0.20 – 0.50 | Log & monitor |
  | High | 0.50 – 0.80 | Secondary review |
  | Critical | 0.80 – 1.00 | Block & investigate |

- Built a multi-panel dashboard visualising risk distribution, fraud amounts caught, and KPI summary cards

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Model explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |
| Model persistence | joblib |
| Environment | Jupyter Notebook |

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/G-Baloyi/Bank-Transaction-Fraud-Detection-System.git
cd Bank-Transaction-Fraud-Detection-System
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib
```

**3. Run the notebooks in order**
```
01 → 02 → 03 → 04 → 05 → 06
```
Each notebook reads outputs saved by the previous one, so running them in sequence is required.

---

## Key Results

- **Best model:** XGBoost
- **Evaluation metrics:** AUC-ROC, Average Precision, F1-score (see Notebook 3 for full comparison table)
- **Cross-validation:** 5-fold stratified CV confirmed model generalisation (Notebook 4)
- **Top predictive features (SHAP):** Transaction Amount, Amount-to-Balance Ratio, Hour of Transaction, Account Balance

---

## Business Value

This system provides a decision-ready output for financial institutions:
- Transactions are automatically tiered by risk level
- Each prediction is explainable at the individual transaction level using SHAP
- Business rules are directly tied to model output thresholds, enabling easy integration into approval workflows
- The pipeline is modular — each stage is independently rerunnable and extendable

---

## Author

**Goitsemang Baloyi**  
