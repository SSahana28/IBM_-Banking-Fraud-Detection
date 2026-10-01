# ??? IBM BOB - Banking Fraud Detection System

**A Machine Learning & Financial Risk Screening Platform**  
*Inspired by the Bank of Baroda (BOB) & IBM Enterprise Analytics initiatives, developed for and compatible with IBM Bob (`bobide`).*

---

## ?? Project Overview

Financial fraud causes billions of dollars in losses annually. In enterprise banking systems, fraudulent transactions represent a tiny fraction of total volume (typically < 2%), creating an extreme **class imbalance** challenge.

This project delivers a complete, production-grade Machine Learning fraud screening engine. It detects suspicious banking transactions in real-time, explains the underlying risk factors, and provides an interactive auditor dashboard.

---

## ?? Key Features

- **Domain-Specific Feature Engineering**:
  - `errorBalanceOrig`: Detects discrepancies between expected balance change and actual sender balance.
  - `errorBalanceDest`: Pinpoints anomalies in destination account credits.
  - `amount_to_bal_ratio`: Measures relative capital drainage.
  - `is_zero_balance_after`: Captures sudden total account liquidations.
  - `hour_sin` / `hour_cos`: Cyclical temporal encoding for off-hours transaction detection.
- **Strict ML Best Practices (Zero Data Leakage)**:
  - Stratified 70% Train / 15% Validation / 15% Test splitting executed **before** fitting transformers and scalers.
  - Preprocessing transformers fit strictly on the Training set.
- **Multi-Model Benchmark & Baselines**:
  - **Naive Baseline**: Majority class predictor (sanity check).
  - **Linear ML Baseline**: Class-weighted Logistic Regression.
  - **Non-Linear Ensemble**: Balanced Random Forest Classifier.
  - **Gradient Boosting**: HistGradientBoostingClassifier.
  - **Unsupervised Anomaly Detector**: Isolation Forest (outlier contamination scoring).
- **Threshold Optimization**:
  - Automatically tunes the classification threshold using the Precision-Recall curve on the validation set to optimize F1-score on the minority fraud class.
- **Explainable Predictions**:
  - Provides natural language risk indicators for compliance audits (e.g., account liquidation, high ratio transfer, unverified destination account).
- **Interactive Web App**:
  - Real-time screener with preset buttons, batch CSV auditing, and interactive diagnostic charts using Streamlit.

---

## ?? Benchmark Results (Held-Out Unseen Test Set: 4,500 Transactions)

| Metric | Score | Industry Target |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9934** | > 0.90 |
| **PR-AUC (Precision-Recall)** | **0.9856** | > 0.80 |
| **Fraud F1-Score** | **0.9692** | > 0.85 |
| **Fraud Precision** | **1.0000 (0 False Positives)** | > 0.90 |
| **Fraud Recall** | **0.9403 (63/67 Frauds Caught)** | > 0.90 |

---

## ?? Repository Structure

```
banking-fraud-detection/
??? .venv/                      # Python virtual environment
??? requirements.txt            # Pinned dependencies
??? README.md                   # Full documentation & project report
??? app.py                      # Streamlit interactive web dashboard
??? data/
?   ??? raw_transactions.csv   # Synthesized 30,000 banking transactions
?   ??? processed/              # Scaled and encoded train/val/test splits
?       ??? X_train.npy, y_train.npy
?       ??? X_val.npy, y_val.npy
?       ??? X_test.npy, y_test.npy
?       ??? test_raw.csv        # Raw test samples for batch auditor
??? models/
?   ??? best_fraud_model.joblib # Champion serialized model
?   ??? preprocessor.joblib     # Serialized preprocessing pipeline
?   ??? model_metrics.json      # Benchmark results & optimal threshold
?   ??? test_evaluation_report.json # Test metrics
?   ??? confusion_matrix.png    # Evaluation heatmap
?   ??? evaluation_curves.png   # ROC and PR curves
?   ??? feature_importance.png  # Feature weights
??? src/
    ??? __init__.py
    ??? generate_data.py        # Generates realistic imbalanced transactions
    ??? preprocess.py           # Strict train/val/test split and feature engineering
    ??? models.py               # Model definitions & baselines
    ??? train.py                # Model training, tuning, and champion selection
    ??? evaluate.py             # Evaluation on unseen test split & visualizations
    ??? predict.py              # Real-time scoring function with risk explanations
```

---

## ?? Quickstart Guide

### 1. Opening in IBM Bob IDE
To open and edit the project inside IBM Bob:
```bash
bobide "C:\Users\Savitha\.gemini\antigravity\scratch\banking-fraud-detection"
```

### 2. Activate Virtual Environment
```powershell
cd C:\Users\Savitha\.gemini\antigravity\scratch\banking-fraud-detection
.\.venv\Scripts\Activate.ps1
```

### 3. Re-run Pipeline (Optional)
```powershell
# 1. Generate data
python -m src.generate_data

# 2. Preprocess & split (No data leakage)
python -m src.preprocess

# 3. Train & benchmark all candidate models
python -m src.train

# 4. Evaluate on held-out test split & generate plots
python -m src.evaluate

# 5. Test real-time prediction CLI
python -m src.predict
```

### 4. Launch Interactive Web Dashboard
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501` to test transactions and view real-time fraud predictions!
