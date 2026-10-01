IBM BOB - Banking Fraud Detection System

A Machine Learning & Financial Risk Screening Platform  
Inspired by the Bank of Baroda (BOB) & IBM Enterprise Analytics initiatives, developed for and compatible with IBM Bob.


Project Overview

Financial fraud causes billions of dollars in losses annually. In enterprise banking systems, fraudulent transactions represent a tiny fraction of total volume (typically < 2%), creating an extreme **class imbalance** challenge.

This project delivers a complete, production-grade Machine Learning fraud screening engine. It detects suspicious banking transactions in real-time, explains the underlying risk factors, and provides an interactive auditor dashboard.


 Key Features

Domain-Specific Feature Engineering:
  - `errorBalanceOrig`: Detects discrepancies between expected balance change and actual sender balance.
  - `errorBalanceDest`: Pinpoints anomalies in destination account credits.
  - `amount_to_bal_ratio`: Measures relative capital drainage.
  - `is_zero_balance_after`: Captures sudden total account liquidations.
  - `hour_sin` / `hour_cos`: Cyclical temporal encoding for off-hours transaction detection.
- Strict ML Best Practices (Zero Data Leakage):
  - Stratified 70% Train / 15% Validation / 15% Test splitting executed **before** fitting transformers and scalers.
  - Preprocessing transformers fit strictly on the Training set.
- Multi-Model Benchmark & Baselines:
  - Naive Baseline: Majority class predictor (sanity check).
  - Linear ML Baseline: Class-weighted Logistic Regression.
  - Non-Linear Ensemble: Balanced Random Forest Classifier.
  - Gradient Boosting: HistGradientBoostingClassifier.
  - Unsupervised Anomaly Detector: Isolation Forest (outlier contamination scoring).
- Threshold Optimization:
  - Automatically tunes the classification threshold using the Precision-Recall curve on the validation set to optimize F1-score on the minority fraud class.
- Explainable Predictions:
  - Provides natural language risk indicators for compliance audits (e.g., account liquidation, high ratio transfer, unverified destination account).
- Interactive Web App:
  - Real-time screener with preset buttons, batch CSV auditing, and interactive diagnostic charts using Streamlit.

