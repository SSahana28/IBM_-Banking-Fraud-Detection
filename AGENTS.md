# AGENTS.md - IBM Bob Project Instructions

## Project Overview
This is an enterprise Machine Learning project: **AI-Powered Banking Fraud Detection & Anomaly Screening System** developed using IBM Bob.

## Technology Stack
- Python 3.14
- Scikit-Learn (Supervised and Unsupervised Classification)
- Streamlit (Real-time Transaction Auditor Dashboard)
- Pandas & NumPy (Data Processing and Feature Engineering)
- Joblib (Model Serialization)

## Architecture & Conventions
- `src/generate_data.py`: Synthesizes 30,000 imbalanced transaction records (~1.5% fraud rate).
- `src/preprocess.py`: Strict Train/Val/Test splitting (70/15/15) followed by domain feature engineering (`errorBalanceOrig`, `errorBalanceDest`, `amount_to_bal_ratio`, cyclical time encoding) with zero data leakage.
- `src/models.py`: Model registry with Naive Baseline, Class-Weighted Logistic Regression, Balanced Random Forest, HistGradientBoosting, and Isolation Forest.
- `src/train.py`: Multi-model benchmarking and threshold calibration on the Precision-Recall curve.
- `src/evaluate.py`: Evaluates champion model on held-out test split (4,500 transactions), generating confusion matrix and ROC/PR curves.
- `src/predict.py`: Production inference engine with natural-language explainable risk factors.
- `app.py`: Streamlit web dashboard running on port 8501.

## Performance Benchmark
- Test ROC-AUC: 0.9934
- Test PR-AUC: 0.9856
- Test Fraud F1: 0.9692
- Test Fraud Precision: 1.0000 (0 false positives)
- Test Fraud Recall: 0.9403 (63/67 caught)
