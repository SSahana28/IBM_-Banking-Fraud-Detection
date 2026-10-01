import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.preprocess import BankingDataPreprocessor, engineer_features
except ImportError:
    from preprocess import BankingDataPreprocessor, engineer_features
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)

def evaluate_on_test(data_dir: str, models_dir: str):
    """
    Evaluates champion model on held-out test split and generates visual artifacts.
    """
    # 1. Load artifacts
    print("[1/4] Loading unseen test data and champion model...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    model = joblib.load(os.path.join(models_dir, "best_fraud_model.joblib"))
    raw_prep = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    if isinstance(raw_prep, dict):
        preprocessor = BankingDataPreprocessor(pipeline=raw_prep["pipeline"], feature_names=raw_prep["feature_names"])
    else:
        preprocessor = raw_prep
    
    with open(os.path.join(models_dir, "model_metrics.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    threshold = meta.get("optimal_threshold", 0.5)
    model_name = meta.get("champion_model", "Champion_Model")
    
    # 2. Predict on Test
    print(f"[2/4] Generating predictions for {len(y_test):,} test transactions (threshold={threshold:.4f})...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    roc_auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    
    print("\n" + "=" * 60)
    print(f"       TEST SET EVALUATION REPORT ({model_name})")
    print("=" * 60)
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Fraud F1: {report['1']['f1-score']:.4f}")
    print(f"Fraud Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")
    print("-" * 60)
    print(f"Confusion Matrix:\nTrue Legitimate: {cm[0,0]} | False Positive: {cm[0,1]}")
    print(f"False Negative:  {cm[1,0]} | True Fraud:      {cm[1,1]}")
    print("=" * 60 + "\n")
    
    # 3. Visualizations
    print("[3/4] Generating evaluation plots...")
    
    # Plot A: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    plt.title(f"Test Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=200)
    plt.close()
    
    # Plot B: ROC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax1.plot(fpr, tpr, color="#0f62fe", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
    ax2.plot(rec_curve, prec_curve, color="#da1e28", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "evaluation_curves.png"), dpi=200)
    plt.close()
    
    # Plot C: Feature Importances / Coefficients
    feature_names = preprocessor.feature_names
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Random Forest Feature Importances"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Logistic Regression Feature Coefficients (Magnitude)"
    else:
        importances = np.ones(len(feature_names))
        title = "Feature Importances"
        
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="crest")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "feature_importance.png"), dpi=200)
    plt.close()
    
    # 4. Save test metrics
    print("[4/4] Writing test evaluation report...")
    test_results = {
        "model_name": model_name,
        "test_records": int(len(y_test)),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fraud_precision": round(float(report["1"]["precision"]), 4),
        "fraud_recall": round(float(report["1"]["recall"]), 4),
        "fraud_f1": round(float(report["1"]["f1-score"]), 4),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
    }
    with open(os.path.join(models_dir, "test_evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
        
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    evaluate_on_test(data_dir, models_dir)
