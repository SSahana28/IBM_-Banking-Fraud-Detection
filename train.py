import os
import json
import joblib
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)
try:
    from src.models import get_models
except ImportError:
    from models import get_models

def train_and_benchmark(data_dir: str, models_dir: str):
    """
    Trains baseline and candidate models, evaluates on validation set,
    tunes decision threshold for class imbalance, and selects champion.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load preprocessed train and validation splits
    print("[1/5] Loading preprocessed train and validation sets...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    print(f"      Train shape: {X_train.shape} | Val shape: {X_val.shape}")
    
    models = get_models()
    results = {}
    best_f1 = -1.0
    champion_name = None
    champion_model = None
    optimal_threshold = 0.50
    
    print("\n[2/5] Training & Evaluating Candidate Models on Validation Split:")
    print("-" * 75)
    print(f"{'Model':<25} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8}")
    print("-" * 75)
    
    for name, model in models.items():
        if name == "Isolation_Forest":
            # Unsupervised outlier detector
            model.fit(X_train)
            raw_scores = -model.score_samples(X_val) # Higher = more anomalous
            # Normalize to 0-1 range
            probs = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
            preds = (model.predict(X_val) == -1).astype(int)
            pr_auc = float(average_precision_score(y_val, probs))
            roc_auc = float(roc_auc_score(y_val, probs))
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            f1 = float(f1_score(y_val, preds, zero_division=0))
            thresh = 0.5
        elif name == "Naive_Baseline":
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = np.zeros(len(y_val))
            pr_auc = float(y_val.mean())
            roc_auc = 0.5
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            f1 = float(f1_score(y_val, preds, zero_division=0))
            thresh = 0.5
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            pr_auc = float(average_precision_score(y_val, probs))
            roc_auc = float(roc_auc_score(y_val, probs))
            
            # Find optimal classification threshold maximizing F1 on validation set
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
            best_idx = np.argmax(f1_scores)
            
            if best_idx < len(thresholds):
                thresh = float(thresholds[best_idx])
            else:
                thresh = 0.50
                
            preds = (probs >= thresh).astype(int)
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            f1 = float(f1_score(y_val, preds, zero_division=0))
            
        results[name] = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "threshold": round(thresh, 4)
        }
        
        print(f"{name:<25} | {pr_auc:<8.4f} | {roc_auc:<8.4f} | {prec:<9.4f} | {rec:<8.4f} | {f1:<8.4f}")
        
        # Supervised champion selection based on F1 / PR-AUC
        if name not in ["Naive_Baseline", "Isolation_Forest"] and f1 > best_f1:
            best_f1 = f1
            champion_name = name
            champion_model = model
            optimal_threshold = thresh
            
    print("-" * 75)
    print(f"\n[3/5] Champion Model Selected: {champion_name}")
    print(f"      Validation F1: {best_f1:.4f} (Optimal Decision Threshold: {optimal_threshold:.4f})")
    
    # 4. Save champion model
    print("[4/5] Serializing champion model artifact...")
    joblib.dump(champion_model, os.path.join(models_dir, "best_fraud_model.joblib"))
    
    # 5. Save metadata and metrics
    print("[5/5] Exporting benchmark results to JSON...")
    meta = {
        "champion_model": champion_name,
        "optimal_threshold": round(optimal_threshold, 4),
        "validation_metrics": results
    }
    with open(os.path.join(models_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print("Training phase completed successfully!")
    return champion_model, champion_name, optimal_threshold

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    train_and_benchmark(data_dir, models_dir)
