import os
import json
import joblib
import pandas as pd
import numpy as np

try:
    from src.preprocess import engineer_features, BankingDataPreprocessor
except ImportError:
    from preprocess import engineer_features, BankingDataPreprocessor

class FraudDetector:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_fraud_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.metrics_path = os.path.join(models_dir, "model_metrics.json")
        
        self.model = joblib.load(self.model_path)
        raw_prep = joblib.load(self.preprocessor_path)
        if isinstance(raw_prep, dict):
            self.preprocessor = BankingDataPreprocessor(
                pipeline=raw_prep["pipeline"],
                feature_names=raw_prep["feature_names"]
            )
        else:
            self.preprocessor = raw_prep
            
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        self.threshold = meta.get("optimal_threshold", 0.5)
        self.model_name = meta.get("champion_model", "Unknown")
        
    def score_transaction(self, tx: dict) -> dict:
        df = pd.DataFrame([tx])
        req_cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest", "device_risk_score", "is_international"]
        for col in req_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required transaction field: {col}")
                
        X_proc = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        is_fraud = bool(prob >= self.threshold)
        
        if prob >= 0.85:
            risk_tier = "CRITICAL RISK"
        elif prob >= 0.50:
            risk_tier = "HIGH RISK"
        elif prob >= 0.20:
            risk_tier = "MEDIUM RISK"
        else:
            risk_tier = "LOW RISK"
            
        risk_factors = []
        if tx["type"] in ["TRANSFER", "CASH_OUT"]:
            if tx["newbalanceOrig"] == 0.0 and tx["oldbalanceOrg"] > 0.0:
                risk_factors.append("Origin account completely drained to 0 balance.")
            if tx["amount"] >= 0.9 * tx["oldbalanceOrg"]:
                risk_factors.append(f"High transfer ratio ({tx['amount'] / (tx['oldbalanceOrg'] + 1):.1%}) of total available funds.")
        if tx["oldbalanceDest"] == 0.0 and tx["newbalanceDest"] > 0.0:
            risk_factors.append("Destination account appears newly created / unverified (0 prior balance).")
        if (tx["step"] % 24) in [1, 2, 3, 4, 5]:
            risk_factors.append(f"Off-hours transaction initiated at hour {tx['step'] % 24}:00.")
        if tx.get("device_risk_score", 0) >= 70.0:
            risk_factors.append(f"High device/IP anomaly score ({tx['device_risk_score']}/100).")
        if tx.get("is_international", 0) == 1:
            risk_factors.append("Cross-border / international transaction routing.")
            
        if not risk_factors and not is_fraud:
            risk_factors.append("Transaction matches typical legitimate customer spending patterns.")
            
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": is_fraud,
            "risk_tier": risk_tier,
            "threshold_used": self.threshold,
            "risk_factors": risk_factors
        }
        
    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = df.copy()
        X_proc = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_proc)[:, 1]
        res_df["fraud_probability"] = np.round(probs, 4)
        res_df["is_fraud_predicted"] = (probs >= self.threshold).astype(int)
        res_df["risk_tier"] = pd.cut(
            probs,
            bins=[-0.01, 0.20, 0.50, 0.85, 1.0],
            labels=["LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK"]
        )
        return res_df

if __name__ == "__main__":
    detector = FraudDetector()
    print(f"Loaded FraudDetector with {detector.model_name} (Threshold: {detector.threshold:.4f})\n")
    
    suspicious_tx = {
        "step": 4,
        "type": "TRANSFER",
        "amount": 450000.0,
        "oldbalanceOrg": 450000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 450000.0,
        "device_risk_score": 88.5,
        "is_international": 1
    }
    print("--- Test Case 1: Suspicious Account Liquidation ---")
    print(json.dumps(detector.score_transaction(suspicious_tx), indent=2))
