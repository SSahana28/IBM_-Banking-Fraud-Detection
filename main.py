import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import FraudDetector

def main():
    print("=" * 65)
    print("???  IBM BOB | Banking Fraud Detection System (ML Project)")
    print("=" * 65)
    
    detector = FraudDetector()
    print(f"Loaded Champion Model: {detector.model_name}")
    print(f"Optimal Decision Threshold: {detector.threshold:.4f}\n")
    
    print("Running Sample Transaction Fraud Screening...")
    print("-" * 65)
    
    # Test Case 1: Suspicious Account Liquidation
    tx1 = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 450000.0,
        "oldbalanceOrg": 450000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 450000.0,
        "device_risk_score": 92.0,
        "is_international": 1
    }
    res1 = detector.score_transaction(tx1)
    print("[TEST 1] High-Value Sudden Liquidation:")
    print(f"  - Type: {tx1['type']} | Amount: ?{tx1['amount']:,.2f}")
    print(f"  - Fraud Probability: {res1['fraud_probability']*100:.1f}%")
    print(f"  - Decision: {'?? FLAGGED AS FRAUD' if res1['is_fraud'] else '? APPROVED'}")
    print(f"  - Risk Tier: {res1['risk_tier']}")
    print("  - Risk Factors:")
    for rf in res1['risk_factors']:
        print(f"    * {rf}")
        
    print("-" * 65)
    # Test Case 2: Normal Coffee/Merchant Payment
    tx2 = {
        "step": 14,
        "type": "PAYMENT",
        "amount": 450.0,
        "oldbalanceOrg": 18000.0,
        "newbalanceOrig": 17550.0,
        "oldbalanceDest": 50000.0,
        "newbalanceDest": 50450.0,
        "device_risk_score": 12.0,
        "is_international": 0
    }
    res2 = detector.score_transaction(tx2)
    print("[TEST 2] Everyday Merchant Payment:")
    print(f"  - Type: {tx2['type']} | Amount: ?{tx2['amount']:,.2f}")
    print(f"  - Fraud Probability: {res2['fraud_probability']*100:.1f}%")
    print(f"  - Decision: {'?? FLAGGED AS FRAUD' if res2['is_fraud'] else '? APPROVED'}")
    print(f"  - Risk Tier: {res2['risk_tier']}")
    print("  - Risk Factors:")
    for rf in res2['risk_factors']:
        print(f"    * {rf}")
        
    print("=" * 65)
    print("?? To launch the interactive web dashboard, run:")
    print("    python app.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
