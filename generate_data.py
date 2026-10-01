import os
import numpy as np
import pandas as pd

def generate_banking_dataset(n_samples=30000, fraud_rate=0.015, random_state=42):
    """
    Synthesize realistic financial transaction records for fraud detection.
    Patterned after the PaySim benchmark and enterprise banking transaction structures.
    """
    np.random.seed(random_state)
    
    # 1. Transaction IDs & Steps (1 step = 1 hour, over 30 days: 1 to 720)
    steps = np.random.randint(1, 721, size=n_samples)
    
    # 2. Transaction Types
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_probs = [0.35, 0.15, 0.30, 0.05, 0.15]
    tx_types = np.random.choice(types, size=n_samples, p=type_probs)
    
    # 3. Base Amounts by type (Log-normal distribution for realistic monetary skew)
    amounts = np.zeros(n_samples)
    for i, t in enumerate(tx_types):
        if t in ["TRANSFER", "CASH_OUT"]:
            amounts[i] = np.random.lognormal(mean=9.5, sigma=1.8)
        elif t == "PAYMENT":
            amounts[i] = np.random.lognormal(mean=6.5, sigma=1.2)
        elif t == "DEBIT":
            amounts[i] = np.random.lognormal(mean=6.0, sigma=1.0)
        else: # CASH_IN
            amounts[i] = np.random.lognormal(mean=8.5, sigma=1.4)
            
    amounts = np.round(np.clip(amounts, 5.0, 5000000.0), 2)
    
    # 4. Account balances
    old_balance_orig = np.random.lognormal(mean=9.0, sigma=2.0, size=n_samples)
    old_balance_orig = np.round(np.clip(old_balance_orig, 0.0, 10000000.0), 2)
    
    new_balance_orig = np.maximum(0.0, old_balance_orig - amounts)
    new_balance_orig = np.round(new_balance_orig, 2)
    
    old_balance_dest = np.random.lognormal(mean=9.2, sigma=2.1, size=n_samples)
    old_balance_dest = np.round(np.clip(old_balance_dest, 0.0, 15000000.0), 2)
    
    new_balance_dest = old_balance_dest + amounts
    new_balance_dest = np.round(new_balance_dest, 2)
    
    # 5. Device risk & international flag
    device_risk = np.random.beta(a=2, b=8, size=n_samples) * 100 # Skewed toward lower risk
    is_international = np.random.binomial(n=1, p=0.03, size=n_samples)
    
    # 6. Fraud Injections (~fraud_rate)
    is_fraud = np.zeros(n_samples, dtype=int)
    n_frauds = int(n_samples * fraud_rate)
    
    # Fraud predominantly occurs on TRANSFER and CASH_OUT
    eligible_indices = np.where(np.isin(tx_types, ["TRANSFER", "CASH_OUT"]))[0]
    fraud_indices = np.random.choice(eligible_indices, size=n_frauds, replace=False)
    
    for idx in fraud_indices:
        is_fraud[idx] = 1
        # Fraud patterns:
        # A) Empties account completely
        orig_bal = np.random.uniform(50000, 2000000)
        old_balance_orig[idx] = np.round(orig_bal, 2)
        amounts[idx] = np.round(orig_bal, 2)
        new_balance_orig[idx] = 0.0
        
        # B) Destination often has 0 previous balance (mule/new account)
        if np.random.rand() < 0.75:
            old_balance_dest[idx] = 0.0
            new_balance_dest[idx] = amounts[idx] if np.random.rand() < 0.5 else 0.0
            
        # C) Off-hours transactions (night hours 1am-4am)
        hour = np.random.choice([1, 2, 3, 4, 23])
        steps[idx] = (steps[idx] // 24) * 24 + hour
        
        # D) Elevated device risk score
        device_risk[idx] = np.random.uniform(70.0, 98.5)
        
        if np.random.rand() < 0.35:
            is_international[idx] = 1
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "step": steps,
        "type": tx_types,
        "amount": amounts,
        "nameOrig": [f"C{1000000 + i}" for i in range(n_samples)],
        "oldbalanceOrg": old_balance_orig,
        "newbalanceOrig": new_balance_orig,
        "nameDest": [f"M{2000000 + i}" if t == "PAYMENT" else f"C{3000000 + i}" for i, t in enumerate(tx_types)],
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "device_risk_score": np.round(device_risk, 1),
        "is_international": is_international,
        "isFraud": is_fraud
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "raw_transactions.csv")
    
    print("[1/3] Generating synthetic banking transactions dataset...")
    df = generate_banking_dataset(n_samples=30000, fraud_rate=0.015)
    df.to_csv(out_file, index=False)
    
    fraud_cnt = df["isFraud"].sum()
    print(f"[2/3] Successfully saved {len(df):,} records to: {out_file}")
    print(f"[3/3] Class distribution: {len(df) - fraud_cnt:,} Normal, {fraud_cnt:,} Fraud ({df['isFraud'].mean()*100:.2f}%)")
