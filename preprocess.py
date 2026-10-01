import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["amount_to_bal_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["is_zero_balance_after"] = (df["newbalanceOrig"] == 0.0).astype(int)
    
    hour = df["step"] % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df

NUMERICAL_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "amount_to_bal_ratio", "is_zero_balance_after",
    "hour_sin", "hour_cos",
    "device_risk_score", "is_international"
]
CATEGORICAL_FEATURES = ["type"]

def build_transformer():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(categories=[["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]], handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

class BankingDataPreprocessor:
    def __init__(self, pipeline=None, feature_names=None):
        self.pipeline = pipeline or build_transformer()
        self.feature_names = feature_names or []
        
    def fit(self, X_train: pd.DataFrame):
        X_eng = engineer_features(X_train)
        self.pipeline.fit(X_eng)
        cat_encoder = self.pipeline.named_transformers_["cat"]
        cat_cols = [f"type_{c}" for c in cat_encoder.categories_[0]]
        self.feature_names = NUMERICAL_FEATURES + cat_cols
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = engineer_features(X)
        return self.pipeline.transform(X_eng)
        
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

def split_and_preprocess(raw_csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv_path)
    
    print(f"[1/4] Loaded {len(df):,} transactions. Checking missing values...")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Missing values detected:\n{null_counts[null_counts > 0]}")
        df = df.dropna()
    else:
        print("Dataset clean: 0 missing or NULL values.")
        
    X = df.drop(columns=["isFraud", "nameOrig", "nameDest"])
    y = df["isFraud"].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"[2/4] Stratified Dataset Splits:")
    print(f"      - Train: {len(X_train):,} ({y_train.sum()} frauds, {y_train.mean()*100:.2f}%)")
    print(f"      - Val:   {len(X_val):,} ({y_val.sum()} frauds, {y_val.mean()*100:.2f}%)")
    print(f"      - Test:  {len(X_test):,} ({y_test.sum()} frauds, {y_test.mean()*100:.2f}%)")
    
    print("[3/4] Fitting feature engineering & scaling pipeline on Train split...")
    preprocessor = BankingDataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    print("[4/4] Saving processed datasets & preprocessor artifact...")
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_proc)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_proc)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_proc)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_proj_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save standard sklearn ColumnTransformer and feature list dictionary
    artifact = {
        "pipeline": preprocessor.pipeline,
        "feature_names": preprocessor.feature_names
    }
    joblib.dump(artifact, os.path.join(models_dir, "preprocessor.joblib"))
    
    test_df = X_test.copy()
    test_df["isFraud"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test_raw.csv"), index=False)
    
    print(f"Preprocessing completed successfully. Features: {len(preprocessor.feature_names)}")
    return preprocessor

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_transactions.csv")
    out_dir = os.path.join(base_dir, "data", "processed")
    split_and_preprocess(raw_path, out_dir)
